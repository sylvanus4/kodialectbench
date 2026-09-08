#!/usr/bin/env python3
"""AI Hub 방언 라벨(zip) → 문장 병렬쌍 + 어절 대응 사전.

AI Hub 라벨은 우리가 만들려던 것을 **이미 갖고 있다**:
  utterance[] = {form, standard_form, dialect_form, eojeolList[{eojeol, standard, isDialect}]}
즉 **어절 단위로 방언 여부가 라벨링**돼 있다. 그래서 diff 를 뜰 필요가 없다 —
`isDialect:true` 인 어절만 모으면 그게 곧 `방언어절 → 표준어절` 대응 사전이다.

⛔ 레인 R 전용. 산출물도 `data/lane-r/parsed/` 아래에만 쓴다.
⛔ zip 을 통째로 풀지 않는다(수 GB). zipfile 로 스트리밍한다.

출력
  data/lane-r/parsed/<region>/pairs.jsonl    문장 병렬쌍 + 화자 메타
  data/lane-r/parsed/<region>/lexicon.jsonl  어절 대응 (dialect, standard, n)
  data/lane-r/parsed/<region>/stats.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "lane-r" / "aihub"
OUT = REPO / "data" / "lane-r" / "parsed"
SPEAKER_KEYS = ("age", "sex", "occupation", "birthplace",
                "principal_residence", "current_residence", "education")


def split_of(doc_id: str, holdout: float) -> str:
    """⛔ 화자 단위 홀드아웃. 문장 단위로 쪼개면 같은 대화·같은 화자가 양쪽에 들어가
    표지가 새고, 검정이 낙관적으로 나온다. 문서(대화) id 를 해시해 고정 분할한다."""
    if holdout <= 0:
        return "train"
    h = int(hashlib.blake2b(doc_id.encode(), digest_size=8).hexdigest(), 16)
    return "test" if (h % 10000) / 10000 < holdout else "train"


def parse_json_bytes(raw: bytes, region: str, pairs, lex, stat, holdout: float) -> None:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        stat["bad_json"] += 1
        return
    spk = {s.get("id"): {k: s.get(k) for k in SPEAKER_KEYS} for s in doc.get("speaker", []) or []}
    split = split_of(str(doc.get("id", "")), holdout)
    stat[f"doc_{split}"] += 1
    for u in doc.get("utterance", []) or []:
        std = (u.get("standard_form") or "").strip()
        dia = (u.get("dialect_form") or "").strip()
        stat["utt"] += 1
        eo = u.get("eojeolList") or []
        marks = [(e.get("eojeol"), e.get("standard")) for e in eo
                 if e.get("isDialect") and e.get("eojeol") and e.get("standard")
                 and e.get("eojeol") != e.get("standard")]
        if split == "train":                    # ⛔ 표지는 train 문서에서만 캔다
            for d, s in marks:
                lex[(d, s)] += 1
        if not std or not dia:
            stat["missing_side"] += 1
            continue
        if std == dia:
            stat["identical"] += 1          # 방언 특징이 없는 발화 — 대조군으로 쓸 수 있다
        else:
            stat["differing"] += 1
        pairs.write(json.dumps({
            "id": u.get("id"), "region": region, "split": split,
            "standard": std, "dialect": dia,
            "n_dialect_eojeol": len(marks),
            "speaker": spk.get(u.get("speaker_id"), {}),
        }, ensure_ascii=False) + "\n")


def run_region(region: str, limit_zip: int | None, holdout: float) -> dict:
    src = SRC / region
    zips = sorted(src.rglob("*.zip"))
    if limit_zip:
        zips = zips[:limit_zip]
    if not zips:
        return {"region": region, "error": "no zip found"}
    out = OUT / region
    out.mkdir(parents=True, exist_ok=True)
    lex: collections.Counter = collections.Counter()
    stat = collections.Counter()
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as pairs:
        for z in zips:
            try:
                zf = zipfile.ZipFile(z)
            except zipfile.BadZipFile:
                stat["bad_zip"] += 1
                continue
            with zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".json")]
                stat["json_files"] += len(names)
                for n in names:
                    parse_json_bytes(zf.read(n), region, pairs, lex, stat, holdout)
    with (out / "lexicon.jsonl").open("w", encoding="utf-8") as fh:
        for (d, s), n in lex.most_common():
            fh.write(json.dumps({"dialect": d, "standard": s, "n": n}, ensure_ascii=False) + "\n")
    res = {"region": region, "zips": len(zips), "holdout_frac": holdout, **dict(stat),
           "lexicon_entries": len(lex),
           "lexicon_tokens": sum(lex.values())}
    (out / "stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", help="기본: data/lane-r/aihub 아래 .done 인 것 전부")
    ap.add_argument("--limit-zip", type=int, help="권역당 zip 개수 제한(스모크)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="검정용으로 뗄 **문서(대화)** 비율. 표지는 train 문서에서만 캔다")
    a = ap.parse_args()
    regions = a.regions or sorted(p.name for p in SRC.iterdir()
                                  if p.is_dir() and (p / ".done").exists())
    if not regions:
        sys.exit(f"no completed region under {SRC}")
    allres = []
    for r in regions:
        res = run_region(r, a.limit_zip, a.holdout_frac)
        allres.append(res)
        print(json.dumps(res, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
