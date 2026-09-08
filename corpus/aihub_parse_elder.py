#!/usr/bin/env python3
"""AI Hub **중·노년층** 방언 라벨(71517/71558) 파서 — 본편(118~122)과 스키마가 다르다.

본편: `utterance[].eojeolList[].isDialect` (어절 라벨이 이미 있다)
중·노년층: `transcription.{dialect,standard,pronunciation}` (문서 단위 전문, 어절 라벨 없음)
  → 어절 라벨이 없으므로 **difflib 로 정렬해서** 방언↔표준 어절 대응을 뽑는다.

권역은 **zip 파일 이름**에 있다(`TL_02._경상도_...`). 문서 안에는 `residenceProvince`
(gs/gw/...) 도 있지만, 두 권역이 한 데이터셋에 섞여 있으므로 파일명이 1차 근거다.

⛔ 본편 파서와 **출력 경로를 공유**한다(`data/lane-r/parsed/<region>/`) — 다만
`--suffix` 로 파일을 나눠 두고, 표지 마이닝이 둘 다 읽게 한다. 같은 권역 데이터를
서로 다른 "권역"으로 취급하면 권역특이도 계산이 망가진다(elder-* 를 그대로 두면 그렇게 된다).
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import pathlib
import re
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "lane-r" / "aihub"
OUT = REPO / "data" / "lane-r" / "parsed"

REGION_BY_NAME = {"강원": "gangwon", "경상": "gyeongsang", "전라": "jeolla",
                  "제주": "jeju", "충청": "chungcheong"}
ELDER_DIRS = ("elder-gangwon-gyeongsang", "elder-chung-jeolla-jeju")


def region_of(zip_path: pathlib.Path) -> str | None:
    for ko, slug in REGION_BY_NAME.items():
        if ko in zip_path.name:
            return slug
    return None


def split_of(doc_id: str, holdout: float) -> str:
    if holdout <= 0:
        return "train"
    h = int(hashlib.blake2b(doc_id.encode(), digest_size=8).hexdigest(), 16)
    return "test" if (h % 10000) / 10000 < holdout else "train"


def align(std: str, dia: str):
    """표준/방언 어절열을 정렬해 바뀐 자리만 뽑는다. 1:1 치환만 취한다 —
    n:m 은 어미 규칙이 아니라 문장 재구성이라 사전으로 못 쓴다."""
    a, b = std.split(), dia.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                s, d = a[i1 + k], b[j1 + k]
                if s != d and len(d) <= 12:
                    out.append((d, s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--limit-zip", type=int)
    a = ap.parse_args()

    zips = []
    for d in ELDER_DIRS:
        p = SRC / d
        if p.exists() and (p / ".done").exists():
            zips += sorted(p.rglob("*.zip"))
    if a.limit_zip:
        zips = zips[: a.limit_zip]
    if not zips:
        sys.exit("no completed elder dataset — corpus/aihub_fetch.py --only 71517 71558")

    per_pairs: dict[str, object] = {}
    per_lex: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stat: collections.Counter = collections.Counter()
    handles = {}
    try:
        for z in zips:
            reg = region_of(z)
            if not reg:
                stat["zip_unmapped"] += 1
                continue
            if reg not in handles:
                (OUT / reg).mkdir(parents=True, exist_ok=True)
                handles[reg] = (OUT / reg / "pairs-elder.jsonl").open("w", encoding="utf-8")
            fh_out = handles[reg]
            try:
                zf = zipfile.ZipFile(z)
            except zipfile.BadZipFile:
                stat["bad_zip"] += 1
                continue
            with zf:
                for n in zf.namelist():
                    if not n.lower().endswith(".json"):
                        continue
                    try:
                        doc = json.loads(zf.read(n).decode("utf-8"))
                    except Exception:
                        stat["bad_json"] += 1
                        continue
                    t = doc.get("transcription") or {}
                    std = (t.get("standard") or "").strip()
                    dia = (t.get("dialect") or "").strip()
                    stat[f"{reg}_docs"] += 1
                    if not std or not dia:
                        stat["missing_side"] += 1
                        continue
                    sp = split_of(doc.get("fileName") or n, a.holdout_frac)
                    marks = align(std, dia)
                    if sp == "train":
                        for d_, s_ in marks:
                            per_lex[reg][(d_, s_)] += 1
                    stat["identical" if std == dia else "differing"] += 1
                    spk = (doc.get("speaker") or [{}])[0]
                    fh_out.write(json.dumps({
                        "id": doc.get("fileName"), "region": reg, "split": sp,
                        "standard": std, "dialect": dia,
                        "pronunciation": (t.get("pronunciation") or "").strip() or None,
                        "n_dialect_eojeol": len(marks),
                        "speaker": {"sex": spk.get("gender"), "birth_year": spk.get("birthYear"),
                                    "province": spk.get("residenceProvince"),
                                    "job": spk.get("job")},
                        "source": "elder",
                    }, ensure_ascii=False) + "\n")
    finally:
        for fh in handles.values():
            fh.close()

    for reg, lex in per_lex.items():
        with (OUT / reg / "lexicon-elder.jsonl").open("w", encoding="utf-8") as fh:
            for (d_, s_), n in lex.most_common():
                fh.write(json.dumps({"dialect": d_, "standard": s_, "n": n},
                                    ensure_ascii=False) + "\n")
    res = {"zips": len(zips), "holdout_frac": a.holdout_frac, **dict(stat),
           "lexicon_entries": {r: len(v) for r, v in per_lex.items()}}
    (OUT / "summary-elder.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
