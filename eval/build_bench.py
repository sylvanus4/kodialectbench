#!/usr/bin/env python3
"""KoDialectBench v0 빌더 — 한국어 방언 능력 벤치마크.

⛔ **본문을 저장하지 않는다.** 문항은 AI Hub 라벨에서 나오는데 AI Hub 이용정책은
   미승인 제3자 제공을 금지한다(계획서 §4.1). 그래서 벤치는 **문항 id + 과제 명세 +
   채점 코드**만 배포하고, 본문은 이용자가 자기 AI Hub 계정으로 받아 `materialize` 한다.
   licensed corpus 위에 얹는 벤치의 통상적 방식이고, 이렇게 해야 공개할 수 있다.

⛔ **홀드아웃 test 문서에서만 뽑는다.** 표지를 캔 train 문서에서 문항을 뽑으면
   우리 지표로 우리 문항을 채점하는 자기충족이 된다.

과제 3종 (v0)
  T1 dialect→standard   방언 문장을 표준어로 옮긴다        → chrF / exact
  T2 region-id          방언 문장의 권역을 고른다(5지선다) → accuracy
  T3 standard→dialect   표준어를 목표 권역 방언으로 쓴다   → dialectness x region_match + 의미보존

⚠️ 정정(2026-09-07): 전라·충청이 구분되지 않던 것은 **언어적 사실이 아니라 데이터 부족**
   이었다. 전라 본편이 275k 발화뿐이라 표지가 125개였고 권역 margin 이 −0.227 이었는데,
   중·노년층(71558, 전라 161,859 문서)을 병합하니 표지 2,752개 · 분리도 96.5% ·
   margin +0.622 로 뒤집혔다. **"구조적으로 못 나눈다"는 해석을 철회한다.**
   4지 병합 변형은 진단용으로 남겨 두되, 기본은 5지다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PARSED = REPO / "data" / "lane-r" / "parsed"
OUT = REPO / "eval" / "bench" / "kodialectbench-v0"
REGIONS = ["gangwon", "gyeongsang", "jeolla", "chungcheong", "jeju"]
MERGED = {"jeolla": "jeolla_chungcheong", "chungcheong": "jeolla_chungcheong"}


def iter_test(region: str, min_marks: int, min_words: int, max_words: int):
    for name in ("pairs.jsonl", "pairs-elder.jsonl"):
        f = PARSED / region / name
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                if o.get("split") != "test":
                    continue
                if o["n_dialect_eojeol"] < min_marks:
                    continue
                w = len(o["dialect"].split())
                if not (min_words <= w <= max_words):
                    continue
                o["_src"] = name
                yield o


def digest(o: dict) -> str:
    """본문 없이 문항을 식별·검증하기 위한 지문. materialize 가 이 값으로 대조한다."""
    return hashlib.blake2b((o["standard"] + "\x1f" + o["dialect"]).encode(),
                           digest_size=10).hexdigest()


def iid(region: str, o: dict) -> str:
    """⛔ 배포용 문항 식별자 — AI Hub 의 원 id 를 절대 그대로 내보내지 않는다.

    원 id 는 `st_set1_collectorgw95_speakergw2539_62_11` 처럼 **수집자·화자 식별자**를
    담고 있다. 그대로 공개하면 (1) AI Hub 파생 식별자의 국외 반출이고(계획서 §4.1)
    (2) 화자 식별자라 개인정보 축도 걸린다. 그래서 단방향 해시로 바꿔 내보내고,
    materialize 는 같은 해시를 로컬에서 계산해 원 문장과 맞춘다.
    """
    return hashlib.blake2b(f"{region}\x1f{o['id']}".encode(), digest_size=8).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--min-marks", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--materialize", action="store_true",
                    help="본문을 채워 로컬 실행용 파일을 만든다(⛔ 배포 금지)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    picked: dict[str, list] = {}
    for r in REGIONS:
        pool = list(iter_test(r, a.min_marks, a.min_words, a.max_words))
        if not pool:
            print(f"⚠️ {r}: test 문장 0 — --holdout-frac 으로 재파싱했는지 확인")
            picked[r] = []
            continue
        rnd.shuffle(pool)
        picked[r] = pool[: a.per_region]
        print(f"{r:<13} pool={len(pool):>7,}  picked={len(picked[r])}")

    items = []
    for r, rows in picked.items():
        for o in rows:
            items.append({
                "iid": iid(r, o), "region": r, "src": o["_src"],
                "digest": digest(o),
                "n_dialect_eojeol": o["n_dialect_eojeol"],
                "n_words": len(o["dialect"].split()),
            })
    spec = {
        "name": "KoDialectBench", "version": "0.1.0",
        "built": "2026-09-07", "seed": a.seed,
        "license_note": ("문항 본문은 포함하지 않는다. AI Hub 이용정책상 미승인 제3자 제공이 "
                         "금지되므로, 이용자가 자기 계정으로 AI Hub 118~122·71517·71558 을 받아 "
                         "corpus/aihub_parse*.py 로 파싱한 뒤 `--materialize` 로 채운다."),
        "split": "test (문서=대화 단위 홀드아웃, blake2b(doc_id)%10000/10000 < 0.2)",
        "filters": {"min_dialect_eojeol": a.min_marks,
                    "min_words": a.min_words, "max_words": a.max_words},
        "tasks": {
            "T1_dialect_to_standard": {
                "input": "dialect", "target": "standard",
                "metric": ["chrF", "exact_match"],
                "note": "방언 이해 축. 의미보존을 직접 잰다."},
            "T2_region_id": {
                "input": "dialect", "target": "region",
                "choices5": REGIONS,
                "choices4_merged": sorted(set(MERGED.get(r, r) for r in REGIONS)),
                "metric": ["accuracy"],
                "note": ("기본은 5지. 4지(전라+충청 병합)는 진단용 보조 지표다. "
                         "2026-09-07 이전 표기 '전라·충청 구분 불가'는 데이터 부족이 원인이었고 "
                         "중·노년층 병합으로 해소됐다(margin −0.227 → +0.622).")},
            "T3_standard_to_dialect": {
                "input": "standard + target_region", "target": "dialect",
                "metric": ["dialectness", "region_match", "adi2", "semantic_preservation"],
                "note": ("생성 축. ⛔ dialectness 와 semantic_preservation 을 한 점수로 합치지 "
                         "마라 — 방언성이 오르고 의미가 무너지면 실패다.")},
        },
        "metric_validity": {
            "measured_2026_09_07_holdout_n400": {
                "jeolla": 96.5, "gangwon": 92.5, "chungcheong": 90.5,
                "gyeongsang": 86.8, "jeju": 86.5},
            "region_margin": {"jeju": 0.784, "gangwon": 0.705, "jeolla": 0.622,
                              "gyeongsang": 0.587, "chungcheong": 0.560},
            "caveat": ("쌍별 분리도(%). 표지=train 문서 / 채점=test 문서(문서 단위 홀드아웃). "
                       "5권역 전부 게이트(70%) 통과, 권역 margin 전부 양수.")},
        "counts": {r: len(v) for r, v in picked.items()},
        "n_items": len(items),
    }
    (OUT / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nspec + {len(items)} items (본문 없음) -> {OUT}")

    if a.materialize:
        idx = {}
        for r in REGIONS:
            for o in iter_test(r, a.min_marks, a.min_words, a.max_words):
                idx[(r, iid(r, o))] = o
        n_ok = 0
        with (OUT / "items.local.jsonl").open("w", encoding="utf-8") as fh:
            for it in items:
                o = idx.get((it["region"], it["iid"]))
                if not o or digest(o) != it["digest"]:
                    continue
                n_ok += 1
                fh.write(json.dumps({**it, "standard": o["standard"],
                                     "dialect": o["dialect"]}, ensure_ascii=False) + "\n")
        print(f"materialized {n_ok}/{len(items)} -> items.local.jsonl  ⛔ 배포 금지")
    return 0


if __name__ == "__main__":
    sys.exit(main())
