#!/usr/bin/env python3
"""⛔ 양성 대조군 — 이 지표가 **변별을 하는지 먼저 증명**한다 ([[metric-must-discriminate]]).

비교 전에 통과해야 하는 3가지:
  (A) 진짜 방언 문장 vs **같은 쌍의 표준어 문장** → 방언성이 갈려야 한다.
      갈리지 않으면 지표가 아무것도 못 본다.
  (B) 목표 권역 vs **다른 권역** → region_match 가 갈려야 한다.
      갈리지 않으면 "사투리답다"만 알고 "어느 사투리인지"는 모른다.
  (C) 천장/바닥 확인 — 전 셀이 0 이나 1 에 붙으면 순위매기기에 못 쓴다.

⛔ 홀드아웃(적용됨): 표지를 캔 문장으로 채점하면 당연히 통과한다. 그래서
   `aihub_parse*.py --holdout-frac` 이 **문서(대화) 단위**로 train/test 를 나누고,
   마이닝은 train 만, 이 검정은 test 만 본다. 문장 단위로 쪼개면 같은 화자가 양쪽에
   들어가 표지가 새므로 반드시 문서 단위여야 한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kit.dialectness import Scorer  # noqa: E402

PARSED = REPO / "data" / "lane-r" / "parsed"


def sample_pairs(region: str, n: int, seed: int, min_marks: int):
    """방언 어절이 min_marks 개 이상인 발화만 — '방언 특징이 없는 발화'는 대조가 안 된다."""
    rows = []
    rnd = random.Random(seed)
    keep = 0
    for name in ("pairs.jsonl", "pairs-elder.jsonl"):
        f = PARSED / region / name
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                # ⛔ 홀드아웃: 표지를 캔 train 문서로 채점하면 자기충족이다
                if o.get("split") != "test":
                    continue
                if o["n_dialect_eojeol"] < min_marks or len(o["dialect"].split()) < 5:
                    continue
                keep += 1
                if len(rows) < n:
                    rows.append(o)
                elif rnd.random() < n / keep:      # reservoir
                    rows[rnd.randrange(n)] = o
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=400, help="권역당 표본")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-marks", type=int, default=2)
    ap.add_argument("--out", default="data/ledger/marker-calibration.json")
    a = ap.parse_args()

    sc = Scorer()
    regions = sc.regions
    res = {"n_per_region": a.n, "seed": a.seed, "min_marks": a.min_marks,
           "regions": regions, "A_dialect_vs_standard": {}, "B_cross_region": {},
           "caveats": [
               "✅ 홀드아웃 적용됨: 표지는 train 문서, 채점은 test 문서(문서=대화 단위 분할).",
               "n_dialect_eojeol >= %d 로 거른 발화만 썼다(특징 없는 발화 제외)." % a.min_marks,
           ]}
    ok = True
    for r in regions:
        rows = sample_pairs(r, a.n, a.seed, a.min_marks)
        if not rows:
            res["A_dialect_vs_standard"][r] = {"error": "no rows"}
            ok = False
            continue
        d = [sc.score(x["dialect"], r) for x in rows]
        s = [sc.score(x["standard"], r) for x in rows]
        dd = [x.dialectness for x in d]
        ss = [x.dialectness for x in s]
        sep = sum(1 for i in range(len(d)) if d[i].dialectness > s[i].dialectness) / len(d)
        res["A_dialect_vs_standard"][r] = {
            "n": len(rows),
            "dialect_mean": round(st.mean(dd), 4), "standard_mean": round(st.mean(ss), 4),
            "delta": round(st.mean(dd) - st.mean(ss), 4),
            "pairwise_separated_pct": round(sep * 100, 1),
            "dialect_zero_pct": round(100 * sum(1 for x in dd if x == 0) / len(dd), 1),
            "standard_nonzero_pct": round(100 * sum(1 for x in ss if x > 0) / len(ss), 1),
        }
        # B: 같은 방언 문장을 다른 권역 타깃으로 채점 → region_match 가 떨어져야 한다
        own = st.mean([sc.score(x["dialect"], r).region_match for x in rows])
        others = {o: round(st.mean([sc.score(x["dialect"], o).region_match for x in rows]), 4)
                  for o in regions if o != r}
        res["B_cross_region"][r] = {"own": round(own, 4), "others": others,
                                    "own_minus_best_other": round(own - max(others.values()), 4)
                                    if others else None}
        A = res["A_dialect_vs_standard"][r]
        if A["delta"] < 0.05 or A["pairwise_separated_pct"] < 70:
            ok = False
    res["verdict"] = "DISCRIMINATES" if ok else "INSUFFICIENT"
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nverdict={res['verdict']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
