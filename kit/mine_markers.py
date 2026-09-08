#!/usr/bin/env python3
"""권역 방언 표지를 **데이터에서 캔다** (손으로 쓰지 않는다).

왜 손으로 안 쓰나 — 2026-09-07 실측: 손으로 쓴 정규식 22종을 Emilia 한국어 1,060문장에
돌렸더니 5건 히트에 **전부 오탐**이었다("밝혔다"의 `혔`, "자신의 아이가"의 `아이가`).
어휘 직관은 정밀도를 못 만든다. 그래서 AI Hub 의 어절 단위 `isDialect` 라벨에서
`방언어절 → 표준어절` 대응을 세고, **순도(purity)** 와 **권역특이도**를 계산해서 고른다.

산출
  kit/markers/<region>.json   {eojeol: [...], suffix: [...], lexicon: {방언:표준}}
  kit/markers/summary.json

정의
  purity(w)        = n_dialect(w) / (n_dialect(w) + n_standard(w))
                     표준어 쪽 코퍼스에도 흔한 형태를 걸러낸다. 이게 없으면 오탐이 난다.
  specificity(w,r) = n_r(w) / Σ_r' n_r'(w)
                     여러 권역에 공통인 형태(범방언)와 그 권역만의 형태를 가른다.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PARSED = REPO / "data" / "lane-r" / "parsed"
OUT = REPO / "kit" / "markers"


def load_region(region: str):
    d = PARSED / region
    lex = collections.Counter()
    lex_map: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    # 본편(118~122)과 중·노년층(71517/71558)을 **같은 권역으로 합친다**.
    # 따로 두면 권역특이도 분모가 쪼개져 표지가 억울하게 탈락한다.
    for name in ("lexicon.jsonl", "lexicon-elder.jsonl"):
        lf = d / name
        if not lf.exists():
            continue
        with lf.open(encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                lex[o["dialect"]] += o["n"]
                lex_map[o["dialect"]][o["standard"]] += o["n"]
    std = collections.Counter()
    for name in ("pairs.jsonl", "pairs-elder.jsonl"):
        pf = d / name
        if not pf.exists():
            continue
        with pf.open(encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                if o.get("split") == "test":     # ⛔ 홀드아웃은 순도 계산에도 안 쓴다
                    continue
                for w in o["standard"].split():
                    std[w] += 1
    return lex, lex_map, std


def tail_rule(dialect: str, standard: str) -> tuple[str, str] | None:
    """앞부분이 같고 꼬리만 다르면 (방언꼬리, 표준꼬리) 를 낸다 — 어미 규칙 후보."""
    i = 0
    while i < min(len(dialect), len(standard)) and dialect[i] == standard[i]:
        i += 1
    if i == 0:
        return None                        # 어두부터 다르면 어미가 아니다
    dt, st = dialect[i:], standard[i:]
    if not dt or len(dt) > 4 or len(st) > 4:
        return None
    return dt, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=20, help="표지 최소 등장 횟수")
    ap.add_argument("--min-purity", type=float, default=0.95)
    ap.add_argument("--min-spec", type=float, default=0.60, help="권역특이도 하한")
    ap.add_argument("--top", type=int, default=4000)
    a = ap.parse_args()

    regions = sorted(p.name for p in PARSED.iterdir()
                     if p.is_dir() and ((p / "lexicon.jsonl").exists()
                                        or (p / "lexicon-elder.jsonl").exists()))
    if not regions:
        sys.exit(f"no parsed region under {PARSED} — corpus/aihub_parse.py 먼저")

    per, std_all = {}, collections.Counter()
    for r in regions:
        lex, lex_map, std = load_region(r)
        per[r] = (lex, lex_map)
        std_all += std
        print(f"load {r}: dialect types={len(lex):,} tokens={sum(lex.values()):,} "
              f"standard types={len(std):,}")

    total = collections.Counter()
    for lex, _ in per.values():
        total += lex
    # 표준어 어절의 끝 1~4글자 빈도를 **한 번만** 만든다.
    # (예전엔 꼬리마다 std_all 전체를 endswith 로 훑어서 O(400 x |V|) 였다 — 분당 단위로 느렸다)
    std_tail: collections.Counter = collections.Counter()
    for w, c in std_all.items():
        for k in range(1, 5):
            if len(w) >= k:
                std_tail[w[-k:]] += c

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in regions:
        lex, lex_map = per[r]
        eoj, suf_num = [], collections.Counter()
        suf_den = collections.Counter()
        lexicon = {}
        for w, n in lex.most_common():
            if n < a.min_n:
                break
            purity = n / (n + std_all.get(w, 0))
            spec = n / max(total[w], 1)
            if purity < a.min_purity or spec < a.min_spec:
                continue
            eoj.append({"w": w, "n": n, "purity": round(purity, 4), "spec": round(spec, 4)})
            lexicon[w] = lex_map[w].most_common(1)[0][0]
            tr = tail_rule(w, lexicon[w])
            if tr:
                suf_num[tr] += n
            if len(eoj) >= a.top:
                break
        # 꼬리 규칙: 방언꼬리가 표준 코퍼스 어절 말미에 얼마나 흔한지로 순도 재계산
        suffixes = []
        for (dt, st), n in suf_num.most_common(400):
            if n < a.min_n:
                continue
            base = std_tail.get(dt, 0)
            p = n / (n + base)
            if p >= a.min_purity:
                suffixes.append({"suffix": dt, "standard": st, "n": n, "purity": round(p, 4)})
        payload = {
            "region": r,
            "provenance": {
                "lane": "R",
                "derived_from": "AI Hub 한국어 방언 발화 데이터 (어절 isDialect 라벨)",
                "note": ("⛔ 이 표지 목록은 AI Hub 라벨에서 파생됐다. 레인 C(공개 가중치)에서는 "
                         "**채점·필터로만** 쓰고 학습 텍스트로 흘리지 않는다. "
                         "파생성 판단은 계획서 §5.1 미결정 #2 를 따른다."),
            },
            "eojeol": eoj, "suffix": suffixes, "lexicon": lexicon,
            "params": {"min_n": a.min_n, "min_purity": a.min_purity,
                       "min_spec": a.min_spec}}
        (OUT / f"{r}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        row = {"region": r, "eojeol_markers": len(eoj), "suffix_rules": len(suffixes),
               "lexicon": len(lexicon),
               "top10": [e["w"] for e in eoj[:10]],
               "top_suffix": [s["suffix"] for s in suffixes[:10]]}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
