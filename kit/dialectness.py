#!/usr/bin/env python3
"""방언성 스코어러 — `방언성 × 목표권역일치` (아랍어 ADI2 방식).

⛔ 단일 스칼라로 합치되 **두 항을 항상 병기**한다. 방언성만 높고 권역이 틀리면 실패다.
⛔ 이건 **게이트**이지 목적함수가 아니다 — 보상/재랭킹/후보선택 점수로 쓰지 마라
   (Human-KO `humanize_gate` 불변식 승계).

표지는 `kit/markers/<region>.json` (데이터에서 캔 것). 손으로 쓰지 않는다.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, asdict

MARKERS = pathlib.Path(__file__).resolve().parent / "markers"
_HANGUL = re.compile(r"[가-힣]")


@dataclass
class Score:
    dialectness: float      # 방언 어절 비율 (0~1)
    region_match: float     # 히트 중 목표 권역 비율 (0~1). 히트 0 이면 0.0
    adi2: float             # dialectness * region_match
    n_eojeol: int
    hits: int
    hit_regions: dict       # {region: n}
    examples: list


class Scorer:
    def __init__(self, regions: list[str] | None = None):
        files = sorted(MARKERS.glob("*.json"))
        # ⛔ 표지 파일만 받는다. markers/ 에 다른 json 이 섞여도 죽지 않아야 한다.
        keep = []
        for f in files:
            try:
                if "eojeol" in json.loads(f.read_text(encoding="utf-8")):
                    keep.append(f)
            except Exception:
                pass
        files = keep
        if regions:
            files = [f for f in files if f.stem in regions]
        if not files:
            raise SystemExit(f"no marker files in {MARKERS} — kit/mine_markers.py 먼저")
        self.eojeol: dict[str, dict[str, float]] = {}
        self.suffix: dict[str, list[str]] = {}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            self.eojeol[d["region"]] = {e["w"]: e["purity"] for e in d["eojeol"]}
            self.suffix[d["region"]] = [s["suffix"] for s in d["suffix"]]
        self.regions = list(self.eojeol)

    def score(self, text: str, target: str | None = None) -> Score:
        words = [w for w in text.split() if _HANGUL.search(w)]
        n = len(words)
        per = {r: 0 for r in self.regions}
        ex: list = []
        hits = 0
        for w in words:
            key = w.strip(".,!?~…\"'()[]")
            fired = [r for r in self.regions if key in self.eojeol[r]]
            if not fired:
                fired = [r for r in self.regions
                         if any(key.endswith(s) for s in self.suffix[r])]
            if fired:
                hits += 1
                for r in fired:
                    per[r] += 1
                if len(ex) < 12:
                    ex.append({"w": key, "regions": fired})
        dialectness = hits / n if n else 0.0
        if target and hits:
            region_match = per.get(target, 0) / sum(per.values())
        else:
            region_match = 0.0
        return Score(round(dialectness, 4), round(region_match, 4),
                     round(dialectness * region_match, 4), n, hits,
                     {k: v for k, v in per.items() if v}, ex)


def main() -> int:
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--target")
    a = ap.parse_args()
    s = Scorer()
    txt = a.text or sys.stdin.read()
    print(json.dumps(asdict(s.score(txt, a.target)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
