#!/usr/bin/env python3
"""방언 어휘 사전(glossary) — T1(방언→표준어)의 copy-through 를 메우는 검색 자원.

왜 웹검색이 아니라 이 사전이 1순위인가
--------------------------------------
1. **정확하다**: AI Hub 라벨의 `방언어절 → 표준어절` 정답 쌍에 빈도가 붙어 있다.
   웹검색은 잡글을 읽어 추정하는 것이라 정밀도가 떨어진다.
2. **빠르다**: 메모리 조회 0ms vs 쿼리당 수백 ms~수 초.
3. ⛔ **벤치 오염을 피한다**: AI Hub 파생본이 HF 에 미러돼 있어(junyeong-nero·Junhoee)
   웹검색이 T1 정답을 그대로 긁어올 수 있다. 그러면 T1 점수가 무의미해진다.

⛔ 사전은 **train 분할 문서에서만** 캔 것이다(`lexicon*.jsonl`). 벤치 문항은 test 분할이라
   문항 자체가 사전에 들어 있지는 않다. 다만 **같은 어절이 train 에도 나오면 조회된다** —
   그게 사전의 목적이므로 누출이 아니라 의도다. 이 구분을 결과에 명시한다.
⛔ 이 사전은 AI Hub 파생 = **레인 R**. 레인 C(공개 가중치) 제품에 그대로 실으려면
   계획서 §5.1 미결정 #2 판단이 필요하다.

웹검색 폴백은 `--web` 일 때만 켠다(`scripts/skills/websearch.py`, DuckDuckGo).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
from difflib import SequenceMatcher

REPO = pathlib.Path(__file__).resolve().parents[1]
PARSED = REPO / "data" / "lane-r" / "parsed"
OUT = REPO / "kit" / "glossary.json"
_PUNCT = re.compile(r"[.,!?~…\"')\]（）\[]")


def build(min_n: int = 3) -> dict:
    """권역별 방언어절 → 표준어절 (빈도 1위). 전 권역 통합 사전도 만든다."""
    per: dict[str, dict[str, str]] = {}
    allc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for d in sorted(p for p in PARSED.iterdir() if p.is_dir()):
        c: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for name in ("lexicon.jsonl", "lexicon-elder.jsonl"):
            f = d / name
            if not f.exists():
                continue
            with f.open(encoding="utf-8") as fh:
                for ln in fh:
                    o = json.loads(ln)
                    if o["n"] < min_n or o["dialect"] == o["standard"]:
                        continue
                    # ⛔ 정렬 잡음 제거. difflib 로 뽑은 중·노년층 대응에는
                    #    `소주 -> 마어더기` 같은 **완전히 무관한 쌍**이 섞인다(실측).
                    #    방언형은 표준형의 음운 변이라 보통 글자를 상당히 공유한다.
                    if SequenceMatcher(a=o["dialect"], b=o["standard"]).ratio() < 0.34:
                        continue
                    c[o["dialect"]][o["standard"]] += o["n"]
                    allc[o["dialect"]][o["standard"]] += o["n"]
        per[d.name] = {k: v.most_common(1)[0][0] for k, v in c.items()}
        print(f"  {d.name:<13} {len(per[d.name]):>7,} 항목")
    per["_all"] = {k: v.most_common(1)[0][0] for k, v in allc.items()}
    print(f"  {'통합':<13} {len(per['_all']):>7,} 항목")
    return per


PUBLIC = REPO / "kit" / "glossary-public.json"


class Glossary:
    public = False

    def __init__(self, path: pathlib.Path = OUT, public: bool = False):
        """public=True 면 **공개 라이선스 사전**만 쓴다(레인 C). AI Hub 파생을 섞지 않는다."""
        self.public = public
        if public:
            if not PUBLIC.exists():
                raise SystemExit(f"{PUBLIC} 없음 — corpus/public_glossary_build.py 먼저")
            raw = json.loads(PUBLIC.read_text(encoding="utf-8"))
            per: dict[str, dict[str, str]] = collections.defaultdict(dict)
            for w, v in raw.items():
                std = v.get("standard") if isinstance(v, dict) else v
                if not std or std == w:
                    continue
                per[v.get("region", "unknown") if isinstance(v, dict) else "unknown"][w] = std
                per["_all"][w] = std
            self.d = dict(per)
            return
        if not path.exists():
            raise SystemExit(f"{path} 없음 — kit/glossary.py build 먼저")
        self.d = json.loads(path.read_text(encoding="utf-8"))

    def _probe(self, tbl: dict, fb: dict, k: str, stem: bool):
        """정확일치 우선. stem=True 면 꼬리를 1~3자 떼어 어간으로 재조회한다.

        ⛔ 공개 사전은 **표제어(사전형)** 중심인데 우리 텍스트는 **활용형 어절**이다.
           `고뱅이`는 잡아도 `벵원에`·`나오더거`는 못 잡는다(2026-09-07 실측, chrF +3.4 에 그침).
           그래서 어간 조회를 붙인다. ⛔ 어간이 2자 미만이면 오탐이 폭발하므로 자른다.
        """
        v = tbl.get(k) or fb.get(k)
        if v and v != k:
            return k, v
        if not stem:
            return None
        for cut in (1, 2, 3):
            st = k[:-cut]
            if len(st) < 2:
                break
            v = tbl.get(st) or fb.get(st)
            if v and v != st:
                return st, v
        return None

    def lookup(self, text: str, region: str | None = None, limit: int = 12,
               stem: bool | None = None):
        """문장에서 사전에 있는 방언 어절만 뽑아 (방언, 표준) 목록으로 준다."""
        if stem is None:
            # ⛔ **기본 OFF.** 어간 조회를 켜면 실측 chrF 56.2 -> 53.6 (전 권역 하락).
            #    `가지고 -> 가지 -> 처마` 같은 오탐이 추가 커버리지를 상쇄하고 남는다.
            #    공개 사전이 못 잡는 `벵원에`·`나오더거` 는 **형태가 아니라 어휘 문제**다
            #    (음운 변이형 자체가 사전에 없다) — 어간으로는 애초에 안 풀린다.
            #    해법은 사전 확장이 아니라 **음운 규칙**이다(2026-09-07 실측).
            stem = False
        tbl = self.d.get(region or "_all", self.d["_all"])
        fb = self.d["_all"]
        hits, seen = [], set()
        for tok in text.split():
            k = _PUNCT.sub("", tok)
            if len(k) < 2 or k in seen:
                continue
            got = self._probe(tbl, fb, k, stem)
            if got:
                hits.append(got)
                seen.add(k)
            if len(hits) >= limit:
                break
        return hits

    def misses(self, text: str, region: str | None = None):
        """사전에 없는 어절 — 웹검색 폴백 대상."""
        tbl = self.d.get(region or "_all", self.d["_all"])
        fb = self.d["_all"]
        out = []
        for tok in text.split():
            k = _PUNCT.sub("", tok)
            if len(k) >= 2 and k not in tbl and k not in fb:
                out.append(k)
        return out


def web_lookup(word: str, timeout: int = 25) -> str | None:
    """⛔ 폴백 전용. 벤치 채점에 쓰면 오염 위험이 있으니 결과에 사용 여부를 명시한다."""
    ws = REPO.parent / "scripts" / "skills" / "websearch.py"
    if not ws.exists():
        return None
    try:
        r = subprocess.run([sys.executable, str(ws), f"{word} 사투리 표준어 뜻"],
                           capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()[:400] or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "test"])
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--region")
    ap.add_argument("text", nargs="?")
    a = ap.parse_args()
    if a.cmd == "build":
        per = build(a.min_n)
        OUT.write_text(json.dumps(per, ensure_ascii=False), encoding="utf-8")
        print(f"-> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
        return 0
    g = Glossary()
    txt = a.text or sys.stdin.read()
    print(json.dumps({"hits": g.lookup(txt, a.region), "misses": g.misses(txt, a.region)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
