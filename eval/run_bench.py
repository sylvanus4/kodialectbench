#!/usr/bin/env python3
"""KoDialectBench v0 러너 — T1/T2/T3 를 엔드포인트에 돌리고 **코드가 채점**한다.

⛔ 모델에게 자기 점수를 묻지 않는다. chrF·정답률·방언성은 전부 여기서 계산한다.
⛔ 엔드포인트는 `serve_config_gate.py --profile humanko-prod` 를 통과한 것만 쓴다.
⛔ T2 5지 정답률과 4지(전라+충청 병합) 정답률을 함께 낸다 — 진단용.

과제
  T1 dialect→standard  : 방언 문장을 표준어로. chrF / exact
  T2 region-id         : 방언 문장의 권역. accuracy(5지·4지)
  T3 standard→dialect  : 표준어를 목표 권역 방언으로.
                         dialectness / region_match / adi2 +
                         **의미보존**(생성 vs 입력 표준어 chrF) +
                         **참조근접**(생성 vs 실제 방언 chrF)
                         ⛔ 방언성과 의미보존을 한 점수로 합치지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kit.dialectness import Scorer  # noqa: E402
from kit.glossary import Glossary  # noqa: E402

BENCH = REPO / "eval" / "bench" / "kodialectbench-v0"
REGION_KO = {"gyeongsang": "경상도", "jeolla": "전라도", "chungcheong": "충청도",
             "jeju": "제주도", "gangwon": "강원도"}
MERGE = {"jeolla": "jeolla_chungcheong", "chungcheong": "jeolla_chungcheong"}
PARSED = REPO / "data" / "lane-r" / "parsed"
MARKERS = REPO / "kit" / "markers"
_WS = re.compile(r"\s+")


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """문자 n-gram F-score. sacrebleu 가 없어 직접 계산한다(공백 제거, n=1..6, beta=2)."""
    h, r = _WS.sub("", hyp), _WS.sub("", ref)
    if not h or not r:
        return 0.0
    ps, rs = [], []
    for k in range(1, n + 1):
        hg = collections.Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rg = collections.Counter(r[i:i + k] for i in range(len(r) - k + 1))
        ov = sum((hg & rg).values())
        ps.append(ov / max(sum(hg.values()), 1))
        rs.append(ov / max(sum(rg.values()), 1))
    p, rc = sum(ps) / n, sum(rs) / n
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * rc / (b2 * p + rc)


SYS = "너는 한국어 전문가다. 요청한 것만 정확히 답하고 설명·머리말·따옴표를 붙이지 않는다."
CHEAT = ""
FEWSHOT: dict = {}
STEM: bool | None = None


def prompt_for(task: str, it: dict, gl=None) -> str:
    if task == "T1":
        hint = ""
        if gl is not None:
            hits = gl.lookup(it["dialect"], it["region"])
            if hits:
                # ⛔ 정답 문장을 주지 않는다. **어절 단위 뜻풀이만** 준다.
                hint = ("\n\n참고 어휘(방언 → 표준어):\n" +
                        "\n".join(f"  {a} → {b}" for a, b in hits))
        return ("다음 방언 문장을 표준어로 바꿔라. 바뀐 문장만 출력한다."
                + hint + f"\n\n{it['dialect']}")
    if task == "T2":
        opts = " / ".join(REGION_KO[r] for r in sorted(REGION_KO))
        head = (CHEAT + "\n\n") if CHEAT else ""
        return (head + "다음 문장이 어느 지역 방언인지 하나만 고른다. 지역 이름만 출력한다.\n"
                f"보기: {opts}\n\n{it['dialect']}")
    shots = ""
    if FEWSHOT.get(it["region"]):
        shots = "\n\n예시:\n" + "\n".join(
            f"  표준어: {a}\n  {REGION_KO[it['region']]} 방언: {b}"
            for a, b in FEWSHOT[it["region"]]) + "\n"
    return ("다음 표준어 문장을 " + REGION_KO[it["region"]] +
            " 방언으로 바꿔라. 뜻은 그대로 두고 말투만 바꾼다. 바뀐 문장만 출력한다."
            + shots + f"\n{it['standard']}")


def cheatsheet(top: int = 15) -> str:
    """권역별 대표 표지 목록. ⛔ 정답을 주는 게 아니라 **각 방언이 어떻게 생겼는지**를 준다.
    T2 실패가 '지식 부재'인지 '전라·경상으로 몰아 찍는 prior'인지 가른다."""
    lines = []
    for slug, ko in sorted(REGION_KO.items()):
        f = MARKERS / f"{slug}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        ws = [e["w"] for e in d["eojeol"][:top]]
        sf = [x["suffix"] for x in d["suffix"][:8]]
        lines.append(f"  {ko}: 어휘 {' '.join(ws)} / 어미 {' '.join(sf)}")
    return "각 지역 방언의 대표 표현:\n" + "\n".join(lines)


def fewshot_pool(region: str, k: int, seed: int = 11):
    """⛔ **train 분할**에서만 예시를 뽑는다. test(벤치 문항)에서 뽑으면 정답 누출이다."""
    import random
    rows = []
    for name in ("pairs.jsonl", "pairs-elder.jsonl"):
        f = PARSED / region / name
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for i, ln in enumerate(fh):
                if i > 60000:
                    break
                o = json.loads(ln)
                if o.get("split") == "train" and o["n_dialect_eojeol"] >= 3 \
                        and 8 <= len(o["standard"].split()) <= 22:
                    rows.append((o["standard"], o["dialect"]))
    random.Random(seed).shuffle(rows)
    return rows[:k]


def parse_region(text: str) -> str | None:
    for slug, ko in REGION_KO.items():
        if ko in text or ko[:2] in text:
            return slug
    return None


def call(base: str, model: str, prompt: str, max_tokens: int, timeout: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18101")
    # ⛔ 기본값을 두지 않는다 — 공개 하네스에 사내 모델명이 박히면 안 되고,
    #    쓰는 사람이 자기 모델을 명시하는 편이 맞다.
    ap.add_argument("--model", required=True, help="서빙 중인 모델 이름")
    ap.add_argument("--tasks", nargs="*", default=["T1", "T2", "T3"])
    ap.add_argument("--per-region", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--cheatsheet", action="store_true",
                    help="T2 에 권역별 대표 표지 목록을 붙인다(지식 vs prior 판별)")
    ap.add_argument("--fewshot", type=int, default=0,
                    help="T3 에 train 분할 예시 N개를 붙인다")
    ap.add_argument("--glossary", action="store_true",
                    help="T1 에 방언 어휘 사전 힌트를 붙인다(레인 R 자원)")
    ap.add_argument("--stem", action="store_true", help="공개 사전 어간 조회 ON/OFF 실측용")
    ap.add_argument("--public-glossary", action="store_true",
                    help="⭐ **공개 라이선스 사전만** 쓴다 — 레인 C 에서 재현되는지 본다")
    ap.add_argument("--label", default="base-27b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    local = BENCH / "items.local.jsonl"
    if not local.exists():
        sys.exit(f"{local} 없음 — eval/build_bench.py --materialize 먼저")
    items = [json.loads(l) for l in local.open(encoding="utf-8")]
    byreg: dict[str, list] = collections.defaultdict(list)
    for it in items:
        if len(byreg[it["region"]]) < a.per_region:
            byreg[it["region"]].append(it)
    items = [x for v in byreg.values() for x in v]

    sc = Scorer()
    gl = Glossary(public=True) if a.public_glossary else (Glossary() if a.glossary else None)
    global CHEAT, FEWSHOT, STEM
    STEM = True if a.stem else (False if a.public_glossary else None)
    if a.cheatsheet:
        CHEAT = cheatsheet()
    if a.fewshot:
        FEWSHOT = {r: fewshot_pool(r, a.fewshot) for r in REGION_KO}
        print("few-shot 예시 수:", {r: len(v) for r, v in FEWSHOT.items()})
    jobs = [{"task": t, "it": it} for t in a.tasks for it in items]
    print(f"{len(jobs)} calls  ({len(items)} items x {len(a.tasks)} tasks, conc={a.concurrency})")

    sem = asyncio.Semaphore(a.concurrency)
    loop = asyncio.get_running_loop()
    done = [0]
    t0 = time.time()

    async def one(j):
        async with sem:
            it, task = j["it"], j["task"]
            try:
                out = await loop.run_in_executor(
                    None, call, a.base, a.model, prompt_for(task, it, gl), a.max_tokens, 300)
            except (urllib.error.URLError, OSError, KeyError) as e:
                j["error"] = f"{type(e).__name__}: {e}"
                j["output"] = ""
                return j
            j["output"] = out
            if task == "T1":
                j["chrf"] = round(chrf(out, it["standard"]), 2)
                j["exact"] = int(_WS.sub("", out) == _WS.sub("", it["standard"]))
            elif task == "T2":
                pred = parse_region(out)
                j["pred"] = pred
                j["correct5"] = int(pred == it["region"])
                j["correct4"] = int(MERGE.get(pred, pred) == MERGE.get(it["region"], it["region"]))
            else:
                s = sc.score(out, it["region"])
                j["score"] = asdict(s)
                j["chrf_vs_ref"] = round(chrf(out, it["dialect"]), 2)
                j["chrf_vs_input"] = round(chrf(out, it["standard"]), 2)
                j["ref_dialectness"] = sc.score(it["dialect"], it["region"]).dialectness
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(jobs)}  {time.time()-t0:.0f}s")
            return j

    rows = await asyncio.gather(*[one(j) for j in jobs])
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731

    agg: dict = {}
    for task in a.tasks:
        rs = [x for x in rows if x["task"] == task and not x.get("error")]
        if not rs:
            continue
        per_region: dict = {}
        for reg in sorted(REGION_KO):
            c = [x for x in rs if x["it"]["region"] == reg]
            if not c:
                continue
            if task == "T1":
                per_region[reg] = {"n": len(c), "chrF": mean([x["chrf"] for x in c]),
                                   "exact_pct": round(100 * mean([x["exact"] for x in c]), 1)}
            elif task == "T2":
                per_region[reg] = {"n": len(c),
                                   "acc5_pct": round(100 * mean([x["correct5"] for x in c]), 1),
                                   "acc4_pct": round(100 * mean([x["correct4"] for x in c]), 1)}
            else:
                per_region[reg] = {
                    "n": len(c),
                    "dialectness": mean([x["score"]["dialectness"] for x in c]),
                    "ref_dialectness": mean([x["ref_dialectness"] for x in c]),
                    "recovery_pct": round(100 * mean([x["score"]["dialectness"] for x in c])
                                          / max(mean([x["ref_dialectness"] for x in c]), 1e-9), 1),
                    "region_match": mean([x["score"]["region_match"] for x in c]),
                    "adi2": mean([x["score"]["adi2"] for x in c]),
                    "meaning_chrF_vs_input": mean([x["chrf_vs_input"] for x in c]),
                    "chrF_vs_reference": mean([x["chrf_vs_ref"] for x in c]),
                    "zero_hit_pct": round(100 * sum(1 for x in c if x["score"]["hits"] == 0) / len(c), 1),
                }
        overall = {}
        if task == "T1":
            overall = {"chrF": mean([x["chrf"] for x in rs]),
                       "exact_pct": round(100 * mean([x["exact"] for x in rs]), 1)}
        elif task == "T2":
            overall = {"acc5_pct": round(100 * mean([x["correct5"] for x in rs]), 1),
                       "acc4_pct": round(100 * mean([x["correct4"] for x in rs]), 1),
                       "unparsed_pct": round(100 * sum(1 for x in rs if x["pred"] is None) / len(rs), 1),
                       "confusion": {r: dict(collections.Counter(
                           x["pred"] or "none" for x in rs if x["it"]["region"] == r))
                           for r in sorted(REGION_KO)}}
        else:
            overall = {"dialectness": mean([x["score"]["dialectness"] for x in rs]),
                       "region_match": mean([x["score"]["region_match"] for x in rs]),
                       "adi2": mean([x["score"]["adi2"] for x in rs]),
                       "meaning_chrF_vs_input": mean([x["chrf_vs_input"] for x in rs])}
        agg[task] = {"overall": overall, "per_region": per_region}

    errs = sum(1 for x in rows if x.get("error"))
    out_path = REPO / (a.out or f"data/ledger/bench-{a.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "label": a.label, "model": a.model, "date": time.strftime("%Y-%m-%d"),
        "n_items": len(items), "tasks": a.tasks, "temperature": 0.0,
        "glossary": bool(a.glossary), "public_glossary": bool(a.public_glossary),
        "cheatsheet": bool(a.cheatsheet),
        "fewshot": a.fewshot,
        "errors": errs, "elapsed_sec": round(time.time() - t0, 1),
        "aggregate": agg, "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=1)[:4000])
    print(f"\nerrors={errs} elapsed={time.time()-t0:.0f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
