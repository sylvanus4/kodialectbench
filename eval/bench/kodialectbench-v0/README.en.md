# KoDialectBench v0.1.0

A benchmark for Korean **regional dialects**: Gangwon, Gyeongsang, Jeolla, Chungcheong and
Jeju. Two hundred items per region, one thousand items in total.

No Korean dialect benchmark existed. So this project, which set out to build a dialect model,
built the ruler before the model. If the training fails, the ruler still stands.

## Getting started

The code and the item list live in the `satoori-ko` repository under
`eval/bench/kodialectbench-v0/`. Run every command below **from the repository root
(`satoori-ko/`)**.

All you need is Python 3.10 or newer (verified on 3.12.8). **There are no third-party
dependencies** — the scorer, the item builder and the dialectness scorer run on the standard
library alone. There is nothing to install.

Downloading the AI Hub data requires an account and an API key. Register at AI Hub, **file a
usage request for datasets 118, 119, 120, 121, 122, 71517 and 71558 and wait for approval**,
then issue an API key from your account page and export it.

```bash
export AIHUB_APIKEY=your-key
```

## Why the item text is missing

This repository holds **no item text.** It holds item ids, content digests and the scoring
code. The source is AI Hub data, and the AI Hub terms forbid passing the data to third
parties who have not been approved. So we removed the text: you download the source under
your own account and fill in the same items yourself.

Everyone who does this gets the **same 1,000 items.** Item selection is pinned by a random
seed and a document-level split, and the digests let you confirm you have the same items.

```bash
# 1) Download AI Hub 118/119/120/121/122 (+ 71517/71558) label data under your own account
python corpus/aihub_fetch.py --elder      # ⛔ without --elder, 71517/71558 are not
                                          #    fetched and parse_elder dies below
python corpus/aihub_parse.py
python corpus/aihub_parse_elder.py --holdout-frac 0.2

# 2) Regenerate the marker lexicon (⛔ we do not ship this either — see below)
python kit/mine_markers.py --min-n 10 --min-purity 0.70 --min-spec 0.30 --top 20000

# 3) Fill in the same items locally
python eval/build_bench.py --per-region 200 --materialize
```

Check your regenerated lexicon against the sha256 values in
`eval/bench/MARKERS.sha256.json`. Mining is deterministic: the same input gives the same
bytes.

## Three tasks

**T1, comprehension.** Rewrite a dialect sentence in standard Korean. This measures directly
whether the model understood it. Scored with chrF and exact match.

**T2, region identification.** Pick which of the five regions a sentence comes from.

**T3, generation.** Rewrite a standard Korean sentence in a given regional dialect. Scored on
dialectness, region_match, their product (ADI2), and whether the meaning survived.
⛔ Never merge dialectness and meaning preservation into one score. A sentence that looks
like dialect but has lost its meaning is a failure, not a success.

## Running the evaluation

The scorer calls your model as an **OpenAI-compatible endpoint**. You do not hand it a file
of generations; you give it a server address and a model name, and it sends the items and
scores the replies.

```bash
python eval/run_bench.py \
    --base http://127.0.0.1:8000 \   # point this at your own serving endpoint
    --model your-model-name \
    --per-region 200 \
    --label your-run-name \
    --out results.json
```

The result JSON carries per-task totals alongside per-region scores. When comparing with
others, report **the serving configuration**, not just the score: engine and precision
(bf16, fp8, and so on), context length, and how many runs you aggregated and how. Our
baseline is the median of three runs on one engine, because `temperature=0` does not give
you the same number twice.

## Evidence that the ruler has markings

Before using the ruler we checked that it has markings. We fed it a dialect sentence and its
own standard-Korean counterpart, side by side, and asked whether it separates them. Four
hundred pairs per region.

| Region | dialect vs standard separation | region margin |
|---|---|---|
| Jeolla | 96.5% | +0.622 |
| Gangwon | 92.5% | +0.705 |
| Chungcheong | 90.5% | +0.560 |
| Gyeongsang | 86.8% | +0.587 |
| Jeju | 86.5% | +0.784 |

You can rebuild this table yourself.

```bash
python eval/marker_calibration.py -n 400
```

All five regions pass, and every region margin is positive. Markers are mined from
conversations on the training side and scoring uses only conversations on the test side, so
no speaker appears on both.

One lesson came out of this. At first Gyeongsang and Jeolla did not separate well, and we
explained that as a linguistic fact: that Gyeongsang differs mostly in intonation, which
transcription does not capture. **That was wrong.** The sample was simply too small. Adding
the elder-speaker corpus moved Jeolla from 76.8% to 96.5%. When a metric cannot tell two
things apart, look at the sample size before reaching for "they are just similar."

## Baseline (Qwen3.8-27B)

Measured on one engine, changing only the weights, three runs per arm, median reported.

| Metric | base | trained on synthetic | trained on real data |
|---|---|---|---|
| T1 chrF | 50.7 | 70.1 | 80.1 |
| T1 exact match % | 2.2 | 10.6 | 24.7 |
| T2 region id % | 24.7 | 63.1 | 79.4 |
| T3 dialectness | 0.034 | 0.102 | 0.141 |
| (reference) real speaker dialectness | 0.232 | 0.232 | 0.232 |
| T3 ADI2 | 0.012 | 0.065 | 0.107 |

The base model barely writes dialect. On the same items the real speaker sentences score 0.232
for dialectness while the base model scores 0.034, about 1.4 in ten. Giving it three in-context examples helped unevenly across regions and
made Gangwon worse. **Dialect is something a model learns, not something it is told.**

## What not to trust

The items are transcribed text. Intonation and sound are not in here. Sound carries a large
part of what a dialect is, and this ruler measures only what survives in writing.

Source volume is uneven across regions. Jeolla is especially thin in the main corpus and
leans heavily on the elder-speaker data.

The dialectness score depends on a marker lexicon mined from AI Hub labels. A dialect word
absent from that lexicon scores zero. Read the score as a lower bound, not a ceiling.

## Source

Items derive from AI Hub's Korean dialect speech datasets (118, 119, 120, 121, 122) and the
elder-speaker Korean dialect speech datasets (71517, 71558). No item text lives here.

The **marker lexicon (`kit/markers/*.json`) is not shipped either.** It too is derived from AI
Hub labels. You build it the same way you build the items: run `kit/mine_markers.py` on your own
download. Mining is deterministic, so the same input yields the same lexicon, and
`eval/bench/MARKERS.sha256.json` lets you confirm it matches ours.

⇒ As a result **this repository contains no AI Hub text and no original item identifiers** —
only code, plus one-way hashes and counts (`items.jsonl`) for verifying the items you regenerate.
Those hashes and counts are computed from AI Hub data, so they follow the AI Hub terms of use (see License).

## License

**The code is Apache-2.0** (see `LICENSE`). `eval/bench/PUBLISH.json` is the authoritative
list of what counts as the code.

**Item ids and digests (`items.jsonl`) are derived from AI Hub data**, so they follow the **AI
Hub terms of use**, not Apache-2.0. The marker lexicon is not distributed at all (you
regenerate it). Model weights trained on AI Hub data are not published.

## Citation

Cite the benchmark version (`KoDialectBench v0.1.0`) together with **the serving configuration
you measured on**. Without the engine, precision, context length, repeat count and aggregation,
the number cannot be compared with anyone else's.

## Contact

Report item errors, scoring anomalies and reproduction failures as issues on this
repository.
