#!/usr/bin/env python3
"""AI Hub 방언 데이터 — **라벨링데이터(텍스트)만** 내려받는다.

⛔ 레인 격리: 받은 것은 전부 `data/lane-r/` 아래로만 간다(계획서 §5.1).
   레인 C(공개 가중치) 생성 코드는 이 경로를 읽지 못한다.
⛔ 원천데이터(음성)는 권역당 ~290GB 다. 이 스크립트는 **절대** 받지 않는다.

실측 근거(2026-09-07 파일트리): 5개 권역 라벨 합계 1.37GB, 중·노년층 2종 포함 시 10.1GB.

gotcha
------
- AI Hub `info/*.do` 는 **본문이 정상인데 HTTP 502** 를 준다. 상태코드로 판정하지 마라
  (urllib 은 502 에서 예외를 던져 본문을 버린다 — curl 로 받아 본문을 봐야 한다).
- 다운로드는 `down/0.6/<datasetkey>.do?fileSn=<쉼표목록>` + 헤더 `apikey:<키>`.
- 반환물은 tar 이고, 내부에 `.part0..N` 조각이 있으면 이어붙여야 한다(aihubshell 과 동일).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PARENT = REPO.parent
LANE_R = REPO / "data" / "lane-r" / "aihub"
BASE = "https://api.aihub.or.kr"

# datasetkey -> (권역 slug, 라벨링데이터 fileSn)   ※ 2026-09-07 파일트리 실측
LABELS: dict[str, tuple[str, str]] = {
    "118": ("gangwon", "572551,572589"),
    "119": ("gyeongsang", "572701,572713"),
    "120": ("jeolla", "572615,572654"),
    "121": ("jeju", "572660,572694"),
    "122": ("chungcheong", "572595,572609,572612"),
    "71517": (
        "elder-gangwon-gyeongsang",
        "538315,538310,538311,538312,538313,538314,"
        "538322,538323,538324,538325,538326,538327",
    ),
    "71558": (
        "elder-chung-jeolla-jeju",
        "572524,572525,572526,572527,572528,572529,572530,572531,572532,"
        "572542,572543,572544,572545,572546,572547,572548,572549,572550",
    ),
}
CORE = ["118", "119", "120", "121", "122"]
ELDER = ["71517", "71558"]


def api_key() -> str:
    """환경변수 우선, 없으면 부모 레포 .env. 값은 절대 로그에 찍지 않는다."""
    k = os.environ.get("AIHUB_APIKEY")
    if k:
        return k.strip()
    env_path = PARENT / ".env"
    if not env_path.exists():
        sys.exit(f"key not found: set AIHUB_APIKEY or add it to {env_path}")
    m = re.search(r"^AI-HUB-APIKEY=(.+)$", env_path.read_text(errors="ignore"), re.M)
    if not m:
        sys.exit("key not found in parent .env")
    return m.group(1).strip().strip("\"'")


def merge_parts(root: pathlib.Path) -> int:
    """`<name>.partNN` 조각을 이어붙인다. aihubshell 이 하는 일과 동일."""
    merged = 0
    groups: dict[pathlib.Path, list[pathlib.Path]] = {}
    for p in root.rglob("*.part*"):
        if re.search(r"\.part\d+$", p.name):
            groups.setdefault(p.with_suffix("").with_name(re.sub(r"\.part\d+$", "", p.name)), []).append(p)
    for target, parts in groups.items():
        parts.sort(key=lambda x: int(re.search(r"\.part(\d+)$", x.name).group(1)))
        with open(target, "wb") as out:
            for part in parts:
                out.write(part.read_bytes())
                part.unlink()
        merged += 1
    return merged


def fetch(dsk: str, region: str, filesn: str, key: str) -> int:
    out = LANE_R / region
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".done").exists():
        print(f"skip   {dsk} {region} (already done)")
        return 0
    tar = out / f"{dsk}.tar"
    url = f"{BASE}/down/0.6/{dsk}.do?fileSn={filesn}"
    n = filesn.count(",") + 1
    print(f"fetch  {dsk} {region}  ({n} files)")
    rc = subprocess.call(
        [
            "curl", "-L", "-C", "-", "-o", str(tar),
            "-H", f"apikey:{key}",
            "-w", "       http=%{http_code} bytes=%{size_download} time=%{time_total}s\n",
            url,
        ]
    )
    if rc != 0 or not tar.exists() or tar.stat().st_size < 10_000:
        head = tar.read_bytes()[:400].decode("utf-8", "replace") if tar.exists() else "(no file)"
        # ⛔ 실패 tar 를 남기면 다음 재시도의 `-C -` 가 그 바이트에서 이어받으려다 curl rc=33
        #    (Range 미지원)으로 죽는다. 실패분은 지운다 (2026-09-07 전라도에서 실측).
        tar.unlink(missing_ok=True)
        print(f"FAIL   rc={rc}\n{head}")
        return 1
    subprocess.check_call(["tar", "-xf", str(tar)], cwd=out)
    tar.unlink()
    m = merge_parts(out)
    (out / ".done").write_text("ok\n")
    files = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"ok     {region}: {files} files (merged {m} split sets)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="datasetkey 목록 (기본: 5개 권역 본편)")
    ap.add_argument("--elder", action="store_true", help="중·노년층 2종(~9GB)도 받는다")
    ap.add_argument("--list", action="store_true", help="대상만 출력하고 끝낸다")
    a = ap.parse_args()

    keys = a.only or list(CORE)
    if a.elder and not a.only:
        keys += ELDER
    if a.list:
        for k in keys:
            print(k, *LABELS[k])
        return 0

    key = api_key()
    bad = sum(fetch(k, LABELS[k][0], LABELS[k][1], key) for k in keys)
    print(f"\n{'FAILED %d' % bad if bad else 'ALL OK'} -> {LANE_R}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
