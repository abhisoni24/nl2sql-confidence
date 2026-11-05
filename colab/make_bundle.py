"""Build colab/nl2sql_confidence_code.zip: the code the Colab notebook unpacks.

Includes configs, src, scripts, tests, colab helpers and docs; excludes data/,
results/, secrets and caches. Writes colab/BUNDLE_INFO.txt (build time and a
SHA-256 fingerprint of every bundled file) into the zip, so each Colab run can
record exactly which code produced it.

    python colab/make_bundle.py
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["configs", "src", "scripts", "tests", "colab", "pyproject.toml", "environment.yml",
           "Makefile", "README.md", "PLAN.md", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".secrets"}   # anywhere in the path
SKIP_TOP = {"data", "results"}                               # only as top-level folders
OUT = ROOT / "colab" / "nl2sql_confidence_code.zip"


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        candidates = [p] if p.is_file() else sorted(p.rglob("*"))
        for f in candidates:
            rel = f.relative_to(ROOT)
            if (f.is_file() and not SKIP_PARTS & set(rel.parts) and rel.parts[0] not in SKIP_TOP
                    and f != OUT
                    and f.suffix not in {".pyc", ".zip"} and "egg-info" not in str(rel)):
                out.append(f)
    return out


def main() -> None:
    fs = files()
    digest = hashlib.sha256()
    for f in fs:
        digest.update(str(f.relative_to(ROOT)).encode())
        digest.update(f.read_bytes())
    info = (f"nl2sql_confidence code bundle\nbuilt: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"files: {len(fs)}\nsha256: {digest.hexdigest()}\n")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in fs:
            z.write(f, Path("nl2sql_confidence") / f.relative_to(ROOT))
        z.writestr("nl2sql_confidence/colab/BUNDLE_INFO.txt", info)
    print(info + f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
