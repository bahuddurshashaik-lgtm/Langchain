"""Generate a sample PDF for manual end-to-end runs.

Usage:
    python scripts/make_sample_pdf.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import build_pdf  # noqa: E402


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.pdf")
    build_pdf(target, title="LangChain PDF Summarizer Sample")
    print(f"Wrote {target.resolve()} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
