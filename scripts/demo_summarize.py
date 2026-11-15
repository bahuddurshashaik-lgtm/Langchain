"""Phase 2 proof-of-life: real Gemini calls through every chain.

Usage:
    python scripts/demo_summarize.py

Costs a handful of small requests (roughly 1k tokens total).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import STYLES  # noqa: E402
from summarizer import PDFSummarizer  # noqa: E402

TEXT = (
    "Retrieval augmented generation pairs a retriever with a language model. The retriever "
    "selects passages that are likely to answer the question; the model writes an answer "
    "grounded in those passages. Because every claim can be traced to retrieved text, "
    "hallucination rates drop sharply compared with an ungrounded model. The trade-off is "
    "latency and cost: each question now requires a search step plus a longer prompt. "
    "Teams that skip evaluation often ship systems that look correct but cite the wrong "
    "passage, so retrieval quality must be measured separately from answer quality."
)


def main() -> int:
    summarizer = PDFSummarizer()
    print(f"model: {summarizer.model}\n")

    failures = 0
    for style in STYLES:
        try:
            summary = summarizer.summarize_text(TEXT, summary_type=style)
            preview = " ".join(summary.split())
            print(f"[{style}]\n{preview[:220]}\n")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{style}] FAILED: {exc}\n")

    chunks = [TEXT[:400], TEXT[400:]]
    result = summarizer.summarize_document(
        chunks, "brief", include_analysis=True, include_quotes=True
    )
    print(f"chunks: {result['successful_chunks']}/{result['total_chunks']} ok")
    print(f"combined: {' '.join(result['combined_summary'].split())[:220]}")
    print(f"analysis status: {result['analysis']['status']}")
    print(f"quotes found: {len(result['quotes'])}")

    print("\nPHASE 2 RESULT:", "PASSED" if failures == 0 else f"FAILED ({failures} styles)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
