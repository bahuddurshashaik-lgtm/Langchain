"""Phase 4 end-to-end gate.

Exercises the exact code path the Streamlit app uses — real PDF, real Gemini
calls, map-reduce — plus the error states, and exits non-zero on any failure.

Usage:
    python scripts/e2e_check.py [--pages 12] [--model MODEL] [--style brief]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf_processor import PDFProcessor  # noqa: E402
from summarizer import PDFSummarizer  # noqa: E402

LONG_PARAGRAPH = (
    "Section {n}. Retrieval augmented generation improves factual accuracy by grounding every "
    "answer in retrieved passages. The retriever selects candidates using vector similarity, "
    "the reranker orders them, and the language model writes an answer that cites the evidence "
    "it was given. Evaluation must cover retrieval quality and answer quality separately, "
    "because a strong writer can hide a weak retriever. Operationally, teams monitor latency, "
    "cost per answer, grounding rate and refusal rate. A refusal that names the missing "
    "evidence is a success, not a failure."
)

PASS, FAIL = "PASS", "FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{PASS if condition else FAIL}] {label}{f' — {detail}' if detail else ''}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--model", default=None)
    parser.add_argument("--style", default="brief")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    from tests.conftest import build_pdf

    results = []
    print(f"== Phase 4 end-to-end check ({args.pages} pages, style={args.style}) ==\n")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "long_report.pdf"
        paragraphs = [LONG_PARAGRAPH.format(n=i) for i in range(1, args.pages + 1)]
        build_pdf(pdf_path, paragraphs=paragraphs, title="Long Report")
        print(f"fixture: {pdf_path.name} ({pdf_path.stat().st_size} bytes)\n")

        # 1. extraction and chunking
        processor = PDFProcessor()
        with pdf_path.open("rb") as handle:
            processed = processor.prepare(handle, max_tokens=args.max_tokens)
        results.append(
            check(
                "PDF extracted and chunked",
                processed["num_chunks"] >= 2,
                f"{processed['metadata']['num_pages']} pages, {processed['num_tokens']} tokens, "
                f"{processed['num_chunks']} chunks",
            )
        )
        results.append(
            check(
                "every chunk within the token limit",
                all(processor.count_tokens(c) <= args.max_tokens for c in processed["chunks"]),
            )
        )

        # 2. real map-reduce
        summarizer = PDFSummarizer(model=args.model, max_retries=2)
        started = time.perf_counter()
        outcome = summarizer.summarize_document(
            processed["chunks"],
            summary_type=args.style,
            full_text=processed["text"],
            include_analysis=True,
            include_quotes=True,
        )
        elapsed = time.perf_counter() - started

        results.append(
            check(
                "all chunks summarized",
                outcome["failed_chunks"] == 0
                and outcome["successful_chunks"] == outcome["total_chunks"],
                f"{outcome['successful_chunks']}/{outcome['total_chunks']} in "
                f"{outcome['map_elapsed_seconds']}s",
            )
        )
        results.append(
            check(
                "combined summary produced",
                len(outcome["combined_summary"].strip()) > 80,
                f"{len(outcome['combined_summary'])} chars, total run {elapsed:.1f}s",
            )
        )
        results.append(
            check(
                "structure analysis succeeded",
                outcome.get("analysis", {}).get("status") == "success",
            )
        )
        results.append(
            check(
                "quotes extracted",
                bool(outcome.get("quotes")),
                f"{len(outcome.get('quotes', []))} quotes",
            )
        )

        print("\n-- combined summary --")
        print(" ".join(outcome["combined_summary"].split())[:700])
        print()

    # 3. error states
    with tempfile.TemporaryDirectory() as tmp:
        blank = build_pdf(Path(tmp) / "blank.pdf", paragraphs=[" "])
        try:
            with blank.open("rb") as handle:
                PDFProcessor().prepare(handle)
            results.append(check("blank PDF rejected", False))
        except ValueError as exc:
            results.append(check("blank PDF rejected", "No extractable text" in str(exc)))

    try:
        bad = PDFSummarizer(model="gemini-0.5-does-not-exist", max_retries=0)
        bad.summarize_text("hello")
        results.append(check("unknown model rejected", False))
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        results.append(
            check(
                "unknown model rejected",
                "not found" in message or "404" in message or "invalid" in message,
            )
        )

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    print("PHASE 4 RESULT:", PASS if passed == len(results) else FAIL)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
