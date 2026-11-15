"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF

PARAGRAPHS = [
    "Retrieval augmented generation combines a retriever with a language model. "
    "The retriever finds relevant passages and the model answers using them. "
    "This reduces hallucination because every claim is grounded in retrieved text.",
    "Chunking is the least glamorous part of the pipeline and the most common source "
    "of silent quality loss. Chunks that cut a sentence in half produce summaries that "
    "describe fragments instead of arguments.",
    "Map-reduce summarization handles documents that exceed the context window. "
    "Each chunk is summarized independently, then the partial summaries are merged "
    "into one final answer. Cost grows linearly with document length.",
]


def build_pdf(path: Path, paragraphs: list[str] = PARAGRAPHS, title: str = "Sample Report") -> Path:
    """Write a small multi-page PDF with real text content."""
    pdf = FPDF()
    pdf.set_title(title)
    pdf.set_author("Phase 1 Test Suite")
    pdf.set_subject("PDF processor fixtures")
    pdf.set_auto_page_break(auto=True, margin=15)
    for paragraph in paragraphs:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, paragraph)
    pdf.output(str(path))
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "sample.pdf")


@pytest.fixture
def sample_pdf_file(sample_pdf: Path):
    with sample_pdf.open("rb") as handle:
        yield handle
