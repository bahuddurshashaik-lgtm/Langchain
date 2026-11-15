"""Tests for the PDF extraction, cleaning and chunking pipeline."""

from __future__ import annotations

import pytest

from pdf_processor import PDFProcessor
from tests.conftest import PARAGRAPHS, build_pdf


@pytest.fixture
def processor() -> PDFProcessor:
    return PDFProcessor()


class TestCleanText:
    def test_removes_control_characters(self, processor: PDFProcessor) -> None:
        assert processor.clean_text("a\x00b\x07c\x1fd") == "abcd"

    def test_keeps_punctuation_and_newlines(self, processor: PDFProcessor) -> None:
        cleaned = processor.clean_text("Line one.\n\nLine two!")
        assert cleaned == "Line one.\n\nLine two!"

    def test_collapses_runs_of_spaces_and_tabs(self, processor: PDFProcessor) -> None:
        assert processor.clean_text("too     many\t\ttabs") == "too many tabs"

    def test_collapses_excess_blank_lines(self, processor: PDFProcessor) -> None:
        assert processor.clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_surrounding_whitespace(self, processor: PDFProcessor) -> None:
        assert processor.clean_text("  \n hello \n  ") == "hello"


class TestCountTokens:
    def test_non_empty_text_has_tokens(self, processor: PDFProcessor) -> None:
        assert processor.count_tokens("hello world") > 0

    def test_empty_text_is_zero(self, processor: PDFProcessor) -> None:
        assert processor.count_tokens("") == 0

    def test_longer_text_has_more_tokens(self, processor: PDFProcessor) -> None:
        short = processor.count_tokens("one sentence")
        long = processor.count_tokens("one sentence " * 20)
        assert long > short


class TestChunkText:
    def test_short_text_is_single_chunk(self, processor: PDFProcessor) -> None:
        assert processor.chunk_text("Just one short sentence.") == ["Just one short sentence."]

    def test_empty_text_raises(self, processor: PDFProcessor) -> None:
        with pytest.raises(ValueError, match="No extractable text"):
            processor.chunk_text("   \n\t  ")

    def test_non_positive_max_tokens_raises(self, processor: PDFProcessor) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            processor.chunk_text("text", max_tokens=0)

    def test_every_chunk_respects_the_limit(self, processor: PDFProcessor) -> None:
        text = " ".join(PARAGRAPHS) * 5
        chunks = processor.chunk_text(text, max_tokens=60)
        assert len(chunks) > 1
        assert all(processor.count_tokens(chunk) <= 60 for chunk in chunks)

    def test_no_content_is_lost(self, processor: PDFProcessor) -> None:
        text = " ".join(PARAGRAPHS)
        chunks = processor.chunk_text(text, max_tokens=50)
        assert " ".join(chunks).split() == text.split()

    def test_oversized_sentence_is_hard_split(self, processor: PDFProcessor) -> None:
        sentence = "word " * 400 + "end."
        chunks = processor.chunk_text(sentence, max_tokens=100)
        assert len(chunks) > 1
        assert all(processor.count_tokens(chunk) <= 100 for chunk in chunks)

    def test_prefers_sentence_boundaries(self, processor: PDFProcessor) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = processor.chunk_text(text, max_tokens=12)
        assert len(chunks) >= 2
        assert all(chunk.strip().endswith(".") for chunk in chunks)


class TestPdfReading:
    def test_extracts_text_from_all_pages(self, processor: PDFProcessor, sample_pdf_file) -> None:
        text = processor.extract_text_from_pdf(sample_pdf_file)
        assert "Retrieval augmented generation" in text
        assert "Map-reduce summarization" in text

    def test_metadata_is_read(self, processor: PDFProcessor, sample_pdf_file) -> None:
        meta = processor.get_pdf_metadata(sample_pdf_file)
        assert meta["num_pages"] == len(PARAGRAPHS)
        assert meta["title"] == "Sample Report"
        assert meta["author"] == "Phase 1 Test Suite"

    def test_metadata_error_is_reported_not_raised(self, processor: PDFProcessor) -> None:
        class Broken:
            def seek(self, *_):
                raise OSError("cannot seek")

        assert "error" in processor.get_pdf_metadata(Broken())

    def test_invalid_pdf_raises_runtime_error(self, processor: PDFProcessor, tmp_path) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf at all")
        with bad.open("rb") as handle, pytest.raises(RuntimeError, match="Error reading PDF"):
            processor.extract_text_from_pdf(handle)


class TestPrepare:
    def test_prepare_returns_text_chunks_and_stats(
        self, processor: PDFProcessor, sample_pdf_file
    ) -> None:
        result = processor.prepare(sample_pdf_file, max_tokens=100)

        assert result["num_chunks"] == len(result["chunks"])
        assert result["num_chunks"] >= 1
        assert result["num_tokens"] > 0
        assert result["num_characters"] == len(result["text"])
        assert result["metadata"]["num_pages"] == len(PARAGRAPHS)
        assert all(processor.count_tokens(chunk) <= 100 for chunk in result["chunks"])

    def test_prepare_handles_a_blank_pdf(self, processor: PDFProcessor, tmp_path) -> None:
        blank = build_pdf(tmp_path / "blank.pdf", paragraphs=[" "])
        with blank.open("rb") as handle, pytest.raises(ValueError, match="No extractable text"):
            processor.prepare(handle)
