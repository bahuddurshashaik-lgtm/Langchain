"""Tests for the display, export and statistics helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from utils import (
    available_styles,
    build_export_text,
    chunk_dataframe,
    export_csv_bytes,
    export_text_bytes,
    file_hash,
    format_bytes,
    is_api_key_configured,
    safe_filename,
    style_label,
    summary_metrics,
)

TABLE_COLUMNS = ["chunk", "ok", "original_chars", "summary_chars", "compression"]


@pytest.fixture
def result() -> dict:
    return {
        "individual_summaries": [
            {
                "chunk_number": 1,
                "summary": "first summary",
                "ok": True,
                "original_length": 200,
                "summary_length": 50,
            },
            {
                "chunk_number": 2,
                "summary": "Error processing chunk: boom",
                "ok": False,
                "original_length": 100,
                "summary_length": 0,
            },
        ],
        "combined_summary": "the final summary",
        "total_chunks": 2,
        "successful_chunks": 1,
        "failed_chunks": 1,
        "summary_type": "brief",
        "model": "gemini-3.1-flash-lite",
        "map_elapsed_seconds": 1.5,
    }


@pytest.fixture
def processed() -> dict:
    return {
        "text": "some text",
        "chunks": ["some", "text"],
        "num_chunks": 2,
        "num_tokens": 123,
        "num_characters": 9,
        "metadata": {"num_pages": 3, "title": "Report", "author": "Ada", "subject": "Demo"},
    }


class TestHashAndNames:
    def test_hash_is_stable(self) -> None:
        assert file_hash(b"abc") == file_hash(b"abc")

    def test_hash_differs_for_different_content(self) -> None:
        assert file_hash(b"abc") != file_hash(b"abd")

    def test_hash_is_short(self) -> None:
        assert len(file_hash(b"abc")) == 16

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Q3 Report.pdf", "Q3_Report"),
            ("../../etc/passwd", "etc_passwd"),
            ("report.PDF", "report"),
            ("..hidden..pdf", "hidden"),
        ],
    )
    def test_safe_filename(self, name: str, expected: str) -> None:
        assert safe_filename(name) == expected

    def test_safe_filename_falls_back(self) -> None:
        assert safe_filename("...pdf") == "document"
        assert safe_filename("") == "document"


class TestFormatting:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [(512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_format_bytes(self, size: int, expected: str) -> None:
        assert format_bytes(size) == expected

    def test_api_key_detection(self) -> None:
        assert is_api_key_configured("abc")
        assert not is_api_key_configured("   ")
        assert not is_api_key_configured(None)

    def test_style_label_and_fallback(self) -> None:
        assert style_label("brief").startswith("Brief")
        assert style_label("unknown") == "unknown"

    def test_available_styles_match_prompts(self) -> None:
        assert available_styles() == ("brief", "detailed", "bullet_points", "executive")


class TestStatistics:
    def test_chunk_dataframe_shape_and_values(self, result: dict) -> None:
        table = chunk_dataframe(result)

        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == TABLE_COLUMNS
        assert len(table) == 2
        assert table.loc[0, "compression"] == "25%"
        assert table.loc[1, "ok"] == False  # noqa: E712 - pandas boolean comparison

    def test_chunk_dataframe_handles_empty(self) -> None:
        assert chunk_dataframe({}).empty

    def test_summary_metrics(self, result: dict, processed: dict) -> None:
        metrics = summary_metrics(result, processed)

        assert metrics["pages"] == 3
        assert metrics["tokens"] == 123
        assert metrics["chunks"] == 2
        assert metrics["successful"] == 1
        assert metrics["failed"] == 1
        assert metrics["map_seconds"] == 1.5
        assert metrics["summary_chars"] == len("the final summary")


class TestExports:
    def test_export_text_contains_core_sections(self, result: dict, processed: dict) -> None:
        text = build_export_text(result, processed, "report.pdf")

        assert "PDF SUMMARY REPORT" in text
        assert "Source      : report.pdf" in text
        assert "FINAL SUMMARY" in text
        assert "the final summary" in text
        assert "DOCUMENT STRUCTURE" not in text  # analysis was not requested

    def test_export_includes_analysis_and_quotes_when_present(
        self, result: dict, processed: dict
    ) -> None:
        result["analysis"] = {"status": "success", "analysis": "three parts"}
        result["quotes"] = ["quote one", "quote two"]

        text = build_export_text(result, processed, "report.pdf")

        assert "DOCUMENT STRUCTURE" in text
        assert "three parts" in text
        assert "KEY QUOTES" in text
        assert "1. quote one" in text

    def test_export_tolerates_failed_analysis(self, result: dict, processed: dict) -> None:
        result["analysis"] = {"status": "error", "analysis": "unavailable"}
        assert "DOCUMENT STRUCTURE" not in build_export_text(result, processed, "r.pdf")

    def test_export_bytes_are_utf8(self, result: dict, processed: dict) -> None:
        data = export_text_bytes(result, processed, "report.pdf")
        assert isinstance(data, bytes)
        assert data.decode("utf-8").startswith("PDF SUMMARY REPORT")

    def test_csv_export_is_parsable(self, result: dict, tmp_path: Path) -> None:
        path = tmp_path / "chunks.csv"
        path.write_bytes(export_csv_bytes(result))

        table = pd.read_csv(path)
        assert list(table.columns) == TABLE_COLUMNS
        assert len(table) == 2
