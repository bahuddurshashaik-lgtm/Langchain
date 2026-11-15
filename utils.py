"""UI helpers: hashing, formatting, exports and statistics."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime

import pandas as pd
from langchain_core.messages import AIMessage

from prompts import STYLES

MODEL_OPTIONS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)

STYLE_LABELS = {
    "brief": "Brief — 2-3 sentences",
    "detailed": "Detailed — key points and conclusions",
    "bullet_points": "Bullet points — scannable list",
    "executive": "Executive — business insights",
}


def file_hash(data: bytes) -> str:
    """Stable identity for an uploaded file, used as the cache key."""
    return hashlib.sha256(data).hexdigest()[:16]


def safe_filename(name: str, fallback: str = "document") -> str:
    """Turn a PDF name into a safe stem for exported files."""
    stem = re.sub(r"\.pdf$", "", name or "", flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


def format_bytes(num_bytes: int) -> str:
    """Human-readable size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def is_api_key_configured(api_key: str | None) -> bool:
    """True when the key looks non-empty; format details are validated by check_env.py."""
    return bool((api_key or "").strip())


def chunk_dataframe(result: dict) -> pd.DataFrame:
    """Per-chunk statistics table."""
    rows = [
        {
            "chunk": item["chunk_number"],
            "ok": item["ok"],
            "original_chars": item["original_length"],
            "summary_chars": item["summary_length"],
            "compression": (
                f"{item['summary_length'] / item['original_length']:.0%}"
                if item["original_length"]
                else "-"
            ),
        }
        for item in result.get("individual_summaries", [])
    ]
    return pd.DataFrame(rows)


def summary_metrics(result: dict, processed: dict) -> dict:
    """Numbers shown in the metrics row and in exports."""
    return {
        "pages": processed.get("metadata", {}).get("num_pages", 0),
        "tokens": processed.get("num_tokens", 0),
        "chunks": result.get("total_chunks", 0),
        "successful": result.get("successful_chunks", 0),
        "failed": result.get("failed_chunks", 0),
        "map_seconds": result.get("map_elapsed_seconds", 0.0),
        "summary_chars": len(result.get("combined_summary", "")),
    }


def build_export_text(result: dict, processed: dict, source_name: str) -> str:
    """Plain-text export: metadata, final summary, analysis and quotes."""
    meta = processed.get("metadata", {})
    metrics = summary_metrics(result, processed)
    lines = [
        "PDF SUMMARY REPORT",
        "==================",
        f"Source      : {source_name}",
        f"Generated   : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Model       : {result.get('model', 'unknown')}",
        f"Style       : {result.get('summary_type', 'detailed')}",
        "",
        f"Pages       : {metrics['pages']}",
        f"Tokens      : {metrics['tokens']}",
        f"Chunks      : {metrics['successful']}/{metrics['chunks']} summarized",
        f"Map time    : {metrics['map_seconds']}s",
        "",
        f"Title       : {meta.get('title', 'Unknown')}",
        f"Author      : {meta.get('author', 'Unknown')}",
        "",
        "FINAL SUMMARY",
        "-------------",
        result.get("combined_summary", "").strip(),
    ]

    analysis = result.get("analysis")
    if analysis and analysis.get("status") == "success":
        lines += ["", "DOCUMENT STRUCTURE", "------------------", analysis["analysis"].strip()]

    quotes = result.get("quotes")
    if quotes:
        lines += ["", "KEY QUOTES", "----------"]
        lines += [f"{i}. {quote}" for i, quote in enumerate(quotes, start=1)]

    return "\n".join(lines) + "\n"


def export_text_bytes(result: dict, processed: dict, source_name: str) -> bytes:
    return build_export_text(result, processed, source_name).encode("utf-8")


def export_csv_bytes(result: dict) -> bytes:
    """Per-chunk statistics as CSV."""
    table = chunk_dataframe(result)
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def style_label(summary_type: str) -> str:
    return STYLE_LABELS.get(summary_type, summary_type)


def available_styles() -> tuple[str, ...]:
    return tuple(STYLES)


def message_content(message: AIMessage) -> str:
    """Small helper kept for symmetry with summarizer message parsing."""
    return getattr(message, "text", None) or str(message.content)
