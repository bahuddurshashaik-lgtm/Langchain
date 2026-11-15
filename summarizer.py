"""Gemini calls: per-chunk summaries, combine, structure analysis and quotes."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_CONCURRENCY = 4


def _message_text(message: Any) -> str:
    """Normalise an LLM reply to plain text.

    Gemini 3 returns content as a list of typed blocks; older models return a
    string. `str(message)` would leak the block wrapper type, so unwrap properly.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content)


class PDFSummarizer:
    """Wraps the Gemini model in the four chains the app needs."""

    def __init__(self, model: str | None = None, llm: Any | None = None):
        self.model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()

        if llm is not None:
            self.llm = llm
        else:
            api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "then run scripts/check_env.py to verify it."
                )
            self.llm = ChatGoogleGenerativeAI(
                model=self.model, temperature=0.3, google_api_key=api_key
            )

        parser = RunnableLambda(_message_text)
        self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
        self.combine_chain = COMBINE | self.llm | parser
        self.analysis_chain = ANALYSIS | self.llm | parser
        self.quotes_chain = QUOTES | self.llm | parser

    # -- single calls -----------------------------------------------------

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        """Summarize one piece of text in the requested style."""
        style = STYLES.get(summary_type, STYLES["detailed"])
        return self.chunk_chain.invoke({"style": style, "text": text})

    def analyze_document_structure(self, text: str) -> dict:
        """Describe the document layout; failures are reported, never raised."""
        try:
            analysis = self.analysis_chain.invoke({"text": text[:3000]})
            return {"analysis": analysis, "status": "success"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structure analysis failed: %s", exc)
            return {"analysis": f"Error analyzing document: {exc}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        """Return up to 10 verbatim quotes, with list markers and quotes stripped."""
        response = self.quotes_chain.invoke({"text": text})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip(' "“”')
            for line in response.splitlines()
        )
        return [line for line in lines if line][:10]

    # -- map-reduce -------------------------------------------------------

    def summarize_chunks(
        self, chunks: list[str], summary_type: str = "detailed"
    ) -> dict:
        """Summarize every chunk (map), then merge the results (reduce).

        A failing chunk is recorded as an error and skipped by the reduce step,
        so one bad request never discards the whole document.
        """
        if not chunks:
            raise ValueError("No chunks to summarize")

        style = STYLES.get(summary_type, STYLES["detailed"])
        inputs = [{"style": style, "text": chunk} for chunk in chunks]

        started = time.perf_counter()
        results = self.chunk_chain.batch(
            inputs, config={"max_concurrency": MAX_CONCURRENCY}, return_exceptions=True
        )
        elapsed = time.perf_counter() - started

        summaries = []
        for index, (chunk, result) in enumerate(zip(chunks, results, strict=False), start=1):
            ok = not isinstance(result, Exception)
            if not ok:
                logger.warning("Chunk %d failed: %s", index, result)
            summary = result if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": index,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        successful = sum(1 for item in summaries if item["ok"])
        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "successful_chunks": successful,
            "failed_chunks": len(chunks) - successful,
            "summary_type": summary_type,
            "model": self.model,
            "map_elapsed_seconds": round(elapsed, 2),
        }

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        """Merge partial summaries into one; falls back to concatenation on failure."""
        valid = [item["summary"] for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(f"Section {i}: {s}" for i, s in enumerate(valid, start=1))
        style = STYLES.get(summary_type, STYLES["detailed"])
        try:
            return self.combine_chain.invoke({"style": style, "summaries": sections})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combine step failed, returning concatenation: %s", exc)
            return f"Combined Summary:\n\n{sections}"

    # -- convenience ------------------------------------------------------

    def summarize_document(
        self,
        chunks: list[str],
        summary_type: str = "detailed",
        full_text: str | None = None,
        include_analysis: bool = False,
        include_quotes: bool = False,
    ) -> dict:
        """Run the whole pipeline for a document already split into chunks."""
        result = self.summarize_chunks(chunks, summary_type)
        source = full_text if full_text is not None else "\n\n".join(chunks)

        if include_analysis:
            result["analysis"] = self.analyze_document_structure(source)
        if include_quotes:
            try:
                result["quotes"] = self.extract_key_quotes(source[:8000])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Quote extraction failed: %s", exc)
                result["quotes"] = []
                result["quotes_error"] = str(exc)

        return result
