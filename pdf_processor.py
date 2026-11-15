"""PDF text extraction, cleaning and token-aware chunking."""

from __future__ import annotations

import logging
import re

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8000


class PDFProcessor:
    """Turns an uploaded PDF into clean, token-bounded chunks."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Set up the tokenizer used for chunk sizing (an approximation for Gemini)."""
        self.encoding = tiktoken.get_encoding(encoding_name)

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Return the text of all pages; pages that fail to extract are skipped."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            pages = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping page %d: %s", page_num, exc)
            return "\n\n".join(pages)
        except Exception as exc:
            raise RuntimeError(f"Error reading PDF: {exc}") from exc

    def clean_text(self, text: str) -> str:
        """Drop control characters and normalise whitespace, keeping punctuation and paragraphs."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Number of tokens in the text."""
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk_text(self, text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
        """Split text into chunks of at most `max_tokens`, breaking on sentence boundaries.

        Sentences longer than a whole chunk are hard-split by token count so no chunk
        can ever exceed the limit.

        Raises:
            ValueError: if the text has no content (e.g. a scanned PDF).
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not text.strip():
            raise ValueError("No extractable text - the PDF may be scanned images")

        pieces: list[tuple[str, int]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            tokens = self.encoding.encode(sentence, disallowed_special=())
            if len(tokens) > max_tokens:
                for start in range(0, len(tokens), max_tokens):
                    window = tokens[start : start + max_tokens]
                    pieces.append((self.encoding.decode(window), len(window)))
            else:
                pieces.append((sentence, len(tokens)))

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece, n in pieces:
            if current and current_tokens + n + 1 > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += n + 1  # +1 approximates the joining space
        if current:
            chunks.append(" ".join(current))
        return chunks

    def get_pdf_metadata(self, pdf_file) -> dict:
        """Return num_pages, title, author and subject (or {'error': ...})."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            meta = reader.metadata or {}

            def field(key: str) -> str:
                value = meta.get(key)
                return str(value) if value else "Unknown"

            return {
                "num_pages": len(reader.pages),
                "title": field("/Title"),
                "author": field("/Author"),
                "subject": field("/Subject"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def prepare(self, pdf_file, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
        """Full pipeline: extract -> clean -> chunk. Returns text, chunks and stats."""
        raw = self.extract_text_from_pdf(pdf_file)
        cleaned = self.clean_text(raw)
        chunks = self.chunk_text(cleaned, max_tokens=max_tokens)
        return {
            "text": cleaned,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "num_tokens": self.count_tokens(cleaned),
            "num_characters": len(cleaned),
            "metadata": self.get_pdf_metadata(pdf_file),
        }
