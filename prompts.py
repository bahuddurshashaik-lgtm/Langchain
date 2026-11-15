"""All LangChain prompt templates used by the summarizer.

Every prompt is a `ChatPromptTemplate` with a shared system message plus a
task-specific human message, so behaviour stays consistent and easy to tune.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are an expert document analyst. You read carefully, you never invent facts, "
    "and you answer only from the text you are given. If the text does not contain the "
    "information, say so instead of guessing."
)

STYLES: dict[str, str] = {
    "brief": (
        "Write a 2-3 sentence overview that captures the document's purpose and its single "
        "most important takeaway."
    ),
    "detailed": (
        "Write a thorough summary that covers the key points, the main arguments and the "
        "conclusions, preserving important numbers, names and relationships."
    ),
    "bullet_points": (
        "Write a scannable bullet list of the key points. One idea per bullet, no nested "
        "lists, no preamble."
    ),
    "executive": (
        "Write a business-focused summary: the situation, why it matters, the risks, and "
        "concrete recommendations for a decision maker."
    ),
}


def _template(human: str) -> ChatPromptTemplate:
    """Build a chat prompt from the shared system message plus a human message."""
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


CHUNK_SUMMARY = _template(
    "Summarize this section of a larger document.\n\n"
    "Required style: {style}\n\n"
    "Section text:\n{text}"
)

COMBINE = _template(
    "Below are summaries of consecutive sections of one document, in order.\n\n"
    "Merge them into a single coherent summary. Remove repetition, keep the ordering of "
    "ideas, and do not add information that is not present.\n\n"
    "Required style: {style}\n\n"
    "Section summaries:\n{summaries}"
)

ANALYSIS = _template(
    "Describe the structure of this document as a short bullet list: the kind of document, "
    "its apparent sections or parts, and how it is organised. Be concise.\n\n"
    "Document text:\n{text}"
)

QUOTES = _template(
    "Extract up to 10 of the most important verbatim quotations from this document. "
    "Copy them exactly as written, one per line, numbered. No commentary.\n\n"
    "Document text:\n{text}"
)

__all__ = ["ANALYSIS", "CHUNK_SUMMARY", "COMBINE", "QUOTES", "STYLES", "SYSTEM"]
