"""Retry and backoff behaviour for transient Gemini failures."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from summarizer import PDFSummarizer, _is_transient


def prompt_text(prompt_value) -> str:
    messages = getattr(prompt_value, "messages", None)
    if messages:
        return "\n".join(str(m.content) for m in messages)
    return str(prompt_value)


class FlakyModel:
    """Fails `failures` times with `error`, then answers normally."""

    def __init__(
        self, failures: int, error: str = "429 Too Many Requests", marker: str | None = None
    ):
        self.failures = failures
        self.error = error
        self.marker = marker
        self.calls = 0

    def __call__(self, prompt_value) -> AIMessage:
        self.calls += 1
        text = prompt_text(prompt_value)
        applies = self.marker is None or self.marker in text
        if applies and self.failures > 0:
            self.failures -= 1
            raise RuntimeError(self.error)
        return AIMessage(content="recovered summary")


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "Quota exceeded for model",
        "503 Service Unavailable",
        "Deadline exceeded while calling model",
        "The model is overloaded, try again",
    ],
)
def test_transient_errors_are_detected(message: str) -> None:
    assert _is_transient(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["400 INVALID_ARGUMENT", "404 model not found", "API key not valid"],
)
def test_permanent_errors_are_not_retried(message: str) -> None:
    assert not _is_transient(RuntimeError(message))


def test_single_call_retries_then_succeeds() -> None:
    model = FlakyModel(failures=1)
    summarizer = PDFSummarizer(llm=RunnableLambda(model), retry_delay=0)

    assert summarizer.summarize_text("text") == "recovered summary"
    assert model.calls == 2


def test_single_call_gives_up_after_max_retries() -> None:
    model = FlakyModel(failures=99)
    summarizer = PDFSummarizer(llm=RunnableLambda(model), max_retries=2, retry_delay=0)

    with pytest.raises(RuntimeError, match="429"):
        summarizer.summarize_text("text")
    assert model.calls == 3


def test_permanent_failure_is_not_retried() -> None:
    model = FlakyModel(failures=99, error="400 INVALID_ARGUMENT")
    summarizer = PDFSummarizer(llm=RunnableLambda(model), retry_delay=0)

    with pytest.raises(RuntimeError, match="400"):
        summarizer.summarize_text("text")
    assert model.calls == 1


def test_failed_chunk_is_retried_and_recovers() -> None:
    model = FlakyModel(failures=1)
    summarizer = PDFSummarizer(llm=RunnableLambda(model), retry_delay=0)

    result = summarizer.summarize_chunks(["alpha chunk", "beta chunk"])

    assert result["successful_chunks"] == 2
    assert result["failed_chunks"] == 0
    assert model.calls == 4  # two chunks + one retry + the combine step


def test_exhausted_chunk_retries_are_reported() -> None:
    model = FlakyModel(failures=99)
    summarizer = PDFSummarizer(llm=RunnableLambda(model), max_retries=2, retry_delay=0)

    result = summarizer.summarize_chunks(["alpha", "beta"])

    assert result["failed_chunks"] == 2
    assert model.calls == 6  # 2 chunks x 3 attempts
    assert all(not item["ok"] for item in result["individual_summaries"])


def test_permanent_chunk_failure_is_not_retried() -> None:
    model = FlakyModel(failures=99, error="403 permission denied")
    summarizer = PDFSummarizer(llm=RunnableLambda(model), retry_delay=0)

    result = summarizer.summarize_chunks(["alpha", "beta"])

    assert result["failed_chunks"] == 2
    assert model.calls == 2
