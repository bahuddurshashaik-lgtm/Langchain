"""Tests for the summarizer chains — the model is faked, so no API calls are made."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from summarizer import PDFSummarizer, _message_text


def prompt_text(prompt_value) -> str:
    """Flatten a ChatPromptValue to plain text for assertions."""
    messages = getattr(prompt_value, "messages", None)
    if messages:
        return "\n".join(str(m.content) for m in messages)
    return str(prompt_value)


class Recorder:
    """Fake model: records prompts, returns a marker string, can fail on demand."""

    def __init__(self, fail_markers: tuple[str, ...] = (), reply: str | None = None):
        self.fail_markers = fail_markers
        self.reply = reply
        self.seen: list[str] = []

    def __call__(self, prompt_value) -> AIMessage:
        text = prompt_text(prompt_value)
        self.seen.append(text)
        for marker in self.fail_markers:
            if marker in text:
                raise RuntimeError(f"simulated failure: {marker}")
        return AIMessage(content=self.reply if self.reply is not None else f"summary[{len(text)}]")


def make_llm(fail_markers: tuple[str, ...] = (), reply: str | None = None):
    """Return (runnable fake model, recorder) using LangChain's RunnableLambda."""
    recorder = Recorder(fail_markers, reply)
    return RunnableLambda(recorder), recorder


@pytest.fixture
def summarizer() -> PDFSummarizer:
    llm, _ = make_llm()
    return PDFSummarizer(llm=llm)


class TestConstruction:
    def test_missing_api_key_raises_with_guidance(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
            PDFSummarizer()

    def test_model_defaults_to_env_value(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
        assert PDFSummarizer(llm=make_llm()[0]).model == "gemini-test-model"

    def test_explicit_model_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_MODEL", "from-env")
        assert PDFSummarizer(model="explicit", llm=make_llm()[0]).model == "explicit"


class TestMessageParsing:
    def test_plain_string_content(self) -> None:
        assert _message_text(AIMessage(content="hello")) == "hello"

    def test_block_list_content(self) -> None:
        message = AIMessage(content=[{"type": "text", "text": "block one"}])
        assert _message_text(message) == "block one"

    def test_multiple_text_blocks_are_joined(self) -> None:
        message = AIMessage(content=[{"text": "a"}, {"text": "b"}])
        assert _message_text(message) == "ab"

    def test_object_with_text_attribute(self) -> None:
        class Reply:
            text = "from attribute"

        assert _message_text(Reply()) == "from attribute"

    def test_raw_string_passes_through(self) -> None:
        assert _message_text("raw") == "raw"


class TestSummarizeText:
    def test_returns_text(self, summarizer: PDFSummarizer) -> None:
        assert summarizer.summarize_text("some text").startswith("summary[")

    def test_unknown_style_falls_back_to_detailed(self, summarizer: PDFSummarizer) -> None:
        assert summarizer.summarize_text("some text", summary_type="nonsense")

    def test_style_instruction_reaches_the_prompt(self) -> None:
        llm, recorder = make_llm()
        PDFSummarizer(llm=llm).summarize_text("text", summary_type="executive")
        assert "business-focused summary" in recorder.seen[0]


class TestSummarizeChunks:
    def test_counts_and_stats(self, summarizer: PDFSummarizer) -> None:
        result = summarizer.summarize_chunks(["a" * 10, "b" * 20], "brief")

        assert result["total_chunks"] == 2
        assert result["successful_chunks"] == 2
        assert result["failed_chunks"] == 0
        assert result["summary_type"] == "brief"
        assert len(result["individual_summaries"]) == 2
        assert result["individual_summaries"][0]["chunk_number"] == 1
        assert result["individual_summaries"][0]["original_length"] == 10
        assert result["combined_summary"]

    def test_empty_chunk_list_raises(self, summarizer: PDFSummarizer) -> None:
        with pytest.raises(ValueError, match="No chunks"):
            summarizer.summarize_chunks([])

    def test_one_failing_chunk_does_not_kill_the_run(self) -> None:
        llm, _ = make_llm(fail_markers=("POISON",))
        result = PDFSummarizer(llm=llm).summarize_chunks(
            ["alpha text", "POISON chunk", "gamma text"]
        )

        assert result["total_chunks"] == 3
        assert result["successful_chunks"] == 2
        assert result["failed_chunks"] == 1
        failed = result["individual_summaries"][1]
        assert failed["ok"] is False
        assert failed["summary_length"] == 0
        assert "Error processing chunk" in failed["summary"]
        assert result["combined_summary"]

    def test_all_chunks_failing_reports_clearly(self) -> None:
        llm, _ = make_llm(fail_markers=("POISON",))
        result = PDFSummarizer(llm=llm).summarize_chunks(["POISON one", "POISON two"])

        assert result["successful_chunks"] == 0
        assert result["combined_summary"] == "No valid summaries were generated."

    def test_single_chunk_skips_the_reduce_call(self, summarizer: PDFSummarizer) -> None:
        result = summarizer.summarize_chunks(["only chunk"], "detailed")
        assert result["combined_summary"] == result["individual_summaries"][0]["summary"]


class TestCombineFallback:
    def test_combine_failure_returns_concatenation(self) -> None:
        llm, _ = make_llm(fail_markers=("Merge them",))
        result = PDFSummarizer(llm=llm).summarize_chunks(["first chunk", "second chunk"])

        assert result["successful_chunks"] == 2
        assert result["combined_summary"].startswith("Combined Summary:")
        assert "Section 1" in result["combined_summary"]


class TestAnalysisAndQuotes:
    def test_analysis_success(self, summarizer: PDFSummarizer) -> None:
        assert summarizer.analyze_document_structure("text")["status"] == "success"

    def test_analysis_failure_is_reported(self) -> None:
        llm, _ = make_llm(fail_markers=("structure",))
        result = PDFSummarizer(llm=llm).analyze_document_structure("text")
        assert result["status"] == "error"
        assert "Error analyzing document" in result["analysis"]

    def test_analysis_only_sees_the_start_of_long_text(self) -> None:
        llm, recorder = make_llm()
        PDFSummarizer(llm=llm).analyze_document_structure("x" * 5000)
        assert "x" * 3000 in recorder.seen[0]
        assert "x" * 3001 not in recorder.seen[0]

    def test_quotes_are_cleaned_and_capped(self) -> None:
        lines = [f"{i}. quote number {i}" for i in range(1, 15)]
        lines += ["- bullet quote", "• another bullet", '  "quoted text"  ']
        llm, _ = make_llm(reply="\n".join(lines))

        quotes = PDFSummarizer(llm=llm).extract_key_quotes("text")

        assert len(quotes) == 10
        assert quotes[0] == "quote number 1"
        assert "1." not in quotes[0]


class TestSummarizeDocument:
    def test_without_optionals(self, summarizer: PDFSummarizer) -> None:
        result = summarizer.summarize_document(["a", "b"], "brief")
        assert "analysis" not in result
        assert "quotes" not in result

    def test_with_analysis_and_quotes(self, summarizer: PDFSummarizer) -> None:
        result = summarizer.summarize_document(
            ["a", "b"], "brief", include_analysis=True, include_quotes=True
        )
        assert result["analysis"]["status"] == "success"
        assert isinstance(result["quotes"], list) and result["quotes"]

    def test_quote_failure_is_non_fatal(self) -> None:
        llm, _ = make_llm(fail_markers=("verbatim quotations",))
        result = PDFSummarizer(llm=llm).summarize_document(
            ["a", "b"], "brief", include_quotes=True
        )
        assert result["quotes"] == []
        assert "quotes_error" in result
