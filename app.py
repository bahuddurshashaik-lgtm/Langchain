"""PDF Summarizer — Streamlit UI.

Upload a PDF, choose a style, and get a chunked (map-reduce) summary from Gemini.
Results are cached per file hash, so re-running the page never re-bills the API.
"""

from __future__ import annotations

import io
import os

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from prompts import STYLES
from summarizer import PDFSummarizer
from utils import (
    MODEL_OPTIONS,
    STYLE_LABELS,
    chunk_dataframe,
    export_csv_bytes,
    export_text_bytes,
    file_hash,
    format_bytes,
    is_api_key_configured,
    safe_filename,
    summary_metrics,
)

load_dotenv()

st.set_page_config(page_title="PDF Summarizer", page_icon=":page_facing_up:", layout="wide")


# -- configuration ------------------------------------------------------


def get_api_key() -> str:
    """API key from the environment, falling back to Streamlit secrets."""
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        return ""


@st.cache_data(show_spinner=False, max_entries=8)
def process_pdf(file_bytes: bytes, max_tokens: int) -> dict:
    """Extract, clean and chunk the PDF (cached by content)."""
    return PDFProcessor().prepare(io.BytesIO(file_bytes), max_tokens=max_tokens)


@st.cache_resource(show_spinner=False, max_entries=4)
def get_summarizer(model: str) -> PDFSummarizer:
    """One Gemini client per model, reused across reruns."""
    return PDFSummarizer(model=model)


@st.cache_data(show_spinner=False, max_entries=16)
def run_summary(
    chunks: tuple[str, ...],
    summary_type: str,
    model: str,
    full_text: str,
    include_analysis: bool,
    include_quotes: bool,
) -> dict:
    """Map-reduce summarization (cached, so a rerun costs nothing)."""
    summarizer = get_summarizer(model)
    return summarizer.summarize_document(
        list(chunks),
        summary_type=summary_type,
        full_text=full_text,
        include_analysis=include_analysis,
        include_quotes=include_quotes,
    )


api_key = get_api_key()
env_model = (os.getenv("GEMINI_MODEL") or MODEL_OPTIONS[0]).strip()
model_options = list(dict.fromkeys((env_model, *MODEL_OPTIONS)))

# -- sidebar ------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    if is_api_key_configured(api_key):
        st.success(f"API key loaded ({len(api_key)} chars)")
    else:
        st.error("No API key found")
        st.caption("Add `GEMINI_API_KEY` to `.env`, then restart the app.")

    model = st.selectbox(
        "Model", model_options, index=0, help="Any Gemini model your key can access."
    )

    summary_type = st.selectbox(
        "Summary type",
        list(STYLES),
        format_func=lambda key: STYLE_LABELS.get(key, key),
    )

    with st.expander("Processing options", expanded=True):
        max_tokens = st.slider(
            "Tokens per chunk",
            min_value=500,
            max_value=8000,
            value=4000,
            step=500,
            help="Larger chunks mean fewer API calls; smaller chunks suit dense documents.",
        )
        include_analysis = st.checkbox("Analyze document structure", value=False)
        include_quotes = st.checkbox("Extract key quotes", value=False)

    st.divider()
    if st.button("Clear cached results", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("outcome", None)
        st.rerun()
    st.caption("Summaries are cached per file, so reruns do not call the API again.")

# -- main ---------------------------------------------------------------

st.title("📄 PDF AI Summarizer")
st.caption(
    "Extract → clean → chunk → summarize each chunk → combine. Powered by Gemini via LangChain."
)

if not is_api_key_configured(api_key):
    st.info(
        "Copy `.env.example` to `.env`, add `GEMINI_API_KEY`, then run "
        "`python scripts/check_env.py` to verify the key."
    )

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is None:
    st.session_state.pop("outcome", None)
    st.stop()

file_bytes = uploaded.getvalue()
doc_hash = file_hash(file_bytes)

with st.spinner("Reading PDF..."):
    try:
        processed = process_pdf(file_bytes, max_tokens)
    except ValueError as exc:
        st.error(f"{exc}. If the pages are images, run OCR on the PDF first.")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read this PDF: {exc}")
        st.stop()

meta = processed["metadata"]
info_col, size_col, page_col, token_col, chunk_col = st.columns(5)
info_col.metric("Title", (meta.get("title") or "Unknown")[:22])
size_col.metric("File size", format_bytes(len(file_bytes)))
page_col.metric("Pages", meta.get("num_pages", 0))
token_col.metric("Tokens", f"{processed['num_tokens']:,}")
chunk_col.metric("Chunks", processed["num_chunks"])

with st.expander("Document preview", expanded=False):
    st.caption(
        f"Author: {meta.get('author', 'Unknown')} · Subject: {meta.get('subject', 'Unknown')} · "
        f"Chunk limit: {max_tokens} tokens · File ID: {doc_hash}"
    )
    for index, chunk in enumerate(processed["chunks"][:3], start=1):
        st.markdown(f"**Chunk {index}** — {len(chunk):,} chars")
        st.text(chunk[:600] + ("..." if len(chunk) > 600 else ""))
    if processed["num_chunks"] > 3:
        st.caption(f"... and {processed['num_chunks'] - 3} more chunks")

summarize = st.button(
    f"Summarize with {model}",
    type="primary",
    disabled=not is_api_key_configured(api_key),
)

if summarize:
    try:
        with st.status("Summarizing document...", expanded=True) as status:
            status.write(f"Map: summarizing {processed['num_chunks']} chunk(s) in parallel")
            result = run_summary(
                tuple(processed["chunks"]),
                summary_type,
                model,
                processed["text"],
                include_analysis,
                include_quotes,
            )
            status.write("Reduce: merging section summaries")
            if include_analysis:
                status.write("Analyzing document structure")
            if include_quotes:
                status.write("Extracting key quotes")
            status.update(label="Summary ready", state="complete", expanded=False)
        st.session_state["outcome"] = {
            "result": result,
            "processed": processed,
            "source": uploaded.name,
        }
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        st.error(f"Summarization failed: {message}")
        if "429" in message or "quota" in message.lower():
            st.warning("Rate limit hit. Wait a minute, or pick a lighter model in the sidebar.")
        elif "404" in message or "not found" in message.lower():
            st.warning("That model is not available to your key. Try another or run check_env.py.")
        st.stop()

outcome = st.session_state.get("outcome")
if not outcome:
    st.stop()

result = outcome["result"]
processed = outcome["processed"]
source = outcome["source"]
metrics = summary_metrics(result, processed)

if metrics["failed"]:
    st.warning(f"{metrics['failed']} of {metrics['chunks']} chunks failed and were skipped.")

st.subheader("Result")
st.caption(
    f"{metrics['successful']}/{metrics['chunks']} chunks summarized · "
    f"map time {metrics['map_seconds']}s · style `{result['summary_type']}` · "
    f"model `{result['model']}`"
)

tabs = st.tabs(["Summary", "Chunk summaries", "Structure", "Quotes", "Statistics"])
summary_tab, chunks_tab, analysis_tab, quotes_tab, stats_tab = tabs

with summary_tab:
    st.markdown(result["combined_summary"])
    st.divider()
    download_col, csv_col = st.columns(2)
    stem = safe_filename(source)
    download_col.download_button(
        "Download summary (.txt)",
        data=export_text_bytes(result, processed, source),
        file_name=f"{stem}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    csv_col.download_button(
        "Download chunk stats (.csv)",
        data=export_csv_bytes(result),
        file_name=f"{stem}_chunks.csv",
        mime="text/csv",
        use_container_width=True,
    )

with chunks_tab:
    for item in result["individual_summaries"]:
        label = (
            f"Chunk {item['chunk_number']}: {item['original_length']:,} chars → "
            f"{item['summary_length']:,} chars"
        )
        with st.expander(label, expanded=item["chunk_number"] == 1):
            if item["ok"]:
                st.markdown(item["summary"])
            else:
                st.error(item["summary"])

with analysis_tab:
    analysis = result.get("analysis")
    if not analysis:
        st.caption("Enable “Analyze document structure” in the sidebar, then summarize again.")
    elif analysis.get("status") == "success":
        st.markdown(analysis["analysis"])
    else:
        st.error(analysis["analysis"])

with quotes_tab:
    quotes = result.get("quotes")
    if not quotes:
        st.caption("Enable “Extract key quotes” in the sidebar, then summarize again.")
    else:
        for index, quote in enumerate(quotes, start=1):
            st.markdown(f"**{index}.** {quote}")
        if result.get("quotes_error"):
            st.warning(f"Quotes were incomplete: {result['quotes_error']}")

with stats_tab:
    st.dataframe(chunk_dataframe(result), use_container_width=True, hide_index=True)
    st.json(metrics, expanded=False)
