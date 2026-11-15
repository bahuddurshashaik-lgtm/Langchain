# WALKTHROUGH

A trace of one PDF through the system, phase by phase. Every step names the
function that does the work, so you can follow along in the source.

Start the app with `streamlit run app.py` and keep this file open beside it.

---

## 1. Upload

`app.py` → `st.file_uploader`

The bytes of the uploaded file become the identity of the document:

```python
file_bytes = uploaded.getvalue()
doc_hash = file_hash(file_bytes)      # sha256, first 16 hex chars
```

That hash is the cache key. Upload the same PDF twice and the second run reads
from Streamlit's cache instead of calling Gemini, so experimenting with styles
costs nothing after the first summary.

## 2. Extract, clean, chunk

`app.py` → `process_pdf()` → `pdf_processor.PDFProcessor.prepare()`

Three stages run in order:

1. **`extract_text_from_pdf`** — reads page by page with `pypdf`. A page that
   fails to parse is logged and skipped instead of aborting the document.
2. **`clean_text`** — removes control characters, collapses runs of spaces and
   tabs, trims spaces around newlines and caps blank lines at one paragraph.
   Punctuation and paragraph structure survive, because summaries depend on them.
3. **`chunk_text`** — splits on sentence boundaries (`(?<=[.!?])\s+`) and packs
   sentences into chunks up to `max_tokens`. A single sentence longer than a
   whole chunk is hard-split by token count, so the limit is never exceeded.
   Tokens are counted with tiktoken's `cl100k_base` as an approximation of
   Gemini's tokenizer.

The result is a dict with `text`, `chunks`, `num_chunks`, `num_tokens`,
`num_characters` and `metadata`. The UI shows the counts as metrics.

Why this matters: a naive `split()` on characters cuts sentences in half, and a
summary of half-sentences describes fragments instead of arguments.

## 3. Map: summarize each chunk

`app.py` → `run_summary()` → `summarizer.PDFSummarizer.summarize_chunks()`

`CHUNK_SUMMARY | llm | parser` is invoked through `_batch()`, which batches all
chunks with `max_concurrency=4` and `return_exceptions=True`.

Two design decisions here:

- **Per-chunk isolation.** A failed request is recorded with `ok: False` and a
  message, but the run continues. One bad chunk does not cost you the document.
- **Transient-only retries.** `_batch()` inspects each exception. Rate limits,
  quota errors, 5xx responses and timeouts are retried with exponential backoff;
  `400`/`403`/`404` failures are permanent and are reported immediately rather
  than hammering the API.

The reply is normalised by `_message_text()`. Gemini 3 returns content as a list
of typed blocks, so `str(message)` would leak the wrapper type — the helper joins
text blocks properly.

## 4. Reduce: combine the sections

`summarizer.PDFSummarizer.combine_summaries()`

Valid partial summaries are numbered `Section 1..N` and passed to `COMBINE`,
which merges them while preserving order and forbidding new information.

If the combine call itself fails, the code falls back to returning the numbered
sections. Degraded output beats a blank screen.

A single chunk skips this step entirely — combining one section adds cost with no
benefit.

## 5. Optional passes

- `analyze_document_structure()` sends the first 3000 characters to `ANALYSIS`
  and returns `{"status": "success" | "error"}` — it never raises.
- `extract_key_quotes()` sends up to 8000 characters to `QUOTES`, strips list
  markers and surrounding quotes, and caps the result at 10 quotations.

Both are opt-in, because each is an extra API request.

## 6. Render and export

`app.py` → `utils.py`

Metrics come from `summary_metrics()`, per-chunk tables from `chunk_dataframe()`,
and downloads from `export_text_bytes()` / `export_csv_bytes()`.

The text export is a full report: source name, timestamp, model, style, page and
token counts, final summary, then structure analysis and quotes when present.
The CSV is one row per chunk with an honest compression ratio — failed chunks
show zero rather than a misleading number.

## 7. What the tests pin down

| Area | File | Notable cases |
|---|---|---|
| Cleaning and chunking | `tests/test_pdf_processor.py` | token limits, no content loss, hard-split, blank and corrupt PDFs |
| Chains and failures | `tests/test_summarizer.py` | block parsing, failed chunk isolation, combine fallback, quote cleanup |
| Retries | `tests/test_retry.py` | transient detection, recovery, exhaustion, no retry on 4xx |
| Exports | `tests/test_utils.py` | filename safety, CSV parsing, report sections |
| UI | `tests/test_app_smoke.py` | script runs headless, widgets present |

`scripts/e2e_check.py` is the only test that spends money: real PDF, real API,
real map-reduce, plus the blank-PDF and unknown-model error states.

## 8. Failure modes you will actually see

| Symptom | Cause | Where it is handled |
|---|---|---|
| "No extractable text" | scanned image PDF | `chunk_text` raises `ValueError`; the UI suggests OCR |
| 429 / quota | free-tier limits | `_batch` retries with backoff, then reports per chunk |
| 404 model | model not enabled for the key | `check_env.py` names a reachable model |
| Slow first call | tiktoken vocabulary download | cached for later runs |
| Wrong summary style | old cached result | "Clear cached results" in the sidebar |
