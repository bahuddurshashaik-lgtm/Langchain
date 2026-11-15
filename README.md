# 📄 PDF AI Summarizer

A Streamlit app that summarizes PDF documents with **Google Gemini** (via LangChain). Upload a PDF, pick a style, and get a summary, a structure analysis and optional key quotes.

> **Status:** Phase 0 complete — environment, credentials and tooling verified. Application modules land in phases 1–6.

## What it does

- Extracts and cleans text from a PDF, then splits it into token-limited chunks
- Summarizes each chunk, then combines them into one final summary (map-reduce)
- Four summary types: `brief`, `detailed`, `bullet_points`, `executive`
- Optional document structure analysis and key quote extraction
- Exports the summary as text and per-chunk statistics as CSV

## Requirements

- Python 3.10+
- A Google Gemini API key — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env
```

`.env`:

```dotenv
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.1-flash-lite
```

## Verify your setup

```bash
python scripts/check_env.py
```

Checks that `.env` loads, that the API key is accepted, and that the configured model can actually generate text. Exit code `0` means you are good to go.

## Run

```bash
streamlit run app.py
```

## Troubleshooting

- **"No extractable text"** — the PDF is probably scanned images; run OCR first.
- **429 / rate limit** — wait a minute, use a smaller document, or switch model.
- **Model not available / 404** — run `scripts/check_env.py`; it names a model your key can reach.
- **API key not found** — make sure `.env` sits next to `app.py`.

## Project layout

```
app.py             Streamlit UI
pdf_processor.py   PDF extraction, cleaning, token counting, chunking
summarizer.py      Gemini chains: chunk summaries, combine, analysis, quotes
prompts.py         All ChatPromptTemplates
utils.py           Display, export, statistics helpers
scripts/           Environment and credential checks
```

## License

MIT
