"""PDF Summarizer — Phase 0 environment dashboard.

This is a temporary landing page; Phase 3 replaces it with the real UI.
It exists so `streamlit run app.py` can be verified before any app logic lands.
"""

from __future__ import annotations

import importlib.metadata as md
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PACKAGES = (
    "streamlit",
    "langchain-core",
    "langchain-google-genai",
    "pypdf",
    "python-dotenv",
    "pandas",
    "tiktoken",
)

st.set_page_config(page_title="PDF Summarizer", page_icon=":page_facing_up:", layout="wide")
st.title("PDF Summarizer")
st.caption("Phase 0 — environment and credential dashboard")

api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
model = (os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite").strip()

left, right = st.columns(2)

with left:
    st.subheader("Configuration")
    if api_key:
        st.success(f"GEMINI_API_KEY loaded ({len(api_key)} chars, {api_key[:4]}...)")
    else:
        st.error("GEMINI_API_KEY is missing — copy .env.example to .env and add your key.")
    st.write(f"**Model:** `{model}`")

with right:
    st.subheader("Installed packages")
    for package in PACKAGES:
        try:
            st.write(f"`{package}` {md.version(package)}")
        except md.PackageNotFoundError:
            st.write(f"`{package}` **not installed**")

st.divider()

if st.button("Run live Gemini check", type="primary", disabled=not api_key):
    with st.spinner("Calling Gemini..."):
        try:
            import time

            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
            started = time.perf_counter()
            reply = llm.invoke("Reply with exactly one word: OK")
            elapsed = int((time.perf_counter() - started) * 1000)
            st.success(f"`{model}` replied in {elapsed} ms")
            st.code((getattr(reply, "text", None) or str(reply.content)).strip())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gemini call failed: {exc}")

st.caption("Next: Phase 1 adds PDF extraction, cleaning and token-aware chunking.")
