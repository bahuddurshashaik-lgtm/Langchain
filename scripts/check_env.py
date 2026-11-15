"""Phase 0 credential smoke test.

Validates, in order:
  1. .env is loadable and GEMINI_API_KEY looks like a Google API key
  2. the key is accepted by the Gemini API (model listing works)
  3. the configured GEMINI_MODEL can actually generate text through LangChain

Exit codes (CI-usable):
  0  key valid and at least one model generated text
  1  configuration problem (.env missing / key empty)
  2  key rejected by the API (401/403)
  3  key valid but no model could generate text
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
FALLBACK_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash")

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def _status(marker: str, label: str, detail: str = "") -> None:
    line = f"{marker} {label}"
    print(f"{line:<42} {detail}" if detail else line, flush=True)


def check_key_format(key: str) -> bool:
    """Accept both classic keys (AIza..., 39 chars) and the newer AQ. format."""
    if key.startswith("AIza") and len(key) == 39:
        return True
    if key.startswith("AQ.") and len(key) >= 20:
        return True
    return False


def list_models(key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=key)
    names = []
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        if name.startswith("models/"):
            name = name[len("models/") :]
        if name:
            names.append(name)
    return names


def generate(model: str, key: str) -> tuple[str, int | None]:
    """One real generation call through the same LangChain path the app uses."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=key)
    started = time.perf_counter()
    reply = llm.invoke("Reply with exactly one word: OK")
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    text = getattr(reply, "text", None) or str(reply.content)
    usage = getattr(reply, "usage_metadata", None) or {}
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    print(f"       response={text.strip()!r} latency={elapsed_ms}ms tokens={total}")
    return text, total


def main() -> int:
    print("== Gemini credential check ==\n")

    env_file = ROOT / ".env"
    if not env_file.exists():
        _status(FAIL, ".env present", f"not found at {env_file}")
        return 1
    load_dotenv(env_file)
    _status(OK, ".env present", str(env_file))

    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        _status(FAIL, "GEMINI_API_KEY set", "empty or missing")
        return 1
    _status(OK, "GEMINI_API_KEY set", f"{len(key)} chars, {key[:4]}...")

    if check_key_format(key):
        _status(OK, "key format recognised")
    else:
        _status(WARN, "key format unrecognised", "continuing anyway")

    print("\n-- API reachability --")
    try:
        models = list_models(key)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        hint = "key rejected" if ("401" in message or "403" in message or "API key" in message) else "request failed"
        _status(FAIL, "model listing", f"{hint}: {message[:160]}")
        return 2

    _status(OK, "model listing", f"{len(models)} models visible to this key")
    for wanted in ("gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"):
        _status(OK if wanted in models else WARN, f"model available: {wanted}", "" if wanted in models else "not listed")

    print("\n-- generation through LangChain --")
    configured = (os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite").strip()
    candidates = [configured] + [m for m in FALLBACK_MODELS if m != configured]

    working: str | None = None
    for model in candidates:
        try:
            generate(model, key)
            _status(OK, f"generate: {model}")
            working = model
            break
        except Exception as exc:  # noqa: BLE001
            _status(FAIL, f"generate: {model}", str(exc).splitlines()[0][:160])

    print()
    if working is None:
        _status(FAIL, "result", "key accepted but no model generated text")
        return 3

    if working != configured:
        _status(WARN, "GEMINI_MODEL mismatch", f"set GEMINI_MODEL={working} in .env")
    else:
        _status(OK, "result", f"{working} is ready to summarize")

    print("\nPhase 0 credential check: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
