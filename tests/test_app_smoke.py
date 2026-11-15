"""Phase 3 smoke test: the Streamlit script runs end to end without a browser."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_upload() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    assert any("PDF AI Summarizer" in title.value for title in at.title)


def test_sidebar_exposes_model_and_style_controls() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    labels = [widget.label for widget in at.selectbox]
    assert "Model" in labels
    assert "Summary type" in labels

    styles = at.selectbox[1].options
    assert len(styles) == 4
    for expected in ("Brief", "Detailed", "Bullet points", "Executive"):
        assert any(style.startswith(expected) for style in styles), expected


def test_uploader_and_options_are_present() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    assert at.slider, "chunk-size slider missing"
    assert len(at.checkbox) == 2, "analysis and quotes toggles missing"
    assert at.button, "controls missing"
