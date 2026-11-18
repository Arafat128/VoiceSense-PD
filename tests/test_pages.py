"""The live and About screens render with the shipped model and reports."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _run(page: str) -> AppTest:
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.session_state["active_page"] = page
    return at.run()


def test_live_page_shows_data_derived_policy() -> None:
    at = _run("Live voice studio")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Decision policy" in text
    assert "72%" not in text
    assert "never saw" in text


def test_about_page_reports_held_out_corpora() -> None:
    at = _run("About & method")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "A corpus the model never saw" in text
    assert "not** used in live analysis" in text
