"""The audio -> BenSParX path was measured and kept out of the app."""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_audio_check_report_decides_not_to_ship() -> None:
    r = json.loads((ROOT / "reports" / "bensparx_audio_check.json").read_text(encoding="utf-8"))
    assert r["clips_scored"] > 600
    assert r["ship_in_ui"] is False
    assert r["auc_all"]["value"] < 0.65


def test_app_never_builds_bensparx_rows_from_audio() -> None:
    sources = [p.read_text(encoding="utf-8") for p in (ROOT / "app").rglob("*.py")]
    sources += [(ROOT / "src" / "predict.py").read_text(encoding="utf-8")]
    assert not any("bensparx_audio_check" in s or "bensparx_row" in s for s in sources)


def test_bangla_mode_explains_missing_bengali_audio_model() -> None:
    at = AppTest.from_file(str(Path(ROOT, "app", "streamlit_app.py")), default_timeout=60)
    at.session_state["lang_mode"] = "bangla"
    at.run()
    assert not at.exception
    assert any("There is no Bengali audio model" in c.value for c in at.caption)
