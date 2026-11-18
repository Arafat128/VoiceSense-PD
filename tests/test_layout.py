"""The running app does not import the offline training package."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_app_and_src_do_not_import_training_or_removed_libs() -> None:
    roots = [ROOT / "app", ROOT / "src"]
    for root in roots:
        for path in root.rglob("*.py"):
            modules = _imports(path)
            joined = " ".join(modules)
            assert not any(name == "training" or name.startswith("training.") for name in modules), path
            assert "matplotlib" not in joined, path
            assert not any(name == "scipy" or name.startswith("scipy.") for name in modules), path


def test_reports_do_not_claim_group_shuffle_split() -> None:
    for path in (ROOT / "reports").glob("*.json"):
        assert "GroupShuffleSplit" not in path.read_text(encoding="utf-8")
    tables = (ROOT / "training" / "train_tables.py").read_text(encoding="utf-8")
    audio = (ROOT / "training" / "train_audio.py").read_text(encoding="utf-8")
    assert "GroupShuffleSplit" not in tables
    assert "GroupShuffleSplit" not in audio
    assert "SPLIT_METHOD" in tables
    assert "subject-balanced group holdout" in (ROOT / "training" / "common.py").read_text(
        encoding="utf-8"
    )
