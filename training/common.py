"""Shared offline-training helpers.

The Streamlit app does not import this package. Dataset files are read from
``VOICESENSE_DATA_DIR`` or a ``--data-dir`` argument, not from a hard-coded
thesis folder.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.config import DISCLAIMER, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE

# The split samples test subjects inside each class. It is not GroupShuffleSplit.
SPLIT_METHOD = "subject-balanced group holdout 80/20"


def resolve_data_dir(cli_path: str | None = None) -> Path:
    raw = cli_path or os.environ.get("VOICESENSE_DATA_DIR")
    if not raw:
        raise FileNotFoundError(
            "Dataset root is not set. Pass --data-dir or set VOICESENSE_DATA_DIR "
            "to the folder that contains 01_UCI_Parkinsons, 02_BenSParX, and "
            "07_Italian_PD_Voice."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {path}")
    return path


def stratified_group_holdout(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-aware hold-out that balances PD and HC subjects on both sides.

    GroupShuffleSplit alone can put almost all healthy subjects on one side
    when the healthy subject count is small (UCI has only 8 HC subjects).
    This routine samples test subjects within each class.
    """
    rng = np.random.RandomState(seed)
    group_to_label = {}
    group_to_indices = {}
    for i, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(i)
        group_to_label[group] = int(y[i])

    test_groups = []
    train_groups = []
    for label in sorted(set(group_to_label.values())):
        group_list = [group for group, lab in group_to_label.items() if lab == label]
        group_list = list(rng.permutation(group_list))
        n_test = max(1, round(len(group_list) * test_size))
        if len(group_list) - n_test < 1 and len(group_list) > 1:
            n_test = len(group_list) - 1
        test_groups.extend(group_list[:n_test])
        train_groups.extend(group_list[n_test:])

    train_idx = np.array(
        [i for group in train_groups for i in group_to_indices[group]], dtype=int
    )
    test_idx = np.array(
        [i for group in test_groups for i in group_to_indices[group]], dtype=int
    )
    return rng.permutation(train_idx), rng.permutation(test_idx)


__all__ = [
    "DISCLAIMER",
    "MODELS_DIR",
    "RANDOM_STATE",
    "REPORTS_DIR",
    "SPLIT_METHOD",
    "TEST_SIZE",
    "resolve_data_dir",
    "stratified_group_holdout",
]
