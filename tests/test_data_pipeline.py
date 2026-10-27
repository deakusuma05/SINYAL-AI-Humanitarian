"""
SINYAL Phase 1 — Automated Tests for Data Audit & Reproducible Preparation Pipeline.

Implements Section 12 of the SINYAL Phase 1 specification:
1. Every processed split exists (train.csv, validation.csv, test.csv).
2. Required columns are present (id, split, message, normalized_message, original, genre, + 5 target labels).
3. The five selected target labels are strictly binary {0, 1}.
4. No null or empty/whitespace-only message remains after documented cleaning.
5. Train/validation/test assignments are preserved and match official source splits.
6. No duplicate IDs exist within or across splits.
7. No exact normalized messages leak across splits (or duplicate within splits) after the defined leakage policy.
8. The pipeline produces deterministic output (both on small synthetic fixtures and on the actual dataset).
9. Processed record counts agree with the audit & preprocessing summary logs.
10. Label values are not fabricated or silently altered compared to the raw source records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.prepare_data import (
    OUTPUT_COLUMNS,
    PROCESSED_DIR,
    PROCESSED_SPLIT_FILES,
    RAW_DIR,
    RAW_SPLIT_FILES,
    REPORTS_DIR,
    TARGET_LABELS,
    clean_within_split,
     compute_sha256,
    normalize_text,
    prepare_all_splits,
)


@pytest.fixture(scope="module")
def processed_splits() -> dict[str, pd.DataFrame]:
    """Load the three processed splits."""
    splits: dict[str, pd.DataFrame] = {}
    for split_name, filename in PROCESSED_SPLIT_FILES.items():
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Processed split file missing: {path}"
        splits[split_name] = pd.read_csv(path, encoding="utf-8", low_memory=False)
    return splits


@pytest.fixture(scope="module")
def preprocessing_summary() -> dict:
    """Load reports/preprocessing_summary.json."""
    summary_path = REPORTS_DIR / "preprocessing_summary.json"
    assert summary_path.exists(), f"Preprocessing summary missing: {summary_path}"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_1_every_processed_split_exists() -> None:
    """Verify train.csv, validation.csv, and test.csv exist and are non-empty."""
    for split_name, filename in PROCESSED_SPLIT_FILES.items():
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Missing processed split: {split_name} ({path})"
        assert path.stat().st_size > 1024, f"Processed split {split_name} is suspiciously small."


def test_2_required_columns_are_present(processed_splits: dict[str, pd.DataFrame]) -> None:
    """Verify required metadata and target columns exist in exact expected order."""
    for split_name, df in processed_splits.items():
        assert list(df.columns) == OUTPUT_COLUMNS, (
            f"Split '{split_name}' columns {list(df.columns)} do not match expected {OUTPUT_COLUMNS}"
        )


def test_3_target_labels_are_strictly_binary(processed_splits: dict[str, pd.DataFrame]) -> None:
    """Verify all five selected target labels contain only {0, 1} with no nulls."""
    assert TARGET_LABELS == ["request", "water", "food", "shelter", "medical_help"]
    for split_name, df in processed_splits.items():
        for lbl in TARGET_LABELS:
            assert not df[lbl].isna().any(), f"Null values found in '{lbl}' of split '{split_name}'"
            unique_vals = set(df[lbl].unique())
            assert unique_vals.issubset({0, 1}), (
                f"Non-binary values {unique_vals} found in '{lbl}' of split '{split_name}'"
            )


def test_4_no_null_or_empty_messages_remain(processed_splits: dict[str, pd.DataFrame]) -> None:
    """Verify no null, empty, or whitespace-only message or normalized_message remains."""
    for split_name, df in processed_splits.items():
        assert not df["message"].isna().any(), f"Null message in '{split_name}'"
        assert not df["normalized_message"].isna().any(), f"Null normalized_message in '{split_name}'"
        stripped_lengths = df["message"].astype(str).str.strip().str.len()
        assert (stripped_lengths > 0).all(), f"Whitespace-only message found in '{split_name}'"
        norm_lengths = df["normalized_message"].astype(str).str.strip().str.len()
        assert (norm_lengths > 0).all(), f"Empty normalized_message found in '{split_name}'"


def test_5_split_assignments_are_preserved(processed_splits: dict[str, pd.DataFrame]) -> None:
    """Verify that every record in each processed split originates from its corresponding raw split."""
    for split_name, df in processed_splits.items():
        assert (df["split"] == split_name).all(), (
            f"Split column mismatch inside '{split_name}'"
        )
        raw_df = pd.read_csv(RAW_DIR / RAW_SPLIT_FILES[split_name], usecols=["id"], encoding="utf-8")
        raw_ids = set(raw_df["id"].unique())
        proc_ids = set(df["id"].unique())
        assert proc_ids.issubset(raw_ids), (
            f"Processed split '{split_name}' contains IDs not from its official raw source split!"
        )


def test_6_no_duplicate_ids_within_or_across_splits(processed_splits: dict[str, pd.DataFrame]) -> None:
    """Verify zero duplicate IDs within each split and across any pair of splits."""
    for split_name, df in processed_splits.items():
        dup_count = int(df.duplicated(subset=["id"]).sum())
        assert dup_count == 0, f"Found {dup_count} duplicate IDs within '{split_name}'"

    train_ids = set(processed_splits["train"]["id"])
    val_ids = set(processed_splits["validation"]["id"])
    test_ids = set(processed_splits["test"]["id"])

    assert len(train_ids & val_ids) == 0, "Duplicate IDs leaked between train and validation"
    assert len(train_ids & test_ids) == 0, "Duplicate IDs leaked between train and test"
    assert len(val_ids & test_ids) == 0, "Duplicate IDs leaked between validation and test"


def test_7_no_normalized_messages_leak_within_or_across_splits(
    processed_splits: dict[str, pd.DataFrame],
) -> None:
    """Verify zero duplicate normalized_message strings within or across splits."""
    for split_name, df in processed_splits.items():
        recomputed_norm = df["message"].apply(normalize_text)
        assert (df["normalized_message"] == recomputed_norm).all(), (
            f"Stored normalized_message disagrees with normalize_text() in '{split_name}'"
        )
        dup_msg_count = int(df.duplicated(subset=["normalized_message"]).sum())
        assert dup_msg_count == 0, (
            f"Found {dup_msg_count} duplicate normalized messages within '{split_name}'"
        )

    train_msgs = set(processed_splits["train"]["normalized_message"])
    val_msgs = set(processed_splits["validation"]["normalized_message"])
    test_msgs = set(processed_splits["test"]["normalized_message"])

    assert len(train_msgs & val_msgs) == 0, "Normalized messages leaked between train and validation"
    assert len(train_msgs & test_msgs) == 0, "Normalized messages leaked between train and test"
    assert len(val_msgs & test_msgs) == 0, "Normalized messages leaked between validation and test"


def test_8_pipeline_is_deterministic_on_fixture_and_files(
    preprocessing_summary: dict,
) -> None:
    """Verify deterministic cleaning on a small synthetic fixture and checksum stability on disk."""
    fixture_df = pd.DataFrame(
        [
            # Normal valid row
            {"id": 1, "message": "We need clean water in Carrefour!", "original": None, "genre": "direct",
             "request": 1, "water": 1, "food": 0, "shelter": 0, "medical_help": 0},
            # Exact duplicate normalized message & identical labels -> keep first only
            {"id": 2, "message": "  we need   clean WATER in carrefour! ", "original": None, "genre": "direct",
             "request": 1, "water": 1, "food": 0, "shelter": 0, "medical_help": 0},
            # Conflicting duplicate message -> both rows must be dropped when drop_conflicting_duplicates=True
            {"id": 3, "message": "Food distribution at central school", "original": None, "genre": "news",
             "request": 0, "water": 0, "food": 1, "shelter": 0, "medical_help": 0},
            {"id": 4, "message": "food distribution at central school", "original": None, "genre": "news",
             "request": 1, "water": 0, "food": 1, "shelter": 0, "medical_help": 0},
            # Whitespace-only message -> must be dropped
            {"id": 5, "message": "    \t  ", "original": None, "genre": "social",
             "request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
        ]
    )
    cleaned_1, stats_1 = clean_within_split(fixture_df, "train", drop_conflicting_duplicates=True)
    cleaned_2, stats_2 = clean_within_split(fixture_df, "train", drop_conflicting_duplicates=True)

    pd.testing.assert_frame_equal(cleaned_1, cleaned_2)
    assert stats_1 == stats_2
    assert len(cleaned_1) == 1
    assert int(cleaned_1.iloc[0]["id"]) == 1
    assert stats_1["dropped_null_or_whitespace_message"] == 1
    assert stats_1["dropped_within_split_conflicting_rows"] == 2
    assert stats_1["dropped_within_split_duplicate_norm_message"] == 1

    # Also verify that re-running prepare_all_splits() produces identical file SHA-256 checksums
    before_hashes = {
        s: compute_sha256(PROCESSED_DIR / fname) for s, fname in PROCESSED_SPLIT_FILES.items()
    }
    prepare_all_splits()
    after_hashes = {
        s: compute_sha256(PROCESSED_DIR / fname) for s, fname in PROCESSED_SPLIT_FILES.items()
    }
    assert before_hashes == after_hashes, "Re-running prepare_all_splits() changed output file checksums!"


def test_9_processed_counts_agree_with_audit_and_summary_logs(
    processed_splits: dict[str, pd.DataFrame],
    preprocessing_summary: dict,
) -> None:
    """Verify that processed record counts match preprocessing_summary.json and audit_results.json."""
    audit_json_path = REPORTS_DIR / "audit_results.json"
    assert audit_json_path.exists(), "reports/audit_results.json does not exist"
    audit_data = json.loads(audit_json_path.read_text(encoding="utf-8"))

    for split_name, df in processed_splits.items():
        split_summary = preprocessing_summary["splits"][split_name]
        raw_audit_rows = audit_data["split_integrity"][split_name]["num_rows"]

        assert split_summary["initial_rows"] == raw_audit_rows
        assert split_summary["final_rows"] == len(df)
        assert split_summary["initial_rows"] - split_summary["total_removed_rows"] == len(df)

        # Verify accounting of removed rows
        expected_removed = (
            split_summary["dropped_null_or_whitespace_message"]
            + split_summary["dropped_potential_pii_rows"]
            + split_summary["dropped_within_split_conflicting_rows"]
            + split_summary["dropped_within_split_duplicate_id"]
            + split_summary["dropped_within_split_duplicate_norm_message"]
            + split_summary["dropped_cross_split_leakage_with_test"]
            + split_summary["dropped_cross_split_leakage_with_validation"]
        )
        assert split_summary["total_removed_rows"] == expected_removed, (
            f"Removed row accounting mismatch in '{split_name}': "
            f"{split_summary['total_removed_rows']} != {expected_removed}"
        )


def test_10_label_values_are_not_fabricated_or_altered(
    processed_splits: dict[str, pd.DataFrame],
) -> None:
    """Verify every row in each processed split has the exact same message and 5 target labels as its raw source row."""
    for split_name, proc_df in processed_splits.items():
        raw_df = pd.read_csv(
            RAW_DIR / RAW_SPLIT_FILES[split_name],
            usecols=["id", "message"] + TARGET_LABELS,
            encoding="utf-8",
        )
        # Since raw_df can have duplicate IDs, keep the first occurrence per ID (matching our deterministic rule)
        raw_first_by_id = raw_df.drop_duplicates(subset=["id"], keep="first").set_index("id")
        proc_indexed = proc_df.set_index("id")

        aligned_raw = raw_first_by_id.loc[proc_indexed.index]
        assert (proc_indexed["message"] == aligned_raw["message"]).all(), (
            f"Message text was altered in '{split_name}'!"
        )
        for lbl in TARGET_LABELS:
            assert (proc_indexed[lbl] == aligned_raw[lbl]).all(), (
                f"Ground-truth label '{lbl}' was altered or fabricated in '{split_name}'!"
            )
