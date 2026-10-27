"""
SINYAL Phase 1 — Reproducible Data Preparation Pipeline.

Implements Section 9 of the SINYAL Phase 1 specification:
1. Reads only the verified raw CSVs from data/raw/.
2. Preserves the official training, validation, and testing split assignments (never merges or repartitions).
3. Selects id, split, message, original, genre, and the five target labels:
   request, water, food, shelter, medical_help, plus creates normalized_message.
4. Validates that all five target labels exist and contain strictly binary {0, 1} values.
5. Applies deterministic, documented filtering rules:
   - Rule 1: Drop missing or whitespace-only message texts.
   - Rule 2: Within-split conflicting annotations (identical normalized_message with different 5-target vectors)
     are dropped in train and validation so models do not train/tune on contradictory ground truth.
     In test, exact duplicate (id, normalized_message) rows or duplicate normalized_message rows are
     deduplicated deterministically (keep='first') while recording any label discrepancies.
   - Rule 3: Within-split exact duplicates on id or normalized_message (with identical labels) are deduplicated (keep='first').
   - Rule 4: Cross-split leakage remediation preserving the held-out test set as reference:
     * Any id or normalized_message present in test is removed from train and validation.
     * Any id or normalized_message present in validation is removed from train.
6. Saves deterministic UTF-8 CSV outputs to:
   - data/processed/train.csv
   - data/processed/validation.csv
   - data/processed/test.csv
7. Writes complete before/after audit trace to reports/preprocessing_summary.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_LABELS = [
    "request",
    "water",
    "food",
    "shelter",
    "medical_help",
]

RAW_SPLIT_FILES = {
    "train": "disaster_response_training.csv",
    "validation": "disaster_response_validation.csv",
    "test": "disaster_response_test.csv",
}

PROCESSED_SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

METADATA_COLUMNS = ["id", "split", "message", "normalized_message", "original", "genre", "event"]
OUTPUT_COLUMNS = METADATA_COLUMNS + TARGET_LABELS


def normalize_text(text: Any) -> str:
    """Deterministic text normalization for duplicate/leakage detection and clean reference.
    Preserves numbers, negations, location descriptions, and urgent expressions.
    """
    if pd.isna(text):
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_raw_split(df: pd.DataFrame, split_name: str) -> None:
    """Validate required columns and allowed binary target values {0, 1}."""
    required_cols = ["id", "message", "genre"] + TARGET_LABELS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Split '{split_name}' is missing required columns: {missing_cols}")

    for lbl in TARGET_LABELS:
        if df[lbl].isna().any():
            raise ValueError(f"Split '{split_name}' has null values in target label '{lbl}'.")
        invalid_vals = set(df[lbl].unique()) - {0, 1}
        if invalid_vals:
            raise ValueError(
                f"Split '{split_name}' has non-binary values in target label '{lbl}': {invalid_vals}"
            )


def find_conflicting_norm_messages(df: pd.DataFrame) -> set[str]:
    """Return the set of normalized_message strings that appear multiple times in df with different 5-target labels."""
    dup_mask = df.duplicated(subset=["normalized_message"], keep=False)
    if not dup_mask.any():
        return set()
    conflicting_texts: set[str] = set()
    for norm_msg, grp in df.loc[dup_mask].groupby("normalized_message"):
        if len(grp[TARGET_LABELS].drop_duplicates()) > 1:
            conflicting_texts.add(str(norm_msg))
    return conflicting_texts


def has_direct_personal_pii(df: pd.DataFrame) -> pd.Series:
    """
    Detect direct-report messages containing unredacted personal contact patterns
    (e.g., personal email addresses or phone numbers in `genre == 'direct'`),
    in compliance with the source repository's privacy restriction:
    'If any message is found to still contain personal identifying information
     it should be excluded from your study.'
    Never logs or outputs the raw message text.
    """
    email_pat = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_pat = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")

    is_direct = df["genre"].astype(str).str.lower() == "direct"
    has_email = df["message"].fillna("").astype(str).apply(lambda s: bool(email_pat.search(s)))
    has_phone = df["message"].fillna("").astype(str).apply(lambda s: bool(phone_pat.search(s)))
    has_pii_flag = (
        df["PII"].fillna(0).astype(int) != 0
        if "PII" in df.columns
        else pd.Series(False, index=df.index)
    )
    return (is_direct & (has_email | has_phone)) | has_pii_flag


def clean_within_split(
    df: pd.DataFrame,
    split_name: str,
    drop_conflicting_duplicates: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic within-split cleaning and record every step."""
    initial_rows = len(df)
    work = df.copy()

    # Ensure split column reflects official split assignment
    work["split"] = split_name
    if "original" not in work.columns:
        work["original"] = pd.NA
    if "event" not in work.columns:
        work["event"] = pd.NA

    # Compute normalized_message
    work["normalized_message"] = work["message"].apply(normalize_text)

    # Step 1: Drop missing or whitespace-only messages
    empty_mask = work["message"].isna() | (work["normalized_message"] == "")
    dropped_empty = int(empty_mask.sum())
    work = work.loc[~empty_mask].copy()

    # Step 1b: Privacy exclusion check (direct messages containing unredacted personal email/phone PII)
    pii_mask = has_direct_personal_pii(work)
    dropped_pii_count = int(pii_mask.sum())
    dropped_pii_ids = [int(x) for x in work.loc[pii_mask, "id"].tolist()]
    work = work.loc[~pii_mask].copy()

    # Step 2: Inspect conflicting duplicate normalized messages within this split
    conflicting_texts = find_conflicting_norm_messages(work)
    conflicting_group_count = len(conflicting_texts)
    if drop_conflicting_duplicates and conflicting_texts:
        conflict_mask = work["normalized_message"].isin(conflicting_texts)
        dropped_within_conflict = int(conflict_mask.sum())
        work = work.loc[~conflict_mask].copy()
    else:
        # In test split, record how many rows had conflicting duplicate texts before deduplicating keep='first'
        dropped_within_conflict = 0
        test_conflict_rows_involved = int(work["normalized_message"].isin(conflicting_texts).sum())

    # Step 3: Deduplicate exact duplicate IDs within split (keep='first')
    before_id_dedup = len(work)
    work = work.drop_duplicates(subset=["id"], keep="first").copy()
    dropped_dup_id = before_id_dedup - len(work)

    # Step 4: Deduplicate exact duplicate normalized_message within split (keep='first')
    before_msg_dedup = len(work)
    work = work.drop_duplicates(subset=["normalized_message"], keep="first").copy()
    dropped_dup_norm_msg = before_msg_dedup - len(work)

    # Enforce integer types on id and target labels
    work["id"] = work["id"].astype("int64")
    for lbl in TARGET_LABELS:
        work[lbl] = work[lbl].astype("int64")

    stats = {
        "split": split_name,
        "initial_rows": initial_rows,
        "dropped_null_or_whitespace_message": dropped_empty,
        "dropped_potential_pii_rows": dropped_pii_count,
        "dropped_potential_pii_record_ids_redacted_note": [
            f"id={rid} (excluded: unredacted personal contact pattern in direct message; content redacted)"
            for rid in dropped_pii_ids
        ],
        "within_split_conflicting_text_groups_detected": conflicting_group_count,
        "dropped_within_split_conflicting_rows": dropped_within_conflict,
        "dropped_within_split_duplicate_id": dropped_dup_id,
        "dropped_within_split_duplicate_norm_message": dropped_dup_norm_msg,
        "rows_after_within_split_cleaning": len(work),
    }
    if not drop_conflicting_duplicates:
        stats["test_conflicting_duplicate_rows_resolved_by_keep_first"] = test_conflict_rows_involved

    return work, stats


def prepare_all_splits() -> dict[str, Any]:
    """Run the complete reproducible preparation pipeline across train, validation, and test."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_dfs: dict[str, pd.DataFrame] = {}
    for split_name, filename in RAW_SPLIT_FILES.items():
        raw_path = RAW_DIR / filename
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Required raw split file not found: {raw_path}. Run `python -m src.audit_data` first."
            )
        df = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)
        validate_raw_split(df, split_name)
        raw_dfs[split_name] = df

    # Step A: Within-split cleaning
    # For train and validation, drop conflicting duplicate messages so models never train/select on contradictory labels.
    # For test, preserve the official test set as reference (deduplicating exact duplicate IDs/messages deterministically).
    train_clean, train_stats = clean_within_split(raw_dfs["train"], "train", drop_conflicting_duplicates=True)
    val_clean, val_stats = clean_within_split(raw_dfs["validation"], "validation", drop_conflicting_duplicates=True)
    test_clean, test_stats = clean_within_split(raw_dfs["test"], "test", drop_conflicting_duplicates=False)

    # Step B: Cross-split leakage removal (Test has highest priority, then Validation, then Train)
    test_ids = set(test_clean["id"].unique())
    test_msgs = set(test_clean["normalized_message"].unique())

    # Remove from validation any ID or normalized_message that appears in test
    val_leak_id_mask = val_clean["id"].isin(test_ids)
    val_leak_msg_mask = val_clean["normalized_message"].isin(test_msgs)
    val_leak_mask = val_leak_id_mask | val_leak_msg_mask
    val_dropped_cross_test = int(val_leak_mask.sum())
    val_final = val_clean.loc[~val_leak_mask, OUTPUT_COLUMNS].reset_index(drop=True)

    val_ids = set(val_final["id"].unique())
    val_msgs = set(val_final["normalized_message"].unique())

    # Remove from train any ID or normalized_message that appears in test OR validation
    train_leak_test_mask = train_clean["id"].isin(test_ids) | train_clean["normalized_message"].isin(test_msgs)
    train_dropped_cross_test = int(train_leak_test_mask.sum())
    train_after_test = train_clean.loc[~train_leak_test_mask].copy()

    train_leak_val_mask = train_after_test["id"].isin(val_ids) | train_after_test["normalized_message"].isin(val_msgs)
    train_dropped_cross_val = int(train_leak_val_mask.sum())
    train_final = train_after_test.loc[~train_leak_val_mask, OUTPUT_COLUMNS].reset_index(drop=True)

    test_final = test_clean[OUTPUT_COLUMNS].reset_index(drop=True)

    # Record cross-split removal counts
    train_stats["dropped_cross_split_leakage_with_test"] = train_dropped_cross_test
    train_stats["dropped_cross_split_leakage_with_validation"] = train_dropped_cross_val
    train_stats["final_rows"] = len(train_final)
    train_stats["total_removed_rows"] = train_stats["initial_rows"] - len(train_final)

    val_stats["dropped_cross_split_leakage_with_test"] = val_dropped_cross_test
    val_stats["dropped_cross_split_leakage_with_validation"] = 0
    val_stats["final_rows"] = len(val_final)
    val_stats["total_removed_rows"] = val_stats["initial_rows"] - len(val_final)

    test_stats["dropped_cross_split_leakage_with_test"] = 0
    test_stats["dropped_cross_split_leakage_with_validation"] = 0
    test_stats["final_rows"] = len(test_final)
    test_stats["total_removed_rows"] = test_stats["initial_rows"] - len(test_final)

    # Save deterministic UTF-8 CSV outputs
    final_dfs = {
        "train": train_final,
        "validation": val_final,
        "test": test_final,
    }
    output_files_meta: dict[str, Any] = {}
    processed_label_distributions: list[dict[str, Any]] = []

    for split_name, df_out in final_dfs.items():
        out_path = PROCESSED_DIR / PROCESSED_SPLIT_FILES[split_name]
        df_out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
        sha256_hex = compute_sha256(out_path)
        output_files_meta[split_name] = {
            "path": f"data/processed/{PROCESSED_SPLIT_FILES[split_name]}",
            "rows": len(df_out),
            "columns": list(df_out.columns),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_hex,
        }
        for lbl in TARGET_LABELS:
            pos_count = int((df_out[lbl] == 1).sum())
            neg_count = int((df_out[lbl] == 0).sum())
            total = len(df_out)
            processed_label_distributions.append(
                {
                    "split": split_name,
                    "label": lbl,
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "positive_rate": round(pos_count / total, 6) if total > 0 else 0.0,
                    "total_rows": total,
                }
            )
        print(
            f"[PREPARED] {split_name}: {len( raw_dfs[split_name] ):,} -> {len(df_out):,} rows "
            f"(removed {len(raw_dfs[split_name]) - len(df_out)}) | SHA-256: {sha256_hex}"
        )

    total_initial = sum(s["initial_rows"] for s in (train_stats, val_stats, test_stats))
    total_final = sum(s["final_rows"] for s in (train_stats, val_stats, test_stats))
    total_removed = total_initial - total_final

    summary = {
        "pipeline_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_labels": TARGET_LABELS,
        "output_columns": OUTPUT_COLUMNS,
        "normalization_rules": [
            "Unicode NFKC normalization",
            "Lowercasing",
            "Contiguous whitespace collapsed to single space and stripped",
            "Original message text preserved unmodified in `message` column",
        ],
        "filtering_policy": [
            "Rule 1: Remove records with null or whitespace-only `message`.",
            "Rule 1b (Privacy Protection): Exclude `direct` genre records containing unredacted personal contact patterns (e.g., personal email/phone) and record only a redacted audit note (`id` + reason) without copying message content.",
            "Rule 2: In `train` and `validation`, remove all instances of normalized messages that have conflicting 5-target label vectors within the same split.",
            "Rule 3: Within each split, deduplicate remaining exact duplicate `id`s and exact duplicate `normalized_message`s deterministically (`keep='first'`).",
            "Rule 4: Enforce zero cross-split leakage while preserving `test` as the held-out reference: remove from `validation` and `train` any `id` or `normalized_message` present in `test`, and remove from `train` any `id` or `normalized_message` present in `validation`.",
        ],
        "totals": {
            "initial_total_rows": total_initial,
            "final_total_rows": total_final,
            "total_removed_rows": total_removed,
        },
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
        "output_files": output_files_meta,
        "processed_label_distribution": processed_label_distributions,
    }

    summary_path = REPORTS_DIR / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVED] {summary_path}")
    return summary


if __name__ == "__main__":
    prepare_all_splits()
