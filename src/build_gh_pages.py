"""
SINYAL — GitHub Pages Static Bundle Builder (`src/build_gh_pages.py`).

Builds `_gh_pages/` by:
1. Copying `web/index.html`, `web/css/*`, and `web/js/*`.
2. Exporting exact JSON payloads from the real Phase 2 artifacts:
   - `static_data/metadata.json` (via `src.api.services.get_metadata()`)
   - `static_data/evaluation.json` (via `src.api.services.get_frozen_evaluation()`)
   - `static_data/model_weights.json` (exact TF-IDF vocabulary, IDF vector, and
     LogisticRegression coefficients/intercepts from `models/sinyal_model.joblib`
     plus `models/decision_policy.json`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from src import TARGET_LABELS
from src.api.services import get_frozen_evaluation, get_metadata
from src.modeling.inference import PROJECT_ROOT, load_inference_bundle

OUTPUT_DIR = PROJECT_ROOT / "_gh_pages"
WEB_DIR = PROJECT_ROOT / "web"


def build_github_pages_bundle(output_dir: Path = OUTPUT_DIR) -> None:
    """Create the static GitHub Pages bundle in `_gh_pages/`."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy web/ assets
    shutil.copy2(WEB_DIR / "index.html", output_dir / "index.html")
    shutil.copytree(WEB_DIR / "css", output_dir / "css")
    shutil.copytree(WEB_DIR / "js", output_dir / "js")

    # Add .nojekyll so GitHub Pages serves all files without Jekyll filtering
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    static_data_dir = output_dir / "static_data"
    static_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export metadata.json
    metadata = get_metadata().model_dump()
    (static_data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # 2. Export evaluation.json
    evaluation = get_frozen_evaluation().model_dump()
    (static_data_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # 3. Export exact TF-IDF + LogisticRegression weights from models/sinyal_model.joblib
    bundle = load_inference_bundle()
    pipeline = bundle["model"]
    manifest = bundle["manifest"]
    policy = bundle["policy"]

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Sort vocabulary by feature index 0..N-1
    vocab_dict = {str(term): int(idx) for term, idx in tfidf.vocabulary_.items()}
    idf_list = [round(float(x), 6) for x in tfidf.idf_]

    coefs: dict[str, list[float]] = {}
    intercepts: dict[str, float] = {}

    for j, lbl in enumerate(TARGET_LABELS):
        est = clf.estimators_[j]
        w = est.coef_[0]
        b = est.intercept_[0]
        coefs[lbl] = [round(float(x), 6) for x in w]
        intercepts[lbl] = round(float(b), 6)

    weights_payload = {
        "model_id": str(manifest.get("selected_model_id", "M1_logreg_C2.0_cwNone")),
        "model_version": str(manifest.get("model_version", "2.0.0")),
        "calibration_status": str(
            manifest.get("calibration_status", "uncalibrated_raw_probabilities")
        ),
        "target_labels": list(TARGET_LABELS),
        "policy": policy,
        "tfidf": {
            "lowercase": bool(tfidf.lowercase),
            "ngram_range": list(tfidf.ngram_range),
            "sublinear_tf": bool(tfidf.sublinear_tf),
            "vocab": vocab_dict,
            "idf": idf_list,
        },
        "classifiers": {
            "coefs": coefs,
            "intercepts": intercepts,
        },
    }

    (static_data_dir / "model_weights.json").write_text(
        json.dumps(weights_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[OK] Built GitHub Pages static bundle at: {output_dir}")


if __name__ == "__main__":
    build_github_pages_bundle()
