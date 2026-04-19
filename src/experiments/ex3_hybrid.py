"""
Description:
This script runs Experiment 3: Hybrid mineral classification.
This code was used for the feasibility testing of the research project itself
you may start from the very scrap or make reference of this code.

Experiment Design:
- Train models using real measured training data together with synthetic data
- Test models on real measured test data only
- Evaluate only on test classes that are present in the training set

Purpose:
- Assess whether adding synthetic mineral data to the real training set
  improves classification performance on real measured mineral samples

Implementation Notes:
- The real measured test set is never used during training
- Synthetic data is added only to the training side of the experiment
- The test set is not oversampled, augmented, or merged with synthetic data
- Feature columns between real and synthetic datasets must be aligned
  before training
- If needed, synthetic data may be filtered or sampled to maintain
  computational efficiency and prevent memory issues during model training

Output:
- Performance metrics for each model and threshold
- Classification report
- Confusion matrix
- Saved experiment results for later comparison with other experiments
"""

from pathlib import Path
import json
import time
import traceback
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.naivebayes import build_naive_bayes
from src.models.knn import build_knn
from src.models.decisiontree import build_decision_tree
from src.models.randomforest import build_random_forest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_COL = "mineral_name"
THRESHOLDS = ["thr10", "thr20", "thr50"]

# -----------------------------
# Synthetic sampling controls
# -----------------------------
USE_SYNTHETIC_SAMPLING = True
SYNTHETIC_SAMPLE_MODE = "per_class_cap"   # options: "global", "per_class_cap"
GLOBAL_SYNTHETIC_SAMPLE_SIZE = 300000
PER_CLASS_SYNTHETIC_CAP = 1000
RANDOM_STATE = 42

MODELS = {
    "naive_bayes": build_naive_bayes,
    "knn": build_knn,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_outputs(output_dir, metrics, y_true, y_pred, labels):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv",
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")


def align_features(base_df, other_df, target_col):
    base_features = [c for c in base_df.columns if c != target_col]

    missing_in_other = [c for c in base_features if c not in other_df.columns]
    if missing_in_other:
        raise ValueError(
            f"Dataset is missing required feature columns: {missing_in_other}"
        )

    aligned_other = other_df[base_features + [target_col]].copy()
    return base_features, aligned_other


def sample_synthetic_data(synth_df, train_classes):
    """
    Sample synthetic data to keep experiment computationally manageable.

    This does NOT touch the test set.
    This does NOT mix train/test.
    This only reduces the synthetic training volume.
    """
    print("→ Preparing synthetic dataset for hybrid training...")

    original_rows = len(synth_df)
    original_classes = synth_df[TARGET_COL].nunique()

    # Keep only synthetic rows whose classes are relevant to the measured training set
    synth_df = synth_df[synth_df[TARGET_COL].isin(train_classes)].copy()

    filtered_rows = len(synth_df)
    filtered_classes = synth_df[TARGET_COL].nunique()

    print(f"  Synthetic rows before class filtering: {original_rows:,}")
    print(f"  Synthetic rows after class filtering:  {filtered_rows:,}")
    print(f"  Synthetic classes before filtering:    {original_classes:,}")
    print(f"  Synthetic classes after filtering:     {filtered_classes:,}")

    if not USE_SYNTHETIC_SAMPLING:
        print("✔ Synthetic sampling disabled; using all filtered synthetic rows")
        return synth_df, {
            "sampling_enabled": False,
            "sampling_mode": "none",
            "synthetic_rows_before_filter": int(original_rows),
            "synthetic_rows_after_filter": int(filtered_rows),
            "synthetic_rows_after_sampling": int(filtered_rows),
            "synthetic_classes_after_sampling": int(filtered_classes),
        }

    if SYNTHETIC_SAMPLE_MODE == "global":
        if len(synth_df) > GLOBAL_SYNTHETIC_SAMPLE_SIZE:
            synth_df = synth_df.sample(
                n=GLOBAL_SYNTHETIC_SAMPLE_SIZE,
                random_state=RANDOM_STATE
            ).copy()
            print(f"✔ Applied global synthetic sampling: {len(synth_df):,} rows kept")
        else:
            print("✔ Global sample size exceeds available rows; all filtered rows kept")

    elif SYNTHETIC_SAMPLE_MODE == "per_class_cap":
        synth_df = (
            synth_df.groupby(TARGET_COL, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), PER_CLASS_SYNTHETIC_CAP), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        print(
            "✔ Applied per-class synthetic cap: "
            f"maximum {PER_CLASS_SYNTHETIC_CAP:,} rows per class"
        )
    else:
        raise ValueError(f"Unsupported SYNTHETIC_SAMPLE_MODE: {SYNTHETIC_SAMPLE_MODE}")

    print(f"✔ Synthetic rows after sampling: {len(synth_df):,}")
    print(f"✔ Synthetic classes after sampling: {synth_df[TARGET_COL].nunique():,}")

    sampling_info = {
        "sampling_enabled": True,
        "sampling_mode": SYNTHETIC_SAMPLE_MODE,
        "synthetic_rows_before_filter": int(original_rows),
        "synthetic_rows_after_filter": int(filtered_rows),
        "synthetic_rows_after_sampling": int(len(synth_df)),
        "synthetic_classes_after_sampling": int(synth_df[TARGET_COL].nunique()),
    }

    if SYNTHETIC_SAMPLE_MODE == "global":
        sampling_info["global_sample_size"] = int(GLOBAL_SYNTHETIC_SAMPLE_SIZE)

    if SYNTHETIC_SAMPLE_MODE == "per_class_cap":
        sampling_info["per_class_cap"] = int(PER_CLASS_SYNTHETIC_CAP)

    return synth_df, sampling_info


def run_threshold(threshold_name):
    print(f"\n{'=' * 80}")
    print(f"Running Experiment 3 for {threshold_name}")
    print(f"{'=' * 80}")

    threshold_start = time.time()

    train_path = PROJECT_ROOT / "data" / "splits" / threshold_name / "train.csv"
    test_path = PROJECT_ROOT / "data" / "splits" / threshold_name / "test.csv"
    synth_path = PROJECT_ROOT / "data" / "processed" / "synthetic_preprocessed.csv"

    print("→ Checking required files...")
    if not train_path.exists():
        raise FileNotFoundError(f"Real train dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Real test dataset not found: {test_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {synth_path}")
    print("✔ All required files found")

    print("→ Loading datasets...")
    load_start = time.time()
    real_train_df = pd.read_csv(train_path)
    real_test_df = pd.read_csv(test_path)
    synth_df = pd.read_csv(synth_path)
    load_end = time.time()
    print(f"✔ Datasets loaded in {load_end - load_start:.2f} seconds")

    print("→ Validating target column...")
    for name, df in {
        "real_train": real_train_df,
        "real_test": real_test_df,
        "synthetic": synth_df
    }.items():
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in {name} dataset")
    print("✔ Target column validated")

    print("→ Aligning feature columns...")
    align_start = time.time()
    feature_cols, synth_df = align_features(real_train_df, synth_df, TARGET_COL)
    _, real_test_df = align_features(real_train_df, real_test_df, TARGET_COL)
    align_end = time.time()
    print(f"✔ Feature alignment completed in {align_end - align_start:.2f} seconds")

    train_classes_real_only = set(real_train_df[TARGET_COL].unique())

    synth_df, sampling_info = sample_synthetic_data(synth_df, train_classes_real_only)

    print("→ Building hybrid training dataset...")
    hybrid_start = time.time()
    hybrid_train_df = pd.concat(
        [real_train_df[feature_cols + [TARGET_COL]], synth_df],
        ignore_index=True
    )
    hybrid_end = time.time()
    print(f"✔ Hybrid dataset built in {hybrid_end - hybrid_start:.2f} seconds")

    X_train = hybrid_train_df[feature_cols]
    y_train = hybrid_train_df[TARGET_COL]

    X_test_full = real_test_df[feature_cols]
    y_test_full = real_test_df[TARGET_COL]

    print("→ Filtering test set to classes present in training...")
    filter_start = time.time()
    train_classes = set(y_train.unique())
    eval_mask = y_test_full.isin(train_classes)

    X_test = X_test_full.loc[eval_mask].copy()
    y_test = y_test_full.loc[eval_mask].copy()

    evaluated_classes = sorted(y_test.unique())
    filter_end = time.time()
    print(f"✔ Test filtering completed in {filter_end - filter_start:.2f} seconds")

    if len(evaluated_classes) == 0:
        raise ValueError(f"No evaluable classes found in test set for {threshold_name}")

    print("\nDATA SUMMARY")
    print(f"Real train rows:                {len(real_train_df):,}")
    print(f"Synthetic rows added:           {len(synth_df):,}")
    print(f"Hybrid train rows:              {len(hybrid_train_df):,}")
    print(f"Real test rows (full):          {len(y_test_full):,}")
    print(f"Real test rows used:            {len(y_test):,}")
    print(f"Training classes:               {y_train.nunique():,}")
    print(f"Test classes (full):            {y_test_full.nunique():,}")
    print(f"Evaluated classes:              {len(evaluated_classes):,}")
    print(f"Feature count:                  {len(feature_cols):,}")

    total_models = len(MODELS)
    failed_models = []

    for i, (model_name, build_fn) in enumerate(MODELS.items(), start=1):
        print(f"\n{'-' * 80}")
        print(f"[EX3] {threshold_name} | Model {i}/{total_models}: {model_name}")
        print(f"{'-' * 80}")

        model_total_start = time.time()

        try:
            print("→ Building model...")
            build_start = time.time()
            model = build_fn()
            build_end = time.time()
            print(f"✔ Model built in {build_end - build_start:.2f} seconds")

            print("→ Training started...")
            train_start = time.time()
            model.fit(X_train, y_train)
            train_end = time.time()
            print(f"✔ Training completed in {train_end - train_start:.2f} seconds")

            print("→ Predicting on test set...")
            pred_start = time.time()
            y_pred = model.predict(X_test)
            pred_end = time.time()
            print(f"✔ Prediction completed in {pred_end - pred_start:.2f} seconds")

            print("→ Computing metrics...")
            metrics_start = time.time()
            metrics = compute_metrics(y_test, y_pred)
            metrics_end = time.time()
            print(f"✔ Metrics computed in {metrics_end - metrics_start:.2f} seconds")

            metrics["experiment"] = "ex3_hybrid"
            metrics["threshold"] = threshold_name
            metrics["model"] = model_name
            metrics["real_train_rows"] = int(len(real_train_df))
            metrics["synthetic_rows"] = int(len(synth_df))
            metrics["hybrid_train_rows"] = int(len(hybrid_train_df))
            metrics["test_rows"] = int(len(y_test))
            metrics["total_classes_train"] = int(y_train.nunique())
            metrics["total_classes_test"] = int(y_test_full.nunique())
            metrics["evaluated_classes"] = int(len(evaluated_classes))
            metrics["feature_count"] = int(len(feature_cols))
            metrics["train_time_seconds"] = round(train_end - train_start, 4)
            metrics["predict_time_seconds"] = round(pred_end - pred_start, 4)
            metrics["total_model_time_seconds"] = round(time.time() - model_total_start, 4)
            metrics.update(sampling_info)

            print("\nRESULTS")
            print(f"Accuracy:        {metrics['accuracy']:.6f}")
            print(f"Macro Precision: {metrics['macro_precision']:.6f}")
            print(f"Macro Recall:    {metrics['macro_recall']:.6f}")
            print(f"Macro F1:        {metrics['macro_f1']:.6f}")
            print(f"Weighted F1:     {metrics['weighted_f1']:.6f}")

            print("→ Saving outputs...")
            save_start = time.time()
            output_dir = PROJECT_ROOT / "results" / "ex3_hybrid" / threshold_name / model_name
            save_outputs(output_dir, metrics, y_test, y_pred, evaluated_classes)
            save_end = time.time()
            print(f"✔ Outputs saved in {save_end - save_start:.2f} seconds")
            print(f"✔ Saved to: {output_dir}")

            model_total_end = time.time()
            print(f"✔ Total time for {model_name}: {model_total_end - model_total_start:.2f} seconds")

        except Exception as e:
            failed_models.append(model_name)

            error_dir = PROJECT_ROOT / "results" / "ex3_hybrid" / threshold_name / model_name
            error_dir.mkdir(parents=True, exist_ok=True)

            error_payload = {
                "experiment": "ex3_hybrid",
                "threshold": threshold_name,
                "model": model_name,
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "real_train_rows": int(len(real_train_df)),
                "synthetic_rows": int(len(synth_df)),
                "hybrid_train_rows": int(len(hybrid_train_df)),
                "test_rows": int(len(y_test)),
                "total_classes_train": int(y_train.nunique()),
                "total_classes_test": int(y_test_full.nunique()),
                "evaluated_classes": int(len(evaluated_classes)),
                "feature_count": int(len(feature_cols)),
            }
            error_payload.update(sampling_info)

            with open(error_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump(error_payload, f, indent=4)

            print("\n✘ MODEL FAILED")
            print(f"Model: {model_name}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print(f"✔ Error details saved to: {error_dir / 'error.json'}")
            print("→ Continuing to next model...")

    threshold_end = time.time()
    print(f"\n✅ Completed threshold {threshold_name} in {threshold_end - threshold_start:.2f} seconds")

    if failed_models:
        print("⚠ Failed models:", ", ".join(failed_models))
    else:
        print("✔ All models completed successfully for this threshold")


def main():
    overall_start = time.time()
    total_thresholds = len(THRESHOLDS)

    print(f"{'#' * 80}")
    print("STARTING EXPERIMENT 3: HYBRID")
    print(f"{'#' * 80}")
    print("Synthetic sampling settings:")
    print(f"  USE_SYNTHETIC_SAMPLING   = {USE_SYNTHETIC_SAMPLING}")
    print(f"  SYNTHETIC_SAMPLE_MODE    = {SYNTHETIC_SAMPLE_MODE}")
    print(f"  GLOBAL_SAMPLE_SIZE       = {GLOBAL_SYNTHETIC_SAMPLE_SIZE:,}")
    print(f"  PER_CLASS_SYNTHETIC_CAP  = {PER_CLASS_SYNTHETIC_CAP:,}")
    print(f"  RANDOM_STATE             = {RANDOM_STATE}")

    for i, threshold_name in enumerate(THRESHOLDS, start=1):
        print(f"\n{'#' * 80}")
        print(f"THRESHOLD {i}/{total_thresholds}: {threshold_name}")
        print(f"{'#' * 80}")
        run_threshold(threshold_name)

    overall_end = time.time()
    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENT 3 RUNS COMPLETED")
    print(f"Total runtime: {overall_end - overall_start:.2f} seconds")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()