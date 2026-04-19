"""
Description:
This script runs Experiment 3: Hybrid mineral classification.
This code was used for the feasibility testing of the research project itself
you may start from the very scrap or make reference of this code.

It trains machine learning models using only synthetic mineral data
and evaluates their performance on real measured test data. The goal
is to assess how well patterns learned from synthetic data transfer
to real-world samples.

Experiment Design:
- Training is performed exclusively on the synthetic dataset
- Testing is performed exclusively on the real measured test dataset
- Evaluation is restricted to the intersection of classes present
  in both synthetic training data and real test data

Purpose:
- Measure the generalization capability of synthetic data
- Quantify the domain transfer gap between synthetic and real datasets
- Establish whether synthetic data alone is sufficient for
  real-world mineral classification tasks

Implementation Notes:
- Feature columns between synthetic and real datasets are aligned
  before training and evaluation
- Classes not shared between synthetic and real data are excluded
  to ensure valid evaluation
- The real test set is never modified, augmented, or used during training

Output:
- Model performance metrics (accuracy, precision, recall, F1-score)
- Classification report
- Confusion matrix
- Saved results for comparison with other experiments
"""

from pathlib import Path
import json
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

MODELS = {
    "naive_bayes": build_naive_bayes,
    "knn": build_knn,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_outputs(output_dir, metrics, y_true, y_pred, labels):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")


def align_test_to_train_features(train_df, test_df, target_col):
    train_features = [c for c in train_df.columns if c != target_col]

    missing_in_test = [c for c in train_features if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test data is missing required feature columns: {missing_in_test}")

    test_aligned = test_df[train_features + [target_col]].copy()
    return train_features, test_aligned


def run_threshold(threshold_name):
    print(f"\n{'=' * 80}")
    print(f"Running Experiment 2 for {threshold_name}")
    print(f"{'=' * 80}")

    synth_path = PROJECT_ROOT / "data" / "processed" / "synthetic_preprocessed.csv"
    test_path = PROJECT_ROOT / "data" / "splits" / threshold_name / "test.csv"

    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {synth_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Real test dataset not found: {test_path}")

    synth_df = pd.read_csv(synth_path)
    test_df = pd.read_csv(test_path)

    if TARGET_COL not in synth_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in synthetic dataset")
    if TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in real test dataset")

    feature_cols, test_df = align_test_to_train_features(synth_df, test_df, TARGET_COL)

    synth_classes = set(synth_df[TARGET_COL].unique())
    test_classes = set(test_df[TARGET_COL].unique())
    common_classes = sorted(synth_classes.intersection(test_classes))

    if not common_classes:
        raise ValueError(f"No common classes found between synthetic train and real test for {threshold_name}")

    synth_eval_df = synth_df[synth_df[TARGET_COL].isin(common_classes)].copy()
    test_eval_df = test_df[test_df[TARGET_COL].isin(common_classes)].copy()

    X_train = synth_eval_df[feature_cols]
    y_train = synth_eval_df[TARGET_COL]

    X_test = test_eval_df[feature_cols]
    y_test = test_eval_df[TARGET_COL]

    print(f"Synthetic rows used for training: {len(synth_eval_df)}")
    print(f"Real test rows used for evaluation: {len(test_eval_df)}")
    print(f"Evaluated classes: {len(common_classes)}")

    for model_name, build_fn in MODELS.items():
        print(f"\n[EX2] {threshold_name} | {model_name}")

        model = build_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        metrics["experiment"] = "ex2_synth_to_real"
        metrics["threshold"] = threshold_name
        metrics["model"] = model_name
        metrics["train_rows"] = int(len(synth_eval_df))
        metrics["test_rows"] = int(len(test_eval_df))
        metrics["total_classes_synthetic"] = int(synth_df[TARGET_COL].nunique())
        metrics["total_classes_test"] = int(test_df[TARGET_COL].nunique())
        metrics["evaluated_classes"] = int(len(common_classes))

        output_dir = PROJECT_ROOT / "results" / "ex2_synth_to_real" / threshold_name / model_name
        save_outputs(output_dir, metrics, y_test, y_pred, common_classes)

        print(f"Accuracy:        {metrics['accuracy']:.6f}")
        print(f"Macro Precision: {metrics['macro_precision']:.6f}")
        print(f"Macro Recall:    {metrics['macro_recall']:.6f}")
        print(f"Macro F1:        {metrics['macro_f1']:.6f}")
        print(f"Weighted F1:     {metrics['weighted_f1']:.6f}")


def main():
    for threshold_name in THRESHOLDS:
        run_threshold(threshold_name)


if __name__ == "__main__":
    main()