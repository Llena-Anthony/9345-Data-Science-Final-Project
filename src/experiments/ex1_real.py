"""
Description:
This script runs Experiment 3: Hybrid mineral classification.
This code was used for the feasibility testing of the research project itself
you may start from the very scrap or make reference of this code.

It trains machine learning models using only real measured mineral data
and evaluates their performance on a held-out real test set. This serves
as the baseline for comparison against synthetic and hybrid approaches.

Experiment Design:
- Training is performed on real measured training data
- Testing is performed on real measured test data
- No synthetic data is used in this experiment

Purpose:
- Establish the baseline performance of models on real-world mineral data
- Provide a reference point for evaluating synthetic-only and hybrid models
- Measure how well models perform under purely real data conditions

Implementation Notes:
- The dataset is pre-split into training and testing sets per threshold
- Feature columns and target variable are separated before training
- No data augmentation or resampling is applied to the test set
- All classes present in the test set are evaluated

Output:
- Model performance metrics (accuracy, precision, recall, F1-score)
- Classification report
- Confusion matrix
- Saved results for comparison across experiments and thresholds
"""

# src/experiments/ex1_real.py

import os
from pathlib import Path
import pandas as pd

from src.models.naivebayes import build_naive_bayes
from src.models.knn import build_knn
from src.models.decisiontree import build_decision_tree
from src.models.randomforest import build_random_forest
from src.models.evaluate import evaluate_model, save_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THRESHOLDS = ["thr10", "thr20", "thr50"]
TARGET_COL = "mineral_name"

MODELS = {
    "naive_bayes": build_naive_bayes,
    "knn": build_knn,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
}

def run_experiment_for_threshold(threshold_name):
    split_dir = PROJECT_ROOT / "data" / "splits" / threshold_name
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    for model_name, build_fn in MODELS.items():
        print(f"\n[EX1] {threshold_name} - {model_name}")

        model = build_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        labels = sorted(y_test.unique())
        metrics, report, cm = evaluate_model(y_test, y_pred, labels=labels)

        metrics["experiment"] = "ex1_real"
        metrics["threshold"] = threshold_name
        metrics["train_rows"] = len(train_df)
        metrics["test_rows"] = len(test_df)
        metrics["total_classes_train"] = int(y_train.nunique())
        metrics["total_classes_test"] = int(y_test.nunique())
        metrics["evaluated_classes"] = len(labels)

        output_dir = PROJECT_ROOT / "results" / "ex1_real" / threshold_name / model_name
        save_results(output_dir, metrics, report, cm, class_labels=labels)

def main():
    for threshold_name in THRESHOLDS:
        run_experiment_for_threshold(threshold_name)

if __name__ == "__main__":
    main()