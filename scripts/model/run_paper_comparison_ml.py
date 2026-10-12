"""
Classic ML models matching the MIPA paper's set (LR, Naive Bayes, Random Forest, Gradient
Boosting -- KNN is an extra model in this repo not present in the paper, so excluded here for
an apples-to-apples comparison). Mirrors supervised_ML.py's train_all_models() call, just
without the knn_classifier entry.
"""
import os
import pandas as pd
from collections import defaultdict

from scripts.model.modeling import (train_all_lr, train_all_naive_bayes,
                                     train_all_random_forest, train_all_gradient_boosting)
from scripts.model.evaluate import evaluate_all
from scripts.constants import PHENOTYPES

OUTPUT_DIR = 'data/models/paper_comparison_ml'
OUTPUT_PATH = f'{OUTPUT_DIR}/classic_ml_results.csv'


def _per_type_path(model_type):
    return f'{OUTPUT_DIR}/classic_ml_{model_type}_results.csv'


def run_all(phenotypes=None, imputation_method_lr='median', imputation_method_tree='simple'):
    phenotypes = phenotypes or PHENOTYPES
    model_types = [
        ('lr', train_all_lr, {'imputation_method': imputation_method_lr}),
        ('naive_bayes', train_all_naive_bayes, {}),
        ('random_forest', train_all_random_forest, {'imputation_method': imputation_method_tree}),
        ('gradient_boosting', train_all_gradient_boosting, {'imputation_method': imputation_method_tree}),
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    for model_type, train_func, extra_kwargs in model_types:
        per_type_path = _per_type_path(model_type)
        # Resumable: each model family (LR with its ElasticNet CV grid search alone can take
        # over an hour across 16 phenotypes) is saved to its own file the moment it finishes,
        # instead of only at the very end -- an interruption (machine reboot, session kill)
        # partway through a later model family no longer loses already-completed ones.
        if os.path.exists(per_type_path):
            print(f"[SKIPPING {model_type}] already completed, loading {per_type_path}", flush=True)
            metrics_df = pd.read_csv(per_type_path)
            all_metrics.append(metrics_df)
            continue

        print(f"\n=== Training {model_type} models for all phenotypes ===", flush=True)
        models = train_func(feature_reduction=True, phenotypes=phenotypes, **extra_kwargs)
        metrics_df = evaluate_all(models, model_type=model_type)
        if metrics_df is not None and not metrics_df.empty:
            if 'model_type' not in metrics_df.columns:
                metrics_df['model_type'] = model_type
            metrics_df.to_csv(per_type_path, index=False)
            print(f"Saved {model_type} results to {per_type_path}", flush=True)
            all_metrics.append(metrics_df)
        print(f"=== {model_type} DONE ===", flush=True)

    combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved combined results to {OUTPUT_PATH}")
    return combined


if __name__ == '__main__':
    run_all()
    print("TASK COMPLETED")
