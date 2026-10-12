"""
ICD-count threshold baseline, reproducing the MIPA paper's rule-based classifier: for each
admission, count how many of its diagnosis ICD codes (ICD-9/ICD-10, matched version-agnostic
-- same semantics scripts/preprocessing/cohort.py::generate_silver_labels_df already uses for
silver labels) appear in that phenotype's curated code list, then binarize at >=1/>=2/>=3
codes. Per the paper, this is evaluated on the (gold-labeled) test set only.

data/filters/silver_labels.json is used as the code list -- verified to match the paper's
Supplemental File 1 ("Silver Labels") code-for-code (exact per-phenotype code counts), so no
separate code list needs to be sourced.
"""
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scripts.constants import PHENOTYPES, FILTERS_PATH, INDEXES_PATH
from scripts.utils import load_json, get_table_bigquery, generate_pivot_path, load_pivot_sparse

THRESHOLDS = [1, 2, 3]


def load_test_cohort_diagnoses():
    """Raw (unconverted, unfiltered) diagnosis codes for every test/val hadm_id, fetched once
    and shared across phenotypes -- the rule only needs icd_code, version-agnostic."""
    diag_pivot_path = generate_pivot_path(None, 'DIAG', train=False, noisy_label=False, use_shared_paths=True)
    cohort_ids = load_pivot_sparse(diag_pivot_path).index
    cohort_df = pd.DataFrame({'hadm_id': cohort_ids})
    diagnoses = get_table_bigquery('diagnoses_icd', cohort=cohort_df)
    return diagnoses[['hadm_id', 'icd_code']]


def load_phenotype_test_labels(phenotype):
    idx = pd.read_csv(f"{INDEXES_PATH}{phenotype}_indexes.csv")
    return idx[idx['set'] == 'test'][['hadm_id', 'y']].dropna()


def run_phenotype(phenotype, silver_codes, diagnoses):
    codes_for_phenotype = set(silver_codes[phenotype])
    matched = diagnoses[diagnoses['icd_code'].isin(codes_for_phenotype)]
    match_count = matched.groupby('hadm_id').size().rename('match_count')

    test = load_phenotype_test_labels(phenotype)
    test = test.merge(match_count, on='hadm_id', how='left')
    test['match_count'] = test['match_count'].fillna(0).astype(int)
    y_test = test['y'].astype(int).values

    results = []
    for threshold in THRESHOLDS:
        y_pred = (test['match_count'] >= threshold).astype(int).values
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        results.append({
            'phenotype': phenotype, 'threshold': threshold, 'n_test': len(test),
            'n_pos_test': int(y_test.sum()),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return results


def run_all(phenotypes=None, output_path='data/models/icd_threshold_baseline/icd_threshold_results.csv'):
    phenotypes = phenotypes or PHENOTYPES
    silver_codes = load_json(FILTERS_PATH + 'silver_labels.json')

    print("[ICD THRESHOLD BASELINE] Fetching raw diagnosis codes for test/val cohort...")
    diagnoses = load_test_cohort_diagnoses()
    print(f"  {len(diagnoses)} (hadm_id, icd_code) rows for {diagnoses['hadm_id'].nunique()} admissions")

    all_results = []
    for phenotype in phenotypes:
        print(f"[ICD THRESHOLD BASELINE] {phenotype}")
        all_results.extend(run_phenotype(phenotype, silver_codes, diagnoses))

    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    print("\n=== Macro-averaged F1 by threshold ===")
    print(df.groupby('threshold')['f1'].agg(['mean', 'std', 'count']))
    return df


if __name__ == '__main__':
    run_all()
