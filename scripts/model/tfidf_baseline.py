"""
TF-IDF keyword baseline for phenotype classification, reproducing the MIPA paper's
(Supplemental File 2) methodology: a curated per-phenotype keyword dictionary (synonyms,
abbreviations, misspellings) scored via TF-IDF over discharge-summary text, with binary
predictions from percentile-based thresholds.

Per the paper: the TF-IDF matrix is fit on the VALIDATION set of gold-labeled discharge
summaries; thresholds (25th/50th/75th/90th percentile of the validation relevance-score
distribution) are then applied to the TEST set using that same fitted vectorizer. No
downstream classifier is trained -- the "model" is just this threshold rule.

Approximation note: the paper doesn't specify exact vectorizer settings or how multi-word
keyword phrases are tokenized/matched. Here, both keyword phrases and note text are
normalized (lowercase, non-alphanumeric -> space, whitespace-collapsed) and the vectorizer's
vocabulary is restricted to exactly that phenotype's (normalized) keyword phrases, with
ngram_range spanning 1..max_phrase_length so every phrase can appear as its own feature. The
"document-level relevance score" is then just the sum of TF-IDF weights across all of a
document's (phenotype-specific) vocabulary columns.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scripts.constants import PHENOTYPES, FILTERS_PATH, INDEXES_PATH

KEYWORDS_PATH = FILTERS_PATH + 'tfidf_keywords.json'
GOLDEN_NOTES_PATH = FILTERS_PATH + 'golden_notes.csv'
PERCENTILES = [25, 50, 75, 90]


def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_keywords():
    with open(KEYWORDS_PATH) as f:
        raw = json.load(f)
    return {ph: [normalize(k) for k in kws] for ph, kws in raw.items()}


def load_phenotype_split(phenotype):
    idx = pd.read_csv(f"{INDEXES_PATH}{phenotype}_indexes.csv")
    val = idx[idx['set'] == 'validation'][['hadm_id', 'y']].dropna()
    test = idx[idx['set'] == 'test'][['hadm_id', 'y']].dropna()
    return val, test


def run_phenotype(phenotype, keywords, notes_df):
    vocab = sorted(set(keywords[phenotype]))
    max_ngram = max(len(k.split()) for k in vocab)

    val_idx, test_idx = load_phenotype_split(phenotype)
    val = val_idx.merge(notes_df, on='hadm_id', how='inner')
    test = test_idx.merge(notes_df, on='hadm_id', how='inner')
    if val.empty or test.empty:
        print(f"[SKIPPING {phenotype}] empty val or test set after merging with notes")
        return None

    vectorizer = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, max_ngram),
                                  vocabulary=vocab, lowercase=False)
    val_matrix = vectorizer.fit_transform(val['text'])
    val_scores = np.asarray(val_matrix.sum(axis=1)).ravel()

    test_matrix = vectorizer.transform(test['text'])
    test_scores = np.asarray(test_matrix.sum(axis=1)).ravel()

    y_test = test['y'].astype(int).values

    results = []
    for pct in PERCENTILES:
        threshold = np.percentile(val_scores, pct)
        y_pred = (test_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        results.append({
            'phenotype': phenotype, 'percentile': pct, 'threshold': threshold,
            'n_val': len(val), 'n_test': len(test),
            'n_pos_test': int(y_test.sum()),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return results


def run_all(phenotypes=None, output_path='data/models/tfidf_baseline/tfidf_baseline_results.csv'):
    phenotypes = phenotypes or PHENOTYPES
    keywords = load_keywords()
    notes_df = pd.read_csv(GOLDEN_NOTES_PATH)[['hadm_id', 'text']]

    all_results = []
    for phenotype in phenotypes:
        print(f"[TF-IDF BASELINE] {phenotype}")
        res = run_phenotype(phenotype, keywords, notes_df)
        if res:
            all_results.extend(res)

    df = pd.DataFrame(all_results)
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    print("\n=== Macro-averaged F1 by percentile threshold ===")
    print(df.groupby('percentile')['f1'].agg(['mean', 'std', 'count']))
    return df


if __name__ == '__main__':
    run_all()
