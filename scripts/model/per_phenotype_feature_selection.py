"""
Per-phenotype feature selection, matching the MIPA paper: "percentile thresholds were computed
within each phenotypic cohort" (main text) -- i.e. the 25th-percentile (structured: ICD/
procedure/meds/chart/labs) and 10th-percentile (UMLS CUI) frequency cutoffs are each computed
using ONLY that phenotype's own silver-labeled, 50/50-balanced training cohort, not the full
~394,758-admission shared training population every phenotype currently draws its feature
selection from (see generate_features_selection(use_shared_paths=True)).

This does NOT touch the shared pipeline (data/processed/cohort/shared_train/) -- it builds a
parallel, phenotype-specific tree under data/processed/cohort/train/<phenotype>/ (the legacy
per-phenotype path convention that scripts/utils.py's path helpers already support via
use_shared_paths=False), by filtering each shared long-format table down to the phenotype's own
balanced cohort before re-running the existing generate_features_selection/generate_summary/
generate_pivot functions (phenotype=<name>, use_shared_paths=False) -- those already do the
correct per-call quantile computation, this just gives them a smaller, phenotype-scoped input.

Known limitation for DSNOTES/RADNOTES: the shared NLP long files were already CUI-frequency-
filtered once (at a 20th-percentile threshold, not scoped per phenotype) when MedCAT annotation
ran -- that's the expensive step (~3.7 days serial) this deliberately avoids re-running. Running
a fresh 10th-percentile selection on top of that here is still phenotype-scoped (closer to the
paper than the shared version), but can only ever be a subset of what a from-scratch per-
phenotype annotation would keep -- CUIs the original 20th-percentile cut already dropped can't
be recovered without re-annotating.
"""
import os
import pandas as pd

from scripts.constants import PHENOTYPES, INDEXES_PATH, PROCESSED_PATH
from scripts.utils import (generate_file_path, generate_nlp_file_path, read_gzip_or_parts,
                            save_gzip, path_exists)
from scripts.featureselection.feature_selection import generate_features_selection, generate_summary
from scripts.featureselection.pivot import generate_pivot

STRUCTURED_TYPES = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
NLP_TYPES = ['DSNOTES', 'RADNOTES']
MARKER_DIR = 'data/processed/cohort/train'

# generate_nlp_file_path(..., use_shared_paths=True) points at data/processed/cohort/shared_train/
# long/NLP/, which does not exist for the current MIMIC-IV 2.2 cohort: the NLP reindexing done
# earlier this session only reindexed and overwrote the PIVOT outputs directly (load_pivot_sparse /
# reindex_sparse_df / save_pivot), it never recreated this long-format (pre-pivot, one row per note
# with a raw nlp_codes string) intermediate at the 2.2 path. The only place this long-format file
# still exists on disk is the pre-migration backup, which was confirmed (earlier this session, during
# the reindex) to have zero hadm_ids outside the current 2.2 cohort -- i.e. every hadm_id in this file
# is safe to filter against any phenotype's 2.2-based cohort_ids, even though the file predates 2.2.
NLP_LONG_BACKUP_PATH = f"{PROCESSED_PATH}cohort/shared_train_v3_1_backup/long/NLP/"


def build_phenotype_cohort_ids(phenotype, random_state=42):
    """The phenotype's own silver-labeled training cohort, balanced 50/50 (same downsampling
    logic as train_all_lr's balance_training) -- this is the "phenotypic cohort" the paper
    computes percentile thresholds within."""
    idx = pd.read_csv(f"{INDEXES_PATH}{phenotype}_indexes.csv")
    train_idx = idx[idx['set'] == 'train']
    pos = train_idx.loc[train_idx['y'] == 1, 'hadm_id']
    neg = train_idx.loc[train_idx['y'] == 0, 'hadm_id']
    n = min(len(pos), len(neg))
    pos_keep = pos if len(pos) == n else pos.sample(n=n, random_state=random_state)
    neg_keep = neg if len(neg) == n else neg.sample(n=n, random_state=random_state)
    print(f"[{phenotype}] balanced cohort: {len(pos)} pos / {len(neg)} neg -> {n} / {n}", flush=True)
    return set(pos_keep) | set(neg_keep)


def materialize_phenotype_long_files(phenotype, cohort_ids):
    for df_type in STRUCTURED_TYPES:
        shared_path = generate_file_path(None, df_type, train=True, noisy_label=True, long=True, use_shared_paths=True)
        if not path_exists(shared_path):
            print(f"[SKIP {df_type}] shared long file missing: {shared_path}", flush=True)
            continue
        df = read_gzip_or_parts(shared_path)
        df = df[df['hadm_id'].isin(cohort_ids)]
        dest_path = generate_file_path(phenotype, df_type, train=True, noisy_label=True, long=True, use_shared_paths=False)
        save_gzip(df, dest_path)
        print(f"[{phenotype}] {df_type}: {len(df)} rows -> {dest_path}", flush=True)

    for df_type in NLP_TYPES:
        shared_nlp_path = f"{NLP_LONG_BACKUP_PATH}{df_type}_shared_train.csv.gz"
        if not path_exists(shared_nlp_path):
            print(f"[SKIP {df_type}] shared NLP file missing: {shared_nlp_path}", flush=True)
            continue
        df = read_gzip_or_parts(shared_nlp_path)
        df = df[df['hadm_id'].isin(cohort_ids)]
        dest_path = generate_nlp_file_path(phenotype, df_type, train=True, noisy_label=True, use_shared_paths=False)
        save_gzip(df, dest_path)
        print(f"[{phenotype}] {df_type}: {len(df)} rows -> {dest_path}", flush=True)


def run_phenotype(phenotype):
    print(f"=== {phenotype}: building per-phenotype cohort ===", flush=True)
    cohort_ids = build_phenotype_cohort_ids(phenotype)
    materialize_phenotype_long_files(phenotype, cohort_ids)

    generate_features_selection(phenotype, train=True, noisy_label=True, notes=True, NLP=True,
                                 use_shared_paths=False,
                                 lower_labs=0.25, lower_chart=0.25, lower_icd=0.25, lower_meds=0.25, lower_nlp=0.10)
    generate_summary(phenotype, train=True, noisy_label=True, notes=True, NLP=True, use_shared_paths=False)
    generate_pivot(phenotype, train=True, noisy_label=True, notes=True, NLP=True, use_shared_paths=False)
    print(f"=== {phenotype} DONE ===", flush=True)


if __name__ == '__main__':
    for p in PHENOTYPES:
        marker = f"{MARKER_DIR}/{p}/.per_phenotype_fs_done"
        if os.path.exists(marker):
            print(f"[SKIPPING {p}] already done (marker found)", flush=True)
            continue
        run_phenotype(p)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, 'w').close()
    print('ALL PHENOTYPES DONE', flush=True)
