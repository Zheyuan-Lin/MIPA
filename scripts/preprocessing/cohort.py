import pandas as pd
import warnings

from scripts.utils import load_csv, load_json, ensure_pandas_df
from scripts.constants import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def phenotype_list():
 return sorted(["c_diff_complication", "c_diff_past", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2",
         "hfref", "hfpef", "hypertension", "sle", "metastatic_cancer", "obesity",
         "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"])

def generate_baseline_cohort(df_patients: pd.DataFrame,
                             df_admissions: pd.DataFrame) -> pd.DataFrame:
    """
    This function merges the admissions and patients dataframes on the patient id and
    returns the resulting dataframe.

    Parameters
    ----------
    df_patients: pd.DataFrame
        The patients dataframe containing patient demographics data
    df_admissions: pd.DataFrame
        The admissions dataframe containing patient admission and discharge information.

    Returns
    -------
    pd.DataFrame
        The merged dataframe containing patient demographics, clinical data,
        and admission information.
    """
    # Merge the admissions and patients dataframes
    df_core = df_admissions.merge(df_patients, how='left', on='subject_id')
    df_core = ensure_pandas_df(df_core)

    # Keeping only the columns of interest
    to_keep = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_location',
    'insurance', 'language', 'marital_status', 'race', 'gender', 'anchor_age', 'anchor_year', 'dod']
    df_core = df_core[to_keep]

    # Basic data cleaning
    ### Renaming
    df_core = df_core.rename(columns={'anchor_age': 'age'})

    ### Feature engineering
    df_core['los'] = (df_core['dischtime'] - df_core['admittime']).dt.total_seconds() / 3600
    df_core['death'] = df_core['dod'].notnull().astype(int)
    #### Merge racial categories in column 'race'
    df_core['race'] = df_core['race'].apply(
            lambda x: 'WHITE - OTHER EUROPEAN' if x == 'WHITE' else
            'BLACK/AFRICAN AMERICAN' if x in ['BLACK', 'BLACK/CARIBBEAN ISLAND'] else
            'HISPANIC OR LATINO' if x == 'HISPANIC' else x
    )
    #### Replace '?' in language column with null
    df_core['language'] = df_core['language'].replace('?', None)

    ### Feature selection
    # Sanity check
    ### Remove duplicates hadm_id
    df_core = df_core.drop_duplicates(subset=['hadm_id']) # Ensure compute() is called when the DataFrame is ready
    
    print(f"--- generate_baseline_cohort: Unique hadm_ids BEFORE age filter: {df_core['hadm_id'].nunique()}")
    ### Remove patients with age less than 18 or greater than 89
    df_core = df_core[(df_core['age'] > 18) & (df_core['age'] <= 89)]
    print(f"--- generate_baseline_cohort: Unique hadm_ids AFTER age filter: {df_core['hadm_id'].nunique()}")

    # NULL values not handled
    return df_core
def generate_icu_admission_status(df_core: pd.DataFrame, df_icustays: pd.DataFrame) -> pd.DataFrame:
    """
    This function merges the icustays and core dataframes on the subject_id column and
    adds a new column indicating whether the patient was admitted to the ICU or not.

    Parameters
    ----------
    df_core: pd.DataFrame
        The core dataframe containing patient demographics and clinical data.
    df_icustays: pd.DataFrame
        The icustays dataframe containing patient ICU stay information.

    Returns
    -------
    pd.DataFrame
        The merged dataframe containing patient demographics, clinical data,
        and ICU admission status.
    """
    # Make sure to merge based on subject_id column
    df_core = df_core.merge(df_icustays, how='left', on='subject_id')
    # Add a new column indicating whether the patient was admitted to the ICU or not
    df_core['icu_admission'] = df_core.groupby('subject_id')['stay_id'].transform(
        'count').apply(lambda x: 1 if x > 0 else 0)
    return df_core

def generate_addition_icdcodes(df_core: pd.DataFrame, df_diagnoses_icd: pd.DataFrame) -> pd.DataFrame:
    """
    This function merges the diagnoses_icd and core dataframes on the subject_id column
    :param df_core:
    :param df_diagnoses_icd:
    :return:
    """
    #remove seq_num column from df_diagnoses_icd
    df_diagnoses_icd = df_diagnoses_icd.drop(columns=['seq_num'])
    # Step 1: Calculate unique admissions per subject_id
    admissions_count = df_diagnoses_icd.groupby('subject_id')[
        'hadm_id'].nunique().reset_index(name='admissions_number')
    # Step 2: Merge this information back into the original dataframe
    df_diagnoses_icd = df_diagnoses_icd.merge(admissions_count, on='subject_id')
    df_core = df_core.merge(df_diagnoses_icd, how='left',  on=['subject_id', 'hadm_id'])

    return df_core

def generate_silver_labels_df(df_core, phenotype):
    """
    Rule-based silver labeling: an admission (hadm_id) is labeled y=1 if any of its
    diagnosis ICD codes (ICD-9 or ICD-10, mixed together) appear in the code list for
    `phenotype` in data/filters/silver_labels.json, else y=0. No denoising/confidence
    weighting is applied — this mirrors the administrative-code rule mapping MIPA calls
    "silver" labels (see data/filters/silver_labels.json and golden_labels.csv for the
    human-adjudicated benchmark used to evaluate against this noise).

    :param df_core: pd.DataFrame with at least ['subject_id', 'hadm_id', 'icd_code'] columns
                     (one row per diagnosis code per admission).
    :param phenotype: str - phenotype key in silver_labels.json.
    :return: pd.DataFrame with one row per (subject_id, hadm_id) and a 'y' column (0/1).
    """
    silver_codes = load_json(FILTERS_PATH + 'silver_labels.json')
    codes_for_phenotype = set(silver_codes[phenotype])

    matched = df_core['icd_code'].isin(codes_for_phenotype)
    y_by_hadm = (
        df_core.assign(_match=matched)
               .groupby(['subject_id', 'hadm_id'])['_match']
               .max()
               .astype(int)
               .rename('y')
               .reset_index()
    )
    # Keep one row per admission with its other columns (admittime, dischtime, age, ...)
    # intact -- downstream steps (preproc_meds, preproc_proc, ...) need them -- and merge
    # the derived label in, rather than collapsing df_core down to just the label.
    df_labels = df_core.drop_duplicates(subset=['subject_id', 'hadm_id']).merge(
        y_by_hadm, on=['subject_id', 'hadm_id'], how='left'
    )

    return df_labels

def phenotype_cohort_train(df_core, phenotype, silver_labels=False):
    """
    Filters the core dataframe to include only patients with the specified phenotype.
    If silver_labels is True, returns cohorts of patients with a prespecified list of ICD codes.

    :param df_core: pd.DataFrame - Core patient data.
    :param phenotype: str - The target phenotype.
    :param silver_labels: bool - If True, use silver labels for noisy labeling.
    :return: pd.DataFrame - Filtered and labeled patient data.
    """
    phenotypes = ["c_diff_complication", "c_diff_past", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2",
         "hfref", "hfpef", "hypertension", "sle", "metastatic_cancer", "obesity",
         "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"]
    if phenotype not in phenotypes:
        raise ValueError(f'Phenotype not in list of phenotypes: {phenotypes}')

    # Exclude test subjects
    to_exclude = pd.read_csv(FILTERS_PATH + 'test_subject_ids.csv')
    df_core = df_core.merge(to_exclude, on='subject_id', how='left', indicator=True)
    df_core = df_core[df_core['_merge'] == 'left_only'].drop(columns=['_merge'])

    if silver_labels:
        df_core = generate_silver_labels_df(df_core, phenotype)

    else:
        print(f"You extracted the phenotype {phenotype} without using silver labels. The entire cohort was returned "
              f"excluding the test set. You can define noisy labels for training based on your own criteria.")

    # Return hadm_ids only
    df_core = df_core.drop_duplicates(subset=['subject_id', 'hadm_id'])
    columns_to_remove = ['admission_location', 'insurance', 'icd_code', 'icd_version']
    df_core = df_core.drop(columns=columns_to_remove, errors='ignore')

    return df_core


def phenotype_cohort_test(df_core, phenotype, val_frac=0.4):
    """
    Generates test and validation sets for a given phenotype based strictly on golden_labels.csv.
    The total number of hadm_ids in the output (test + val) will match those with non-null
    entries for the phenotype in golden_labels.csv.
    Stratifies positives and negatives into test and validation sets according to val_frac.
    The 'y' column in the output DataFrames is from golden_labels.csv.
    Silver label logic is temporarily removed for simplification.
    """
    valid_phenotypes_list = [
        "c_diff_complication", "c_diff_past", "dementia", "depression",
        "diabetes_type_1", "diabetes_type_2", "hfref", "hfpef", "hypertension",
        "sle", "metastatic_cancer", "obesity", "rheumatoid_arthritis",
        "alcohol_abuse", "vte_complication", "vte_past"
    ]

    if phenotype not in valid_phenotypes_list:
        raise ValueError(f"Phenotype '{phenotype}' not in list of recognized phenotypes.")

    try:
        golden_labels_all_phenotypes_df = pd.read_csv(FILTERS_PATH + 'golden_labels.csv')
    except FileNotFoundError:
        print(f"ERROR: golden_labels.csv not found at {FILTERS_PATH}. Cannot proceed for {phenotype}.")
        return pd.DataFrame(columns=df_core.columns), pd.DataFrame(columns=df_core.columns) # Return empty DFs with df_core columns

    if 'hadm_id' not in golden_labels_all_phenotypes_df.columns:
        print(f"ERROR: 'hadm_id' column not found in golden_labels.csv.")
        return pd.DataFrame(columns=df_core.columns), pd.DataFrame(columns=df_core.columns)

    if phenotype not in golden_labels_all_phenotypes_df.columns:
        print(f"WARNING: Phenotype column '{phenotype}' not found in golden_labels.csv.")
        return pd.DataFrame(columns=df_core.columns), pd.DataFrame(columns=df_core.columns)

    # 1. Define the universe: hadm_ids with non-null golden labels for this phenotype
    # Ensure 'hadm_id' is the first column for clarity if it's not already.
    df_golden_phenotype_specific = golden_labels_all_phenotypes_df[['hadm_id', phenotype]].copy()
    df_golden_phenotype_specific = df_golden_phenotype_specific[df_golden_phenotype_specific[phenotype].notna()]

    if df_golden_phenotype_specific.empty:
        print(f"WARNING: No hadm_ids found in golden_labels.csv with non-null entries for phenotype '{phenotype}'.")
        return pd.DataFrame(columns=df_core.columns), pd.DataFrame(columns=df_core.columns)

    # Rename phenotype column to 'y' and ensure it's integer
    df_golden_phenotype_specific = df_golden_phenotype_specific.rename(columns={phenotype: 'y'})
    df_golden_phenotype_specific['y'] = pd.to_numeric(df_golden_phenotype_specific['y'], errors='coerce').fillna(0).astype(int)
    df_golden_phenotype_specific = df_golden_phenotype_specific.drop_duplicates(subset=['hadm_id'])
    
    initial_golden_count = len(df_golden_phenotype_specific)
    print(f"Phenotype '{phenotype}': Initial unique golden hadm_ids: {initial_golden_count}")

    # 2. Merge with df_core: Left merge from golden_labels to df_core to keep only golden hadm_ids
    # This ensures we only work with hadm_ids from golden_labels and bring in other features from df_core.
    # The 'y' column from df_golden_phenotype_specific will overwrite any 'y' in df_core if names clash after merge.
    # It's safer to ensure 'y' from golden is distinct or handle suffix. Here, 'y' is already from golden.
    
    # Select essential columns from df_core to avoid large merge if df_core is huge.
    # For now, assuming df_core is manageable or has been pre-filtered.
    # If df_core has a 'y' column, it might conflict. Golden 'y' is authoritative.
    # Ensure df_core 'hadm_id' is compatible type.
    if 'hadm_id' not in df_core.columns:
        print(f"ERROR: 'hadm_id' column missing in df_core. Cannot proceed.")
        return pd.DataFrame(columns=df_golden_phenotype_specific.columns.tolist() + ['some_core_col']), pd.DataFrame(columns=df_golden_phenotype_specific.columns.tolist() + ['some_core_col'])


    # Perform the merge. `how='inner'` on unique golden hadm_ids acts like a filter.
    # If df_core also has a 'y' column, pandas adds suffixes (_x, _y). We want the one from golden.
    df_cohort_base = pd.merge(df_core, df_golden_phenotype_specific, on='hadm_id', how='inner')

    if df_cohort_base.empty:
        print(f"WARNING: After merging golden hadm_ids for '{phenotype}' with df_core, the resulting cohort is empty. Check df_core content and hadm_id matching.")
        return pd.DataFrame(columns=df_core.columns.tolist() + ['y']), pd.DataFrame(columns=df_core.columns.tolist() + ['y'])
    
    # Ensure 'y' from golden_labels is the one used. If merge created 'y_y', rename it.
    # If 'y_x' (from df_core) exists and 'y_y' (from golden) exists, drop 'y_x' and rename 'y_y' to 'y'.
    if 'y_y' in df_cohort_base.columns and 'y_x' in df_cohort_base.columns:
        df_cohort_base = df_cohort_base.drop(columns=['y_x'])
        df_cohort_base = df_cohort_base.rename(columns={'y_y': 'y'})
    elif 'y_y' in df_cohort_base.columns and 'y' not in df_cohort_base.columns: # Only y_y if original df_core had no 'y'
        df_cohort_base = df_cohort_base.rename(columns={'y_y': 'y'})
    # If df_core had no 'y' and golden had 'y', it will just be 'y'. No action needed.
    # If df_core had 'y' and golden had 'y', and 'y_x', 'y_y' were created, it's handled.

    # At this point, df_cohort_base should contain only hadm_ids from golden_labels,
    # with features from df_core and the authoritative 'y' from golden_labels.
    # The number of unique hadm_ids in df_cohort_base should match initial_golden_count.
    if len(df_cohort_base['hadm_id'].unique()) != initial_golden_count:
        print(f"WARNING: Discrepancy in hadm_id count after merge for {phenotype}. "
              f"Expected: {initial_golden_count}, Got: {len(df_cohort_base['hadm_id'].unique())}")
        # This could happen if df_core had duplicate hadm_ids that matched a single golden hadm_id.
        # Or if some golden_hadm_ids were not in df_core.
        # For this function's contract, we must restrict to the initial_golden_count.
        # Re-filter to ensure this, though ideally the merge handles it if inputs are clean.
        df_cohort_base = df_cohort_base[df_cohort_base['hadm_id'].isin(df_golden_phenotype_specific['hadm_id'].unique())]
        # And ensure unique hadm_ids if df_core had duplicates
        df_cohort_base = df_cohort_base.drop_duplicates(subset=['hadm_id'], keep='first') # Or decide on a strategy for keep

    # 3. Silver label logic is REMOVED for simplification.

    # 4. Split golden cohort (df_cohort_base) into positives and negatives based on the golden 'y'
    positives_full = df_cohort_base[df_cohort_base['y'] == 1]
    negatives_full = df_cohort_base[df_cohort_base['y'] == 0]

    # 5. Stratify positives into val/test
    pos_val = positives_full.sample(frac=val_frac, random_state=42) if not positives_full.empty else pd.DataFrame(columns=positives_full.columns)
    pos_test = positives_full.drop(pos_val.index) if not positives_full.empty else pd.DataFrame(columns=positives_full.columns)

    # 6. Stratify negatives into val/test
    neg_val = negatives_full.sample(frac=val_frac, random_state=42) if not negatives_full.empty else pd.DataFrame(columns=negatives_full.columns)
    neg_test = negatives_full.drop(neg_val.index) if not negatives_full.empty else pd.DataFrame(columns=negatives_full.columns)
    
    # 7. Combine to form final val and test sets
    df_core_val_list = []
    if not pos_val.empty: df_core_val_list.append(pos_val)
    if not neg_val.empty: df_core_val_list.append(neg_val)

    df_core_test_list = []
    if not pos_test.empty: df_core_test_list.append(pos_test)
    if not neg_test.empty: df_core_test_list.append(neg_test)

    df_core_val = pd.concat(df_core_val_list).sample(frac=1, random_state=42).reset_index(drop=True) if df_core_val_list else pd.DataFrame(columns=df_cohort_base.columns)
    df_core_test = pd.concat(df_core_test_list).sample(frac=1, random_state=42).reset_index(drop=True) if df_core_test_list else pd.DataFrame(columns=df_cohort_base.columns)

    print(f"Phenotype '{phenotype}': Golden universe size (unique hadm_ids from golden_labels): {initial_golden_count}")
    print(f"Val set size: {len(df_core_val)} (Unique hadm_ids: {df_core_val['hadm_id'].nunique() if not df_core_val.empty else 0})")
    print(f"Test set size: {len(df_core_test)} (Unique hadm_ids: {df_core_test['hadm_id'].nunique() if not df_core_test.empty else 0})")
    
    current_total_unique_hadm_ids = (df_core_val['hadm_id'].nunique() if not df_core_val.empty else 0) + \
                                    (df_core_test['hadm_id'].nunique() if not df_core_test.empty else 0)

    # Assert that the total number of unique hadm_ids across val and test equals the initial golden count
    if current_total_unique_hadm_ids != initial_golden_count:
        print(f"CRITICAL WARNING for {phenotype}: Sum of unique hadm_ids in val ({df_core_val['hadm_id'].nunique() if not df_core_val.empty else 0}) "
              f"and test ({df_core_test['hadm_id'].nunique() if not df_core_test.empty else 0}) sets ({current_total_unique_hadm_ids}) "
              f"does NOT equal initial golden count ({initial_golden_count}).")
        # This check is vital. If it fails, something is wrong with the splitting or merging logic.
        
    # Ensure no overlap between val and test sets
    if not df_core_val.empty and not df_core_test.empty:
        val_ids = set(df_core_val['hadm_id'])
        test_ids = set(df_core_test['hadm_id'])
        overlap = val_ids.intersection(test_ids)
        if overlap:
            print(f"CRITICAL WARNING for {phenotype}: Overlap of {len(overlap)} hadm_ids between validation and test sets: {list(overlap)[:5]}")


    # Pre-existing bug: this used to return (df_core_test, df_core_val) while every caller
    # unpacks as `val_df, test_df = phenotype_cohort_test(...)` -- invisible at the old
    # pos_split=0.5 default (both sets the same SIZE, so only their CONTENT was swapped, no
    # overlap, nothing looked wrong), but surfaced immediately once val_frac=0.4 made the two
    # sets different sizes (every phenotype's saved index file had 'validation' holding the
    # larger ~60% share and 'test' the smaller ~40% share -- backwards).
    return df_core_val, df_core_test

def generate_cohort_train_test(df_core, phenotype):
    phenotypes = ["c_diff_complication", "c_diff_past", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2",
         "hfref", "hfpef", "hypertension", "sle", "metastatic_cancer", "obesity",
         "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"]
    # Verify that the phenotype is in the list of phenotypes
    if phenotype not in phenotypes:
        raise ValueError(f'Phenotype not in list of phenotypes : {phenotypes}')

    return phenotype_cohort_train(df_core, phenotype, silver_labels=True), phenotype_cohort_test(df_core, phenotype)
