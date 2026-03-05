import pandas as pd
from scripts.constants import FILTERS_PATH # Assuming FILTERS_PATH is in constants

# Attempt to import the actual generate_silver_labels_df
# Fallback to a placeholder if not found (for isolated execution/linting)
_actual_generate_silver_labels_df_imported = True
try:
    from scripts.preprocessing.cohort import generate_silver_labels_df # Adjust path as needed
except ImportError:
    _actual_generate_silver_labels_df_imported = False
    print("WARNING: Actual generate_silver_labels_df not found. Using placeholder.")
    def generate_silver_labels_df(df_core_placeholder, phenotype_placeholder):
        print(f"[PLACEHOLDER] generate_silver_labels_df called for {phenotype_placeholder}")
        if isinstance(df_core_placeholder, pd.DataFrame) and 'hadm_id' in df_core_placeholder.columns:
            return df_core_placeholder[['hadm_id']].copy().assign(y=0)
        return pd.DataFrame(columns=['hadm_id', 'y'])

def phenotype_cohort_test(df_core, phenotype, val_frac=0.4):
    """
    Generates test and validation sets for a given phenotype based strictly on golden_labels.csv.
    The total number of hadm_ids in the output (test + val) will match those with non-null
    entries for the phenotype in golden_labels.csv.
    Stratifies positives and negatives (further stratifying negatives by silver label insight)
    into test and validation sets according to val_frac.
    The 'y' column in the output DataFrames is from golden_labels.csv.
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
        return pd.DataFrame(), pd.DataFrame()

    if phenotype not in golden_labels_all_phenotypes_df.columns:
        print(f"WARNING: Phenotype '{phenotype}' column not found in golden_labels.csv.")
        return pd.DataFrame(), pd.DataFrame()

    # 1. Define the universe: hadm_ids with non-null golden labels for this phenotype
    df_golden_for_phenotype = golden_labels_all_phenotypes_df[golden_labels_all_phenotypes_df[phenotype].notna()][['hadm_id', phenotype]]
    if df_golden_for_phenotype.empty:
        print(f"WARNING: No hadm_ids found in golden_labels.csv with non-null entries for phenotype '{phenotype}'.")
        return pd.DataFrame(), pd.DataFrame()
    
    df_golden_for_phenotype = df_golden_for_phenotype.rename(columns={phenotype: 'y'})
    df_golden_for_phenotype['y'] = pd.to_numeric(df_golden_for_phenotype['y'], errors='coerce').fillna(0).astype(int)

    # 2. Merge with df_core to get other necessary columns, but only for this golden set
    # Ensure df_core has hadm_id. If df_core is very large, select only necessary columns first.
    # essential_core_cols = ['hadm_id', 'subject_id', 'admittime', ...] # Define these
    # df_core_subset = df_core[essential_core_cols]
    # df_cohort_base = df_core_subset.merge(df_golden_for_phenotype, on='hadm_id', how='inner')
    df_cohort_base = df_core.merge(df_golden_for_phenotype, on='hadm_id', how='inner')

    if df_cohort_base.empty:
        print(f"WARNING: After merging golden hadm_ids for '{phenotype}' with df_core, the resulting cohort is empty. Check df_core content.")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Silver label insight (for stratifying negatives later)
    silver_labels_df_processed = pd.DataFrame(columns=['hadm_id', 'silver_label_y'])
    try:
        temp_silver_df = generate_silver_labels_df(df_core, phenotype) # Generate on broader df_core
        if isinstance(temp_silver_df, pd.DataFrame):
            # Standardize to 'hadm_id' and 'silver_label_y'
            if 'y' in temp_silver_df.columns:
                temp_silver_df = temp_silver_df.rename(columns={'y': 'silver_label_y'})
            if 'hadm_id' in temp_silver_df.columns and 'silver_label_y' in temp_silver_df.columns:
                silver_labels_df_processed = temp_silver_df[['hadm_id', 'silver_label_y']]
            elif 'hadm_id' in temp_silver_df.columns:
                 silver_labels_df_processed = temp_silver_df[['hadm_id']].copy()
                 silver_labels_df_processed['silver_label_y'] = 0
            else:
                print(f"WARNING: generate_silver_labels_df output for {phenotype} lacks hadm_id for silver label processing.")
    except Exception as e:
        print(f"WARNING: Exception ('{e}') during silver_labels_df generation for {phenotype}. Using empty silver_labels_df.")

    positive_silver_hadm_ids = set()
    if not silver_labels_df_processed.empty and 'hadm_id' in silver_labels_df_processed.columns and 'silver_label_y' in silver_labels_df_processed.columns:
        silver_labels_df_processed['silver_label_y'] = pd.to_numeric(silver_labels_df_processed['silver_label_y'], errors='coerce').fillna(0)
        positive_silver_hadm_ids = set(silver_labels_df_processed[silver_labels_df_processed['silver_label_y'] == 1]['hadm_id'])
    else:
        print(f"INFO: No effective silver label insights for {phenotype} to stratify negatives.")

    # 4. Split golden cohort into positives and negatives
    positives_full = df_cohort_base[df_cohort_base['y'] == 1]
    negatives_full = df_cohort_base[df_cohort_base['y'] == 0]

    # 5. Stratify positives into val/test
    pos_val = positives_full.sample(frac=val_frac, random_state=42) if not positives_full.empty else pd.DataFrame(columns=positives_full.columns)
    pos_test = positives_full.drop(pos_val.index) if not positives_full.empty else pd.DataFrame(columns=positives_full.columns)

    # 6. Stratify negatives into val/test (with silver label insight)
    neg_val_list = []
    neg_test_list = []

    neg_with_silver_insight = negatives_full[negatives_full['hadm_id'].isin(positive_silver_hadm_ids)]
    neg_without_silver_insight = negatives_full[~negatives_full['hadm_id'].isin(positive_silver_hadm_ids)]

    if not neg_with_silver_insight.empty:
        val_neg_silver = neg_with_silver_insight.sample(frac=val_frac, random_state=42)
        neg_val_list.append(val_neg_silver)
        neg_test_list.append(neg_with_silver_insight.drop(val_neg_silver.index))
    
    if not neg_without_silver_insight.empty:
        val_neg_no_silver = neg_without_silver_insight.sample(frac=val_frac, random_state=42)
        neg_val_list.append(val_neg_no_silver)
        neg_test_list.append(neg_without_silver_insight.drop(val_neg_no_silver.index))

    # 7. Combine to form final val and test sets
    df_core_val_list = [pos_val]
    df_core_test_list = [pos_test]

    if neg_val_list:
        df_core_val_list.extend(neg_val_list)
    if neg_test_list:
        df_core_test_list.extend(neg_test_list)
    
    df_core_val = pd.concat(df_core_val_list).sample(frac=1, random_state=42).reset_index(drop=True) if len(df_core_val_list) > 1 or (len(df_core_val_list)==1 and not df_core_val_list[0].empty) else pd.DataFrame(columns=df_cohort_base.columns)
    df_core_test = pd.concat(df_core_test_list).sample(frac=1, random_state=42).reset_index(drop=True) if len(df_core_test_list) > 1 or (len(df_core_test_list)==1 and not df_core_test_list[0].empty) else pd.DataFrame(columns=df_cohort_base.columns)

    print(f"Phenotype '{phenotype}': Golden universe size: {len(df_golden_for_phenotype)}, Cohort base size: {len(df_cohort_base)}")
    print(f"Val set size: {len(df_core_val)}, Test set size: {len(df_core_test)}")
    if (len(df_core_val) + len(df_core_test)) != len(df_cohort_base):
        print(f"WARNING: Sum of val ({len(df_core_val)}) and test ({len(df_core_test)}) sets does not equal cohort base size ({len(df_cohort_base)}) for {phenotype}.")

    return df_core_test, df_core_val

# Clean up placeholder if actual import failed
if not _actual_generate_silver_labels_df_imported:
    del generate_silver_labels_df 