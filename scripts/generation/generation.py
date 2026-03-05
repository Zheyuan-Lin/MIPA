from scripts.preprocessing.cohort import *
from scripts.preprocessing.hosp.hosp_preprocess import *
from scripts.constants import *
from scripts.utils import *
import os
import dask as dd
# os.chdir('/Users/eyamga/Documents/Medecine/Recherche/Maitrise MSc/Thèse/MIPA-Git/MIMIC-IV-Phenotype-Atlas')



def load_data(sample=None):
    """Load multiple tables from MIMIC-IV and return as a dictionary of DataFrames."""
    tables = ['admissions', 'transfers', 'patients', 'icustays', 'diagnoses_icd']
    data = {}
    for table in tables:
        try:
            compute = False if is_gcs_path(CORE_MIMICIV_PATH) else True
            # Load data, assuming it returns a dictionary with the table name as a key
            loaded_data = load_mimiciv(CORE_MIMICIV_PATH, NOTE_MIMICIV_PATH, tables_to_return=[table],
                                       n_sample=sample, compute=compute)
            # Extract the DataFrame from the nested dictionary
            data[table] = loaded_data[table] if table in loaded_data else pd.DataFrame()
            print(f"Loaded {table}, type: {type(data[table])}")  # Debugging line
        except Exception as e:
            print(f"Error loading data for {table}: {e}")
            data[table] = pd.DataFrame()  # Create an empty DataFrame if loading fails
    return data
def merge_and_process_data(data):
    """Merge and process data to create a baseline cohort."""
    df_base_core = generate_baseline_cohort(data['patients'], data['admissions'])
    df_base_core = generate_icu_admission_status(df_base_core, data['icustays'])
    df_base_core = cleanup_merged_dataframe(df_base_core)

    print(f"--- merge_and_process_data: Unique hadm_ids BEFORE generate_addition_icdcodes & dropna: {df_base_core['hadm_id'].nunique()}")
    df_temp_before_dropna = generate_addition_icdcodes(df_base_core, data['diagnoses_icd'])
    print(f"--- merge_and_process_data: Unique hadm_ids AFTER generate_addition_icdcodes (BEFORE dropna): {df_temp_before_dropna['hadm_id'].nunique()}")
    
    # Check for rows that would be dropped by a blanket .dropna()
    newly_added_cols_by_icd_merge = ['icd_code', 'icd_version', 'admissions_number'] 
    if all(col in df_temp_before_dropna.columns for col in newly_added_cols_by_icd_merge):
        rows_with_nan_in_new_cols = df_temp_before_dropna[df_temp_before_dropna[newly_added_cols_by_icd_merge].isnull().any(axis=1)]
        if not rows_with_nan_in_new_cols.empty:
            print(f"--- merge_and_process_data: Number of rows with any NaN in {newly_added_cols_by_icd_merge} (candidates for dropna): {len(rows_with_nan_in_new_cols)}")
            print(f"--- merge_and_process_data: Unique hadm_ids in those NaN rows: {rows_with_nan_in_new_cols['hadm_id'].nunique()}")
        else:
            print(f"--- merge_and_process_data: No rows found with NaNs in {newly_added_cols_by_icd_merge} that would be dropped by .dropna() on these columns.")
    else:
        print(f"--- merge_and_process_data: One or more expected columns ({newly_added_cols_by_icd_merge}) not found after generate_addition_icdcodes.")

    # df_base_core = df_temp_before_dropna.dropna() # Temporarily remove or comment out this line
    df_base_core = df_temp_before_dropna # Keep all rows for now
    print(f"--- merge_and_process_data: Unique hadm_ids AFTER generate_addition_icdcodes & SKIPPING dropna: {df_base_core['hadm_id'].nunique()}")
    return df_base_core

def process_and_save_shared_test_val_data(master_test_val_cohort, noisy_label=False, time=True, notes=True,
                                          root_icd9=True, group_icd10=True, diag_options='ICD10',
                                          med_grouping=True, debug=None):
    """
    Processes and saves all shared data types (DIAG, MEDS, etc.) for the combined test/validation cohort.
    """
    print("[PROCESSING SHARED TEST/VALIDATION DATA]")
    # Define the data types and corresponding processing functions
    data_types = [
        ('BASELINE', lambda df: df), # Baseline is just the cohort itself
        ('DIAG', preproc_diagnoses),
        ('MEDS', preproc_meds),
        ('PROC', preproc_proc),
        ('CHART', preproc_chart),
        ('LABS', preproc_labs)
    ]
    if notes:
        data_types.extend([
            ('DSNOTES', preproc_dsnotes),
            ('RADNOTES', preproc_radnotes)
        ])

    if debug is not None:
        data_types = [dt for dt in data_types if dt[0] == debug]

    for df_type, func in data_types:
        try:
            # Note: phenotype is None for shared paths, noisy_label is typically False for golden test/val
            # 'long=True' is assumed for most processed data before pivoting.
            # 'validation=False' as it's a combined test/val set, not a specific validation subset path.
            path = generate_file_path(phenotype=None, df_type=df_type, train=False,
                                      noisy_label=noisy_label, # Should be False for shared golden data
                                      long=True, validation=False, use_shared_paths=True)
            print(f"[PROCESSING SHARED {df_type} and path {path}]")

            if path_exists(path) and debug is None: # Only skip if not in debug mode for a specific type
                print(f"Skipping SHARED file already processed: {path}")
                continue

            if df_type == 'DIAG':
                df_processed = func(cohort=master_test_val_cohort, icd_map_path=ICD_MAP_PATH,
                                    root_icd9=root_icd9, group_icd10=group_icd10, options=diag_options)
            elif df_type in ['RADNOTES', 'DSNOTES']:
                # NLP data paths might need their own shared generation logic if processed separately
                nlp_path = generate_nlp_file_path(phenotype=None, df_type=df_type, train=False,
                                                  noisy_label=noisy_label, # Should be False
                                                  use_shared_paths=True)
                print(f"[PROCESSING SHARED NLP {df_type} and path {nlp_path}]")
                if path_exists(nlp_path) and debug is None:
                    print(f"Skipping SHARED NLP file already processed: {nlp_path}")
                    continue
                df_processed = func(cohort=master_test_val_cohort)
                save_csv(df_processed, nlp_path) # Save NLP data to its specific shared NLP path
                print(f"SHARED NLP File saved successfully: {nlp_path}")
                # For the main 'long' file path, it might be a subset of columns or a direct copy
                # depending on whether NLP features are added back to a main table or kept separate.
                # Assuming for now that the main 'long' path for RADNOTES/DSNOTES might not be needed
                # if NLP features are primary output and handled by generate_nlp_file_path.
                # If generate_file_path for RADNOTES/DSNOTES (non-NLP) is still desired,
                # ensure func returns the appropriate DataFrame.
                if df_type in ['RADNOTES', 'DSNOTES'] and path != nlp_path: # Avoid double saving if paths are same
                     # If generate_file_path is different from generate_nlp_file_path for notes
                     # and you want to save a non-NLP version or a link:
                     # save_csv(df_processed_non_nlp_version, path)
                     print(f"Skipping save for {path} as NLP data handled by {nlp_path}")
                continue # Continue to next df_type after handling NLP notes

            elif df_type == 'MEDS':
                df_processed = func(cohort=master_test_val_cohort, time=time, group=med_grouping)
            elif df_type != 'BASELINE': # Ensure 'BASELINE' isn't processed by a time-dependent func
                df_processed = func(cohort=master_test_val_cohort, time=time)
            else: # BASELINE
                df_processed = master_test_val_cohort


            save_csv(df_processed, path)
            print(f"SHARED File saved successfully: {path}")
        except Exception as e:
            print(f"Error processing SHARED {df_type}: {e}")


def process_and_save_phenotype_data(df_base_core, phenotype, train=True, noisy_label=True, time=True, notes=True, root_icd9=True, group_icd10=True,
                                    diag_options='ICD10', med_grouping=True, pos_split=0.5, debug=None):
    """Process and save all phenotype data.
       Pos plit determines the ratio of pos/negative controls for each phenotype
    """
    try:
        if train: # Training data processing remains phenotype-specific
            core_df = phenotype_cohort_train(df_base_core, phenotype, silver_labels=noisy_label)
            datasets = [('train', core_df)]
            print(f'Training dataset created successfully for {phenotype} using silver labels: {noisy_label}')

            # Define the data types and corresponding processing functions
            data_types = [
                ('BASELINE', lambda df: df),
                ('DIAG', preproc_diagnoses),
                ('MEDS', preproc_meds),
                ('PROC', preproc_proc),
                ('CHART', preproc_chart),
                ('LABS', preproc_labs)
            ]
            if notes:
                data_types.extend([
                    ('DSNOTES', preproc_dsnotes),
                    ('RADNOTES', preproc_radnotes)
                ])

            if debug is not None:
                data_types = [dt for dt in data_types if dt[0] == debug]

            for dataset_name, dataset_df in datasets: # Should only be 'train'
                for df_type, func in data_types:
                    try:
                        # For training, use_shared_paths is False, validation is False
                        path = generate_file_path(phenotype, df_type, train=True,
                                                  noisy_label=noisy_label, long=True,
                                                  validation=False, use_shared_paths=False)
                        print(f"[PROCESSING TRAINING {df_type} FOR {phenotype} and path {path}]")
                        if path_exists(path) and debug is None:
                            print(f"Skipping TRAINING file already processed: {path}")
                            continue
                        if df_type == 'DIAG':
                            df_processed = func(cohort=dataset_df, icd_map_path=ICD_MAP_PATH, root_icd9=root_icd9, group_icd10=group_icd10, options=diag_options)
                        elif df_type in ['RADNOTES', 'DSNOTES']:
                             # Training NLP data path generation
                            nlp_path = generate_nlp_file_path(phenotype, df_type, train=True, noisy_label=noisy_label, use_shared_paths=False)
                            print(f"[PROCESSING TRAINING NLP {df_type} FOR {phenotype} and path {nlp_path}]")
                            if path_exists(nlp_path) and debug is None:
                                print(f"Skipping TRAINING NLP file already processed: {nlp_path}")
                                continue
                            df_processed = func(cohort=dataset_df)
                            save_csv(df_processed, nlp_path)
                            print(f"TRAINING NLP File saved successfully: {nlp_path}")
                            if path != nlp_path: # if main long path for notes is different
                                print(f"Skipping save for {path} as NLP data handled by {nlp_path} for training")
                            continue # Continue to next df_type

                        elif df_type == 'MEDS':
                            df_processed = func(cohort=dataset_df, time=time, group=med_grouping)
                        elif df_type != 'BASELINE':
                            df_processed = func(cohort=dataset_df, time=time)
                        else:
                            df_processed = dataset_df

                        save_csv(df_processed, path)
                        print(f"TRAINING File saved successfully: {path}")
                    except Exception as e:
                        print(f"Error processing TRAINING {df_type} for {phenotype}: {e}")
        # For non-train (test/validation), this function will now only create index files.
        # The actual data processing is handled by process_and_save_shared_test_val_data
        # and orchestrated by generate_raw_csv.
        else: # train == False logic for creating index files
            val_df, test_df = phenotype_cohort_test(df_base_core, phenotype, val_frac=pos_split)
            datasets_for_index = [('validation', val_df), ('test', test_df)]
            print(f'Index Validation and Testing datasets created for {phenotype}')

            dfs_index = []
            for set_name, df_index_part in datasets_for_index:
                temp_df = df_index_part.copy()
                temp_df['set'] = set_name
                temp_df['phenotype_set'] = phenotype # Record which phenotype this index entry belongs to
                # Ensure 'y' (golden label) is present
                if 'y' not in temp_df.columns:
                     # This might happen if phenotype_cohort_test doesn't return 'y'
                     # or if df_base_core was missing it for these hadm_ids.
                     # Add a placeholder or handle as an error.
                     print(f"WARNING: 'y' column missing in {set_name} set for {phenotype} during index creation. Golden labels might be incomplete.")
                     temp_df['y'] = -1 # Placeholder for missing golden label
                dfs_index.append(temp_df[['hadm_id', 'phenotype_set', 'y', 'set']])

            combined_index_df = pd.concat(dfs_index, axis=0, ignore_index=True)
            # Ensure INDEXES_PATH is defined in constants.py and directory exists
            index_file_path = f"{INDEXES_PATH}{phenotype}_indexes.csv"
            os.makedirs(os.path.dirname(index_file_path), exist_ok=True) # Ensure directory exists
            save_csv(combined_index_df, index_file_path)
            print(f'Index file for {phenotype} saved successfully to {index_file_path}')
            # Return the val_df and test_df so generate_raw_csv can collect all hadm_ids
            return val_df, test_df

    except Exception as e:
        print(f"Error in process_and_save_phenotype_data for {phenotype} (train={train}): {e}")
        if not train:
             return pd.DataFrame(), pd.DataFrame() # Return empty DFs on error for non-train
         # For train, error is caught and printed, function effectively ends.


def generate_raw_csv(phenotypes_to_process, train=True, noisy_label=True, time=True, notes=True, root_icd9=True, group_icd10=True,
                                        diag_options='ICD10', med_grouping = True, debug=None):
    """
    Generate raw csv files.
    If train=True, processes each phenotype in phenotypes_to_process individually.
    If train=False, creates index files for each phenotype, then processes a single shared dataset
    for all test/validation admissions.

    :param phenotypes_to_process: A single phenotype string (if train=True) or a list of phenotype strings (if train=False).
    """
    data = load_data()
    # Create the initial base core (patients, admissions, icustays, basic cleaning, age filter)
    # This version is used for generating the indexes to ensure golden_labels are the primary driver.
    df_base_core_for_indexes = generate_baseline_cohort(data['patients'], data['admissions'])
    df_base_core_for_indexes = generate_icu_admission_status(df_base_core_for_indexes, data['icustays'])
    df_base_core_for_indexes = cleanup_merged_dataframe(df_base_core_for_indexes)
    print(f"--- generate_raw_csv: df_base_core_for_indexes created with {df_base_core_for_indexes['hadm_id'].nunique()} unique hadm_ids (for index generation).")

    # Now, create the more processed df_base_core that includes diagnosis codes and will undergo further processing (like a more targeted dropna)
    # This starts from the same point as df_base_core_for_indexes
    df_base_core_processed = df_base_core_for_indexes.copy() # Start from the same base
    
    print(f"--- merge_and_process_data logic (within generate_raw_csv): Unique hadm_ids BEFORE generate_addition_icdcodes & dropna: {df_base_core_processed['hadm_id'].nunique()}")
    df_temp_before_dropna = generate_addition_icdcodes(df_base_core_processed, data['diagnoses_icd'])
    print(f"--- merge_and_process_data logic (within generate_raw_csv): Unique hadm_ids AFTER generate_addition_icdcodes (BEFORE dropna): {df_temp_before_dropna['hadm_id'].nunique()}")

    newly_added_cols_by_icd_merge = ['icd_code', 'icd_version', 'admissions_number']
    if all(col in df_temp_before_dropna.columns for col in newly_added_cols_by_icd_merge):
        rows_with_nan_in_new_cols = df_temp_before_dropna[df_temp_before_dropna[newly_added_cols_by_icd_merge].isnull().any(axis=1)]
        if not rows_with_nan_in_new_cols.empty:
            print(f"--- merge_and_process_data logic (within generate_raw_csv): Number of rows with any NaN in {newly_added_cols_by_icd_merge} (candidates for dropna): {len(rows_with_nan_in_new_cols)}")
            print(f"--- merge_and_process_data logic (within generate_raw_csv): Unique hadm_ids in those NaN rows: {rows_with_nan_in_new_cols['hadm_id'].nunique()}")
        else:
            print(f"--- merge_and_process_data logic (within generate_raw_csv): No rows found with NaNs in {newly_added_cols_by_icd_merge} that would be dropped by .dropna() on these columns.")
    else:
        print(f"--- merge_and_process_data logic (within generate_raw_csv): One or more expected columns ({newly_added_cols_by_icd_merge}) not found after generate_addition_icdcodes.")

    # TODO: Re-evaluate the dropna strategy. For now, it's bypassed as per previous step.
    # df_base_core_processed = df_temp_before_dropna.dropna() 
    df_base_core_processed = df_temp_before_dropna # Keep all rows for now (consistent with previous temporary change)
    print(f"--- merge_and_process_data logic (within generate_raw_csv): Unique hadm_ids AFTER generate_addition_icdcodes & SKIPPING dropna: {df_base_core_processed['hadm_id'].nunique()}")
    print("Successfully created Processed Base Core Dataframe (df_base_core_processed)")


    if train:
        if not isinstance(phenotypes_to_process, str):
            raise ValueError("For training, phenotypes_to_process must be a single phenotype string.")
        phenotype = phenotypes_to_process
        print(f"--- Generating TRAINING data for phenotype: {phenotype} ---")
        # Training data likely uses the fully processed df_base_core_processed
        process_and_save_phenotype_data(df_base_core_processed, phenotype, train=True, noisy_label=noisy_label,
                                        time=time, notes=notes, root_icd9=root_icd9, group_icd10=group_icd10,
                                        diag_options=diag_options, med_grouping=med_grouping, debug=debug)
    else: # train == False : Processing for Test/Validation (Index creation + Shared Data)
        if not isinstance(phenotypes_to_process, list):
            raise ValueError("For test/validation (train=False), phenotypes_to_process must be a list of phenotype strings.")

        all_test_val_hadm_ids = set()
        print("--- Generating Test/Validation INDEXES for all specified phenotypes (using df_base_core_for_indexes) ---")
        for phenotype in phenotypes_to_process:
            print(f"-- Creating index for phenotype: {phenotype} --")
            # This call now primarily creates and saves the index file.
            # It uses df_base_core_for_indexes
            val_df, test_df = process_and_save_phenotype_data(
                df_base_core_for_indexes, phenotype, train=False, noisy_label=False, # noisy_label=False for golden test/val
                time=time, notes=notes, root_icd9=root_icd9, group_icd10=group_icd10,
                diag_options=diag_options, med_grouping=med_grouping, debug=debug 
                # pos_split is still passed from main_noarg, ensure process_and_save_phenotype_data handles it or remove if not needed for train=False
            )
            if val_df is not None and not val_df.empty:
                all_test_val_hadm_ids.update(val_df['hadm_id'].unique())
            if test_df is not None and not test_df.empty:
                all_test_val_hadm_ids.update(test_df['hadm_id'].unique())

        if not all_test_val_hadm_ids:
            print("WARNING: No hadm_ids collected for test/validation sets. Skipping shared data generation.")
            return

        print(f"Total unique HADM_IDs for shared test/validation processing: {len(all_test_val_hadm_ids)}")
        # Create the master cohort for all test/validation admissions using the *processed* base core
        # Ensure it only contains hadm_ids that are in our collected all_test_val_hadm_ids
        master_test_val_cohort = df_base_core_processed[df_base_core_processed['hadm_id'].isin(list(all_test_val_hadm_ids))]

        if master_test_val_cohort.empty:
            print("WARNING: Master test/validation cohort is empty after filtering df_base_core_processed. Skipping shared data generation.")
            return

        print("--- Generating SHARED data for all Test/Validation admissions (using master_test_val_cohort from df_base_core_processed) ---")
        process_and_save_shared_test_val_data(
            master_test_val_cohort, noisy_label=False, time=time, notes=notes,
            root_icd9=root_icd9, group_icd10=group_icd10, diag_options=diag_options,
            med_grouping=med_grouping, debug=debug
        )
