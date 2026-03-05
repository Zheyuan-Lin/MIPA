########## Importing libraries ##########

import pandas as pd
from tqdm import tqdm
from scripts.constants import *
from scripts.subtask.outlier_removal import *
from scripts.featureselection.medcat_parser import *
from scripts.utils import *

# os.chdir('/Users/eyamga/Documents/Medecine/Recherche/Maitrise MSc/Thèse/MIPA-Git/MIMIC-IV-Phenotype-Atlas')

######## General function : cleaning and parsing all features ########


def apply_feature_selection(df, feature, lower_quantile=0.05):
    """
    Apply feature selection to a DataFrame based on a feature's unique value count, using a dynamic threshold.

    The threshold is set to the 5th percentile of the counts of unique values for the specified feature.
    This ensures that features with counts below this threshold are removed.

    Parameters:
    - df: The DataFrame to apply feature selection to.
    - feature: The name of the feature column to apply selection on.

    Returns:
    - The modified DataFrame with features selected based on the dynamic threshold.
    """
    # TODO: consider normalizing the counts based on count/LOS
    # Count the number of occurrences for each feature value
    count = df[feature].value_counts()

    # Calculate the 5th percentile of the counts
    threshold = count.quantile(lower_quantile)

    # Identify the indices of features with a count above the threshold
    valid_indices = count[count > threshold].index

    # Filter the DataFrame to only include rows where the feature is in valid_indices
    df_filtered = df[df[feature].isin(valid_indices)]
    print(f"Previous number of unique {feature} values: {df[feature].nunique()}")
    # Optionally, print the new number of unique feature values for comparison
    print(f"New number of unique {feature} values: {df_filtered[feature].nunique()}")

    return df_filtered

def apply_summary(df, df_type, NLP = False):
    """
    Generate a summary for the given DataFrame based on the data type.

    Parameters:
    - df: The DataFrame to generate a summary for.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).

    Returns:
    - A DataFrame containing the summary.
    """

    def summarize_cui_codes(df, cui_column='nlp_codes'):
        """
        Summarize the CUI codes from a DataFrame containing CUI counts in a string format and calculate text length.

        Parameters:
        - df: DataFrame containing the CUI data.
        - cui_column: The name of the column containing the CUI counts.

        Returns:
        - A DataFrame summarizing the total occurrences of each CUI and the text length.
        """
        # Expand the CUI counts into a list of tuples (CUI, count, text length)
        all_cuis = []
        for item in df[cui_column].dropna():
            text_length = len(item)  # Calculate the length of the text
            cuis = item.split(', ')
            for cui in cuis:
                cui_code, count = cui.split(': ')
                all_cuis.append((cui_code, int(count), text_length))

        # Convert list to DataFrame
        cui_df = pd.DataFrame(all_cuis, columns=['CUI', 'Count', 'Text Length'])

        # Aggregate counts by CUI and calculate average text length
        summary_df = cui_df.groupby('CUI').agg({'Count': 'sum', 'Text Length': 'mean'}).reset_index()

        return summary_df

    if df_type == 'DIAG':
        freq = df.groupby(['hadm_id', 'icd_code']).size().reset_index(name="mean_frequency")
        freq = freq.groupby(['icd_code'])['mean_frequency'].mean().reset_index()
        total = df.groupby('icd_code').size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on='icd_code', how='right')
        summary = summary.fillna(0)
    elif df_type == 'MEDS':
        freq = df.groupby(['hadm_id', 'ndc']).size().reset_index(name="mean_frequency")
        freq = freq.groupby(['ndc'])['mean_frequency'].mean().reset_index()
        missing = df[df['ndc'].isna()].groupby('ndc').size().reset_index(name="missing_count")
        total = df.groupby('ndc').size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on='ndc', how='right')
        summary = pd.merge(freq, summary, on='ndc', how='right')
        summary = summary.fillna(0)
    elif df_type == 'PROC':
        freq = df.groupby(['hadm_id', 'icd_code']).size().reset_index(name="mean_frequency")
        freq = freq.groupby(['icd_code'])['mean_frequency'].mean().reset_index()
        total = df.groupby('icd_code').size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on='icd_code', how='right')
        summary = summary.fillna(0)
    elif df_type in ['LABS', 'CHART']:
        freq = df.groupby(['hadm_id', 'itemid']).size().reset_index(name="mean_frequency")
        freq = freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total = df.groupby('itemid').size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on='itemid', how='right')
        summary = summary.fillna(0)

    elif df_type in ['DSNOTES', 'RADNOTES']:
        if NLP:
            #TODO: fix
            # Return summary of CUI codes count + text length
            summary = summarize_cui_codes(df)
        else:
            # Return summary of text length only
            summary = df['text'].apply(len).describe()

    return summary

def generate_features_selection(phenotype, train = True, noisy_label = True, notes = False, NLP = False, debug=None,
                                lower_labs=0.25, lower_chart=0.25, lower_icd=0.25, lower_meds=0.25, lower_nlp=0.10,
                                use_shared_paths=False):
    """
    Apply feature selection and outlier imputation to various data types.

    Parameters:
    - phenotype: The phenotype to process. Used for phenotype-specific paths unless use_shared_paths is True.
    - train: A boolean indicating whether the data is for training.
    - noisy_label: A boolean indicating whether the data includes silver labels.
    - notes: A boolean indicating whether to include notes data types.
    - NLP: A boolean indicating whether to apply NLP feature generation on notes.
    - debug: If specified, only process this data type.
    - lower_*: Quantile thresholds for feature selection.
    - use_shared_paths: If True and train is False, use shared paths for test/validation data.
    """

    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes:
        df_types.extend(['RADNOTES', 'DSNOTES'])
    if debug is not None:
        df_types = [dt for dt in df_types if dt == debug] # Corrected debug type check

    current_phenotype_for_paths = None if (not train and use_shared_paths) else phenotype
    current_noisy_label_for_paths = False if (not train and use_shared_paths) else noisy_label

    print("[LOADING FILES FOR FEATURE SELECTION]")
    for df_type in df_types:
        if not train and use_shared_paths:
            # For shared test/val data, NLP files are the primary source for RADNOTES/DSNOTES if NLP=True
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                path = generate_nlp_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, use_shared_paths)
            else:
                path = generate_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, long=True, use_shared_paths=use_shared_paths)
        else: # Training paths or non-shared test/val paths (if that logic was ever to be used)
            # For training, NLP files are also the primary source for RADNOTES/DSNOTES if NLP=True for feature selection input
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                 path = generate_nlp_file_path(phenotype, df_type, train, noisy_label, use_shared_paths=False)
            else:
                 path = generate_file_path(phenotype, df_type, train, noisy_label, long=True, use_shared_paths=False)

        if not path_exists(path):
            print(f"[SKIPPING FEATURE SELECTION FOR {df_type.upper()}] File does not exist: {path}")
            continue

        print(f"[PROCESSING FEATURE SELECTION FOR {df_type.upper()} DATA FROM {path}]")

        try:
            df = read_gzip_or_parts(path)
            if df.empty:
                print(f"[SKIPPING FEATURE SELECTION FOR {df_type.upper()}] Loaded dataframe is empty from {path}")
                continue

            if df_type in ['DIAG', 'PROC']:
                df = apply_feature_selection(df, 'icd_code', lower_quantile=lower_icd)
                save_gzip(df, path) # Overwrite existing file with selected features
            elif df_type == 'MEDS':
                df = apply_feature_selection(df, 'ndc', lower_quantile=lower_meds)
                save_gzip(df, path)
            elif df_type == 'CHART':
                df = apply_feature_selection(df, 'itemid', lower_quantile=lower_chart)
                save_gzip(df, path)
            elif df_type == 'LABS':
                df = apply_feature_selection(df, 'itemid', lower_quantile=lower_labs)
                df = outlier_imputation(df, 'itemid', 'valuenum', 0.9, 0.1, impute=False)
                save_gzip(df, path)
            elif df_type in ['RADNOTES', 'DSNOTES']:
                if NLP:
                    # This section assumes that the NLP processing (MedCAT annotation)
                    # has already been done in generation.py and the input `path` for NLP notes
                    # points to the file with 'text' or pre-extracted CUIs if that was the design.
                    # If `path` for NLP notes (from generate_nlp_file_path) already contains 'nlp_codes',
                    # then process_and_annotate_dataframe might not be needed here or needs to be conditional.

                    # For feature selection, we re-read the NLP processed file if not already `df`
                    # This df should have 'nlp_codes' column if MedCAT processing happened in generation.py
                    # If MedCAT processing (text -> CUIs) happens HERE, then `df` should have `text` column.
                    # Assuming `process_and_annotate_dataframe` is the MedCAT step to get CUIs.
                    # And `apply_feature_selection_and_export_cuis` selects from those CUIs.

                    print(f"[NLP FEATURE SELECTION STARTED FOR {df_type.upper()}]")
                    # If `df` from `generate_nlp_file_path` does not have the raw text column needed for MedCAT,
                    # this part needs to be re-evaluated. Assume df from generate_nlp_file_path has 'text' for now if MedCAT is here.
                    # OR if `generation.py` already produced 'nlp_codes', this step is simpler.

                    # Let's assume the input `df` (from `generate_nlp_file_path`) ALREADY has nlp_codes column
                    # if it was created and saved by process_and_save_shared_test_val_data or its train equivalent.
                    # If this step is meant to *generate* nlp_codes from text, then input path strategy needs care.

                    # For now, assuming this part is about selecting from existing `nlp_codes`
                    # and exporting a dictionary. If `df` contains raw text, then MedCAT steps would be here.
                    # This function's primary role here for NLP is CUI selection & dict export.

                    cat = generate_cat() # Initialize MedCat if CUI generation is done here.
                                        # If only selection from pre-existing CUIs, cat might not be needed for selection itself.

                    # Path for the dictionary to be saved
                    dict_path_nlp = generate_dict_path(current_phenotype_for_paths, df_type=df_type,
                                                     train=train, label=current_noisy_label_for_paths,
                                                     use_shared_paths=use_shared_paths)

                    # Feature selection based on CUI frequency + Dictionary Export
                    # This function needs `df` to have the column that `apply_feature_selection_and_export_cuis` expects (e.g. 'nlp_codes' or similar)
                    df_selected_cuis = apply_feature_selection_and_export_cuis(df, cat=cat, output_file=dict_path_nlp,
                                                                               lower_quantile=lower_nlp)
                    print(f"[NLP DICTIONARY SAVED FOR {df_type.upper()}] to {dict_path_nlp}")

                    # The output `df_selected_cuis` should contain 'subject_id', 'hadm_id', and the selected 'nlp_codes'.
                    # This DataFrame (with selected CUIs) is what should be saved to be used by the pivot step.
                    # The path it's saved to should be the same `generate_nlp_file_path` used for input if overwriting,
                    # or a new path if keeping intermediate stages.
                    # Let's assume we overwrite the file at `path` (which was `generate_nlp_file_path` for NLP notes)
                    # with the version that has feature-selected `nlp_codes`.
                    if 'nlp_codes' not in df_selected_cuis.columns and 'nlp_codes' in df.columns:
                         # If selection func modified df in place and didn't return it with only selected CUIs
                         # or if it returned a new df that is now df_selected_cuis.
                         # Ensure we save the correct dataframe that has undergone CUI selection.
                         save_gzip(df_selected_cuis[["subject_id", "hadm_id", "nlp_codes"]], path)
                    elif 'nlp_codes' in df_selected_cuis.columns : # df_selected_cuis is the one to save
                         save_gzip(df_selected_cuis[["subject_id", "hadm_id", "nlp_codes"]], path)
                    else:
                        print(f"WARNING: 'nlp_codes' column not found after feature selection for {df_type}. Original data from {path} might be saved if overwrite intended.")
                        # save_gzip(df, path) # Or handle error

                    print(f"[NLP FEATURE SELECTION COMPLETED AND SAVED FOR {df_type.upper()}] to {path}")
                else: # Not NLP for notes, just skip feature selection for now
                    print(f"[SKIPPING FEATURE SELECTION FOR {df_type.upper()}] NLP is False.")
                    continue

            print(f"[SUCCESSFULLY PROCESSED AND SAVED {df_type.upper()} DATA AFTER FEATURE SELECTION]")
        except Exception as e:
            print(f"[ERROR PROCESSING FEATURE SELECTION FOR {df_type.upper()} DATA] {e}")
            continue

    print("[FEATURE SELECTION COMPLETE]")

def generate_summary(phenotype, train = True, noisy_label = True, notes = False, NLP = False, debug=None, use_shared_paths=False):
    """
    Generate feature summary for various data types.

    Parameters:
    - phenotype: The phenotype to process. Used for phenotype-specific paths unless use_shared_paths is True.
    - train: A boolean indicating whether the data is for training.
    - noisy_label: A boolean indicating whether the data includes silver labels.
    - notes: A boolean indicating whether to include notes data types.
    - NLP: A boolean indicating whether to process notes with NLP techniques.
    - debug: If specified, only process this data type.
    - use_shared_paths: If True and train is False, use shared paths for test/validation data.
    """
    print("[GENERATING FEATURE SUMMARY]")

    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes:
        df_types.extend(['RADNOTES', 'DSNOTES'])
    if debug is not None:
        df_types = [dt for dt in df_types if dt == debug] # Corrected debug type check

    current_phenotype_for_paths = None if (not train and use_shared_paths) else phenotype
    current_noisy_label_for_paths = False if (not train and use_shared_paths) else noisy_label
    # For summaries, the 'label' parameter in generate_summary_path corresponds to noisy_label
    current_label_for_summary_path = current_noisy_label_for_paths

    for df_type in df_types:
        # Determine input path
        if not train and use_shared_paths:
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                # For shared NLP summaries, read the output of NLP feature selection
                input_path = generate_nlp_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, use_shared_paths)
            else:
                input_path = generate_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, long=True, use_shared_paths=use_shared_paths)
        else: # Training paths or non-shared test/val paths
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                input_path = generate_nlp_file_path(phenotype, df_type, train, noisy_label, use_shared_paths=False)
            else:
                input_path = generate_file_path(phenotype, df_type, train, noisy_label, long=True, use_shared_paths=False)

        if not path_exists(input_path):
            print(f"[SKIPPING SUMMARY FOR {df_type.upper()}] File does not exist: {input_path}")
            continue

        try:
            df = read_gzip_or_parts(input_path)
            if df.empty:
                print(f"[SKIPPING SUMMARY FOR {df_type.upper()}] Loaded dataframe is empty from {input_path}")
                continue

            # Generate summary based on df_type
            # The apply_summary function needs to handle cases where df might be from NLP (e.g. has 'nlp_codes')
            summary_df = apply_summary(df, df_type, NLP=NLP) # Pass NLP flag to apply_summary

            # Determine output path for summary
            summary_output_path = generate_summary_path(current_phenotype_for_paths, train, current_label_for_summary_path, df_type, use_shared_paths)

            save_csv(summary_df, summary_output_path)
            print(f"[SUCCESSFULLY SAVED {df_type.upper()} SUMMARY] {summary_output_path}")
        except Exception as e:
            print(f"[ERROR PROCESSING SUMMARY FOR {df_type.upper()} DATA] {e}")
            continue

    print("[FEATURE SUMMARY GENERATION COMPLETE]")

