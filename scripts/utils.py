###########################################################################################################
#                      Utils.py Basic dataframe processing and wrangling functions
###########################################################################################################
#

import os
import sys
import gc
import gcsfs
import logging
import gzip

# Base
import math
import copy
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import json
import regex
import re

from dask import dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()
import shutil
from tqdm.auto import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shutil
from google.oauth2 import service_account

from scripts.constants import *
from scripts.constants_etp import *
from difflib import get_close_matches


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                Initializations & Data Loading                                   |
#                                                                                                 |
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
"""

# Initiate Google Cloud Storage file system
def is_gcs_path(path):
    return path.startswith('gs://')


def setup_gcs_access():
    # Path to your service account key file
    key_path = SECRET
    credentials = service_account.Credentials.from_service_account_file(key_path, scopes = SCOPES)
    fs = gcsfs.GCSFileSystem(token=credentials)
    return fs

def path_exists(path):
    if is_gcs_path(path):
        fs = setup_gcs_access()
        return fs.exists(path)
    else:
        return os.path.exists(path)

def ensure_pandas_df(df):
    """Ensure the DataFrame is a pandas DataFrame, computing if it's a Dask DataFrame."""
    if isinstance(df, dd.DataFrame):
        return df.compute()  # Compute if Dask DataFrame to convert to pandas
    elif isinstance(df, pd.DataFrame):
        return df  # Return as is if already a pandas DataFrame
    else:
        raise TypeError("The input must be a Dask DataFrame or a pandas DataFrame.")

def check_credentials(path_to_credentials):
    try:
        credentials = service_account.Credentials.from_service_account_file(path_to_credentials,
                                                                            scopes=[
                                                                                    'https://www.googleapis.com/auth/devstorage.read_write']
                                                                            )
        print("Credentials are loaded successfully.")
        return credentials
    except Exception as e:
        print(f"Failed to load credentials: {e}")


def test_gcs_access(credentials, bucket_name):
    try:
        fs = gcsfs.GCSFileSystem(token=credentials)
        # List the first few files in the bucket
        files = fs.ls(bucket_name)
        print(f"Successfully accessed files in bucket '{bucket_name}': {files[:5]}")  # Show first 5 files
    except Exception as e:
        print(f"Failed to access GCS bucket: {e}")

def generate_file_path(phenotype, df_type, train=True, noisy_label=True, long=False, validation=False, use_shared_paths=False):
    """
    Generate a file path for saving or checking the existence of a file based on the given parameters.
    Function used for preprocessing step
    Parameters:
    - phenotype (str): The name of the phenotype. Relevant unless use_shared_paths is True for test/val.
    - df_type (str): The table (e.g., 'BASELINE', 'DIAG', 'MEDS', etc.).
    - train (bool): Whether the data is for training or testing/validation. Default is True.
    - noisy_label (bool): Whether the data includes silver labels. Default is True.
    - long (bool): Whether to use the 'long' subdirectory. Default is False.
    - validation (bool): Whether the data is for validation. Default is False. Only used if train is False.
    - use_shared_paths (bool): If True and train is False, generate paths for shared test/validation data. Default is False.
    Returns:
    - str: The constructed file path.
    """
    if not train and use_shared_paths:
        # Paths for shared test/validation data
        base_path = COHORT_SHARED_TEST_VAL_LONG_PATH if long else COHORT_SHARED_TEST_VAL_ROOT_PATH
        # For shared paths, phenotype is not part of the directory structure,
        # and noisy_label/validation flags might be standardized (e.g., always False for golden test/val data)
        # Filename uses a generic suffix
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}.csv.gz"
    elif train:
        base_path = COHORT_TRAIN_PATH
        label_suffix = "_silver" if noisy_label else ""
        train_suffix = "_train" # Always _train for this branch
        if long:
            file_path = f"{base_path}{phenotype}/long/{df_type}{train_suffix}{label_suffix}.csv.gz"
        else:
            file_path = f"{base_path}{phenotype}/{df_type}{train_suffix}{label_suffix}.csv.gz"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_VAL_PATH if validation else COHORT_TEST_PATH
        label_suffix = "_silver" if noisy_label else "" # Typically False for val/test golden
        # Suffix depends on whether it's validation or test
        dataset_suffix = "_val" if validation else "_test"
        if long:
            file_path = f"{base_path}{phenotype}/long/{df_type}{dataset_suffix}{label_suffix}.csv.gz"
        else:
            file_path = f"{base_path}{phenotype}/{df_type}{dataset_suffix}{label_suffix}.csv.gz"
    return file_path

def generate_file_path_local(phenotype, df_type, train=True, noisy_label=True, long=False, validation=False, use_shared_paths=False):
    """
    Generate a file path for saving or checking the existence of a file based on the given parameters.
    Function used for preprocessing step
    Parameters:
    - phenotype (str): The name of the phenotype. Relevant unless use_shared_paths is True for test/val.
    - df_type (str): The table (e.g., 'BASELINE', 'DIAG', 'MEDS', etc.).
    - train (bool): Whether the data is for training or testing/validation. Default is True.
    - noisy_label (bool): Whether the data includes silver labels. Default is True.
    - long (bool): Whether to use the 'long' subdirectory. Default is False.
    - validation (bool): Whether the data is for validation. Default is False. Only used if train is False.
    - use_shared_paths (bool): If True and train is False, generate paths for shared test/validation data. Default is False.
    Returns:
    - str: The constructed file path.
    """
    if not train and use_shared_paths:
        # Paths for shared test/validation data
        base_path = LOCAL_COHORT_SHARED_TEST_VAL_LONG_PATH if long else LOCAL_COHORT_SHARED_TEST_VAL_ROOT_PATH
        # For shared paths, phenotype is not part of the directory structure,
        # and noisy_label/validation flags might be standardized (e.g., always False for golden test/val data)
        # Filename uses a generic suffix
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}.csv.gz"
    elif train:
        base_path = LOCAL_COHORT_TRAIN_PATH
        label_suffix = "_silver" if noisy_label else ""
        train_suffix = "_train" # Always _train for this branch
        if long:
            file_path = f"{base_path}{phenotype}/long/{df_type}{train_suffix}{label_suffix}.csv.gz"
        else:
            file_path = f"{base_path}{phenotype}/{df_type}{train_suffix}{label_suffix}.csv.gz"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_VAL_PATH if validation else COHORT_TEST_PATH
        label_suffix = "_silver" if noisy_label else "" # Typically False for val/test golden
        # Suffix depends on whether it's validation or test
        dataset_suffix = "_val" if validation else "_test"
        if long:
            file_path = f"{base_path}{phenotype}/long/{df_type}{dataset_suffix}{label_suffix}.csv.gz"
        else:
            file_path = f"{base_path}{phenotype}/{df_type}{dataset_suffix}{label_suffix}.csv.gz"
    return file_path

def generate_nlp_file_path(phenotype, df_type, train=True, noisy_label=True, use_shared_paths=False):
    """
    Generate a file path for saving or checking the existence of NLP files.
    Parameters:
    - phenotype (str): The name of the phenotype.
    - df_type (str): The table (e.g., 'RADNOTES', 'DSNOTES').
    - train (bool): Whether the data is for training. Default is True.
    - noisy_label (bool): Whether the data includes silver labels. Default is True.
    - use_shared_paths (bool): If True and train is False, generate paths for shared NLP test/validation data. Default is False.
    Returns:
    - str: The constructed file path.
    """
    if not train and use_shared_paths:
        # Shared NLP path for test/validation
        # Assumes NLP data is stored under a 'long/NLP' like structure in the shared root
        base_path = COHORT_SHARED_TEST_VAL_NLP_PATH
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}.csv.gz"
    elif train:
        base_path = COHORT_TRAIN_PATH
        label_suffix = "_silver" if noisy_label else ""
        train_suffix = "_train"
        file_path = f"{base_path}{phenotype}/long/NLP/{df_type}{train_suffix}{label_suffix}.csv.gz"
    else: # Test or Validation specific NLP paths (not shared) - assuming these might still be needed for some reason or this branch becomes unused
        # This part of the logic might need review based on whether you'll ever generate non-shared NLP test/val files
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH depending on a 'validation' flag if needed
        label_suffix = "_silver" if noisy_label else "" # Typically False
        test_suffix = "_test" # Or _val
        file_path = f"{base_path}{phenotype}/long/NLP/{df_type}{test_suffix}{label_suffix}.csv.gz"
    return file_path

def generate_summary_path(phenotype, train, label, df_type, use_shared_paths=False):
    """
    Generate the path for the summary file.
    Parameters:
    - phenotype: The phenotype to process.
    - train: A boolean indicating whether the data is for training.
    - label: A boolean indicating if noisy label processing.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).
    - use_shared_paths (bool): If True and train is False, generate paths for shared summary test/validation data. Default is False.
    Returns:
    - The path for the summary file.
    """
    if not train and use_shared_paths:
        base_path = COHORT_SHARED_TEST_VAL_SUMMARY_PATH
        # For shared summaries, phenotype is not part of the directory structure.
        # Label suffix might be standardized (e.g., always non-silver for shared golden test/val)
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}_summary.csv"

    elif train:
        base_path = COHORT_TRAIN_PATH
        suffix = "_silver" if label else ""
        suffix += "_train"
        file_path = f"{base_path}{phenotype}/summary/{phenotype}_{df_type}{suffix}_summary.csv"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH based on a validation flag
        suffix = "_silver" if label else "" # Typically False
        # Suffix should distinguish test/val if not using shared. Assume _test for now.
        suffix += "_test"
        file_path = f"{base_path}{phenotype}/summary/{phenotype}_{df_type}{suffix}_summary.csv"
    return file_path

def generate_dict_path(phenotype, train, label, df_type, use_shared_paths=False):
    """
    Generate the path for the dictionary file.
    Parameters:
    - phenotype: The phenotype to process.
    - train: A boolean indicating whether the data is for training.
    - label: A boolean indicating if noisy label processing.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).
    - use_shared_paths (bool): If True and train is False, generate paths for shared dictionary test/validation data. Default is False.
    Returns:
    - The path for the dictionary file.
    """
    if not train and use_shared_paths:
        base_path = COHORT_SHARED_TEST_VAL_DICT_PATH
        # For shared dictionaries, phenotype is not part of the directory structure.
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}_dict.csv"
    elif train:
        base_path = COHORT_TRAIN_PATH
        suffix = "_silver" if label else ""
        suffix += "_train"
        file_path = f"{base_path}{phenotype}/dict/{phenotype}_{df_type}{suffix}_dict.csv"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH
        suffix = "_silver" if label else "" # Typically False
        suffix += "_test"
        file_path = f"{base_path}{phenotype}/dict/{phenotype}_{df_type}{suffix}_dict.csv"
    return file_path

def generate_dict_path_local(phenotype, train, label, df_type, use_shared_paths=False):
    """
    Generate the path for the dictionary file.
    Parameters:
    - phenotype: The phenotype to process.
    - train: A boolean indicating whether the data is for training.
    - label: A boolean indicating if noisy label processing.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).
    - use_shared_paths (bool): If True and train is False, generate paths for shared dictionary test/validation data. Default is False.
    Returns:
    - The path for the dictionary file.
    """
    if not train and use_shared_paths:
        base_path = LOCAL_COHORT_SHARED_TEST_VAL_DICT_PATH
        # For shared dictionaries, phenotype is not part of the directory structure.
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}_dict.csv"
    elif train:
        base_path = LOCAL_COHORT_TRAIN_PATH
        suffix = "_silver" if label else ""
        suffix += "_train"
        file_path = f"{base_path}{phenotype}/dict/{phenotype}_{df_type}{suffix}_dict.csv"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH
        suffix = "_silver" if label else "" # Typically False
        suffix += "_test"
        file_path = f"{base_path}{phenotype}/dict/{phenotype}_{df_type}{suffix}_dict.csv"
    return file_path

def generate_pivot_path(phenotype, df_type, train=True, noisy_label=True, use_shared_paths=False):
    """
    Generate the path for the pivot file.
    Parameters:
    - phenotype: The phenotype to process.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).
    - train: A boolean indicating whether the data is for training.
    - noisy_label: A boolean indicating if noisy label processing.
    - use_shared_paths (bool): If True and train is False, generate paths for shared pivot test/validation data. Default is False.
    Returns:
    - The path for the pivot file.
    """
    if not train and use_shared_paths:
        base_path = COHORT_SHARED_TEST_VAL_PIVOT_PATH
        # For shared pivot tables, phenotype is not part of the directory structure.
        # noisy_label might be standardized (e.g., always False for golden test/val data)
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}_pivot.csv.gz"
    elif train:
        base_path = COHORT_TRAIN_PATH
        label_suffix = "_silver" if noisy_label else ""
        train_suffix = "_train"
        file_path = f"{base_path}{phenotype}/pivot/{df_type}{train_suffix}{label_suffix}_pivot.csv.gz"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH
        label_suffix = "_silver" if noisy_label else "" # Typically False
        test_suffix = "_test"
        file_path = f"{base_path}{phenotype}/pivot/{df_type}{test_suffix}{label_suffix}_pivot.csv.gz"
    return file_path

def generate_pivot_path_local(phenotype, df_type, train=True, noisy_label=True, use_shared_paths=False):
    """
    Generate the path for the pivot file.
    Parameters:
    - phenotype: The phenotype to process.
    - df_type: The type of data (e.g., 'DIAG', 'MEDS', etc.).
    - train: A boolean indicating whether the data is for training.
    - noisy_label: A boolean indicating if noisy label processing.
    - use_shared_paths (bool): If True and train is False, generate paths for shared pivot test/validation data. Default is False.
    Returns:
    - The path for the pivot file.
    """
    if not train and use_shared_paths:
        base_path = LOCAL_COHORT_SHARED_TEST_VAL_PIVOT_PATH
        # For shared pivot tables, phenotype is not part of the directory structure.
        # noisy_label might be standardized (e.g., always False for golden test/val data)
        file_path = f"{base_path}{df_type}{SHARED_DATA_FILE_SUFFIX}_pivot.csv.gz"
    elif train:
        base_path = LOCAL_COHORT_TRAIN_PATH
        label_suffix = "_silver" if noisy_label else ""
        train_suffix = "_train"
        file_path = f"{base_path}{phenotype}/pivot/{df_type}{train_suffix}{label_suffix}_pivot.csv.gz"
    else: # Test or Validation specific paths (not shared)
        base_path = COHORT_TEST_PATH # Or COHORT_VAL_PATH
        label_suffix = "_silver" if noisy_label else "" # Typically False
        test_suffix = "_test"
        file_path = f"{base_path}{phenotype}/pivot/{df_type}{test_suffix}{label_suffix}_pivot.csv.gz"
    return file_path

def generate_model_path(phenotype):
    base_path = MODEL_PATH
    file_path = f"{base_path}{phenotype}/model_performance.csv"
    return file_path

def generate_overall_model_path():
    base_path = MODEL_PATH
    file_path = f"{base_path}overall_model_performance.csv"
    return file_path

def generate_parse_path(phenotype, config, detailed = False):
    base_path = PARSE_PATH
    if detailed:
        file_path = f"{base_path}detailed/{phenotype}/detailed_{config}.csv"
    else:
        file_path = f"{base_path}aggregated/{phenotype}/aggregated_{config}.csv"
    return file_path

def generate_parse_reasoning_path(phenotype, config, detailed = False):
    base_path = REASONING_PARSE_PATH
    if detailed:
        file_path = f"{base_path}detailed/{phenotype}/detailed_{config}.csv"
    else:
        file_path = f"{base_path}aggregated/{phenotype}/aggregated_{config}.csv"
    return file_path

def generate_metrics_path(phenotype, config):
    base_path = METRICS_PATH
    file_path = f"{base_path}phenotype_metrics/{phenotype}/{config}.csv"
    return file_path

def generate_metrics_reasoning_path(phenotype, config):
    base_path = REASONING_METRICS_PATH
    file_path = f"{base_path}phenotype_metrics/{phenotype}/{config}.csv"
    return file_path

def generate_errors_path(phenotype, config):
    base_path = ERRORS_PATH
    file_path = f"{base_path}{phenotype}/{config}.csv"
    return file_path

def generate_errors_reasoning_path(phenotype, config):
    base_path = REASONING_ERRORS_PATH
    file_path = f"{base_path}{phenotype}/{config}.csv"
    return file_path


def generate_errors_report_path(error_report_type):
    base_path = ERRORS_REPORT_PATH
    file_path = f"{base_path}{error_report_type}.csv"
    return file_path

def generate_errors_report_reasoning_path(error_report_type):
    base_path = REASONING_ERRORS_REPORT_PATH
    file_path = f"{base_path}{error_report_type}.csv"
    return file_path


def generate_metrics_summary_path(level='overall', phenotype=None):
    base_path = METRICS_PATH + "summary/"
    if level == 'overall':
        file_path = f"{base_path}overallmetrics_summary.csv"
    elif level == 'overall_confidence':
        file_path = f"{base_path}confidence/overallmetrics_summary.csv"
    elif level == 'confidence_all':
        file_path = f"{base_path}confidence/{level}_overallmetrics_summary.csv"
    elif level == 'confidence_low':
        file_path = f"{base_path}confidence/{level}_overallmetrics_summary.csv"
    elif level == 'confidence_medium':
        file_path = f"{base_path}confidence/{level}_overallmetrics_summary.csv"
    elif level == 'confidence_high':
        file_path = f"{base_path}confidence/{level}_overallmetrics_summary.csv"
    elif level == 'phenotype':
        if phenotype is None:
            raise ValueError("Phenotype must be specified when level is 'phenotype'.")
        file_path = f"{base_path}{phenotype}/metrics_summary.csv"
    else:
        raise ValueError(f"Unsupported level '{level}'. Expected 'overall' or 'phenotype'.")
    return file_path


def generate_metrics_summary_reasoning_path(level='overall', phenotype=None):
    base_path = REASONING_METRICS_PATH + "summary/"
    if level == 'overall':
        file_path = f"{base_path}overallmetrics_summary.csv"
    elif level == 'phenotype':
        if phenotype is None:
            raise ValueError("Phenotype must be specified when level is 'phenotype'.")
        file_path = f"{base_path}{phenotype}/metrics_summary.csv"
    else:
        raise ValueError(f"Unsupported level '{level}'. Expected 'overall' or 'phenotype'.")
    return file_path


def generate_misclassification_path(status='raw'):
    if status == 'raw':
        file_path = f"{MISCLASSIFICATION_PATH}misclassified_data.csv"
    elif status == 'reports':
        file_path = f"{MISCLASSIFICATION_ERROR_REPORT_PATH}"
    elif status == 'confidence_all':
        file_path = f"{MISCLASSIFICATION_PATH}all.csv"
    elif status == 'confidence_low':
        file_path = f"{MISCLASSIFICATION_PATH}low.csv"
    elif status == 'confidence_medium':
        file_path = f"{MISCLASSIFICATION_PATH}medium.csv"
    elif status == 'confidence_high':
        file_path = f"{MISCLASSIFICATION_PATH}high.csv"
    elif status == 'confidence_report':
        file_path = f"{MISCLASSIFICATION_ERROR_REPORT_PATH}confidence/"
    else:
        raise ValueError(f"Unsupported status '{status}'. Expected 'raw' or 'reports'.")
    return file_path


def generate_properclassification_path(status='raw'):
    if status == 'raw':
        file_path = f"{PROPERCLASSIFICATION_PATH}proper_classified_data.csv"
    elif status == 'reports':
        file_path = f"{PROPERCLASSIFICATION_ERROR_REPORT_PATH}"
    elif status == 'confidence_all':
        file_path = f"{PROPERCLASSIFICATION_PATH}all.csv"
    elif status == 'confidence_low':
        file_path = f"{PROPERCLASSIFICATION_PATH}low.csv"
    elif status == 'confidence_medium':
        file_path = f"{PROPERCLASSIFICATION_PATH}medium.csv"
    elif status == 'confidence_high':
        file_path = f"{PROPERCLASSIFICATION_PATH}high.csv"
    elif status == 'confidence_report':
        file_path = f"{PROPERCLASSIFICATION_ERROR_REPORT_PATH}confidence/"
    else:
        raise ValueError(f"Unsupported status '{status}'. Expected 'raw' or 'reports'.")
    return file_path


def generate_misclassification_reasoning_path(status='raw'):
    if status == 'raw':
        file_path = f"{REASONING_MISCLASSIFICATION_PATH}misclassified_data.csv"
    elif status == 'reports':
        file_path = f"{REASONING_MISCLASSIFICATION_ERROR_REPORT_PATH}"
    else:
        raise ValueError(f"Unsupported status '{status}'. Expected 'raw' or 'reports'.")
    return file_path


def generate_properclassification_reasoning_path(status='raw'):
    if status == 'raw':
        file_path = f"{REASONING_PROPERCLASSIFICATION_PATH}proper_classified_data.csv"
    elif status == 'reports':
        file_path = f"{REASONING_PROPERCLASSIFICATION_ERROR_REPORT_PATH}"
    else:
        raise ValueError(f"Unsupported status '{status}'. Expected 'raw' or 'reports'.")
    return file_path

def generate_reasoning_metrics_path(phenotype: str, model: str, analysis_type: str, metric_name: str) -> str:
    """
    Generate path for saving reasoning metrics analysis results.

    Args:
        phenotype: The phenotype being analyzed
        model: The model used for analysis
        analysis_type: Type of analysis (confidence, paths, tokens, decisions)
        metric_name: Name of the specific metric

    Returns:
        str: Path where the metric results should be saved
    """
    base_path = REASONING_METRICS_PATH

    # Create directory structure: base/phenotype/model/analysis_type/
    metrics_path = os.path.join(
        base_path,
        'analysis',
        phenotype,
        model,
        analysis_type,
        f"{metric_name}.csv"
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    return metrics_path

def generate_error_report(error_df):
    try:
        print("Generating error report...")

        # Existing analyses
        # Analysis per model
        errors_per_model = error_df['Model'].value_counts().reset_index()
        errors_per_model.columns = ['Model', '#Chunks Error']
        errors_per_model_hadm = error_df.groupby('Model')['hadm_id'].nunique().reset_index(name='#HADM Affected')
        errors_per_hadm_model = error_df.groupby('Model').apply(lambda x: x.shape[0] / x['hadm_id'].nunique()).reset_index(name='Errors per HADM')

        # Analysis per model and phenotype
        errors_per_model_and_phenotype = error_df.groupby(['Model', 'Phenotype']).size().reset_index(name='#Chunks Error')
        errors_per_model_and_phenotype_hadm = error_df.groupby(['Model', 'Phenotype'])['hadm_id'].nunique().reset_index(name='#HADM Affected')
        errors_per_hadm_model_phenotype = error_df.groupby(['Model', 'Phenotype']).apply(lambda x: x.shape[0] / x['hadm_id'].nunique()).reset_index(name='Errors per HADM')

        # Analysis per model configuration
        errors_per_config = error_df.groupby(['Phenotype', 'Configuration']).size().reset_index(name='#Chunks Error')
        errors_per_config_hadm = error_df.groupby(['Phenotype', 'Configuration'])['hadm_id'].nunique().reset_index(name='#HADM Affected')
        errors_per_hadm_config = error_df.groupby(['Phenotype', 'Configuration']).apply(lambda x: x.shape[0] / x['hadm_id'].nunique()).reset_index(name='Errors per HADM')

        # New analysis based on error_type
        # Analysis per model and error type
        errors_per_model_error_type = error_df.groupby(['Model', 'error_type']).size().reset_index(name='#Chunks Error')
        errors_per_model_error_type_hadm = error_df.groupby(['Model', 'error_type'])['hadm_id'].nunique().reset_index(name='#HADM Affected')
        errors_per_hadm_model_error_type = error_df.groupby(['Model', 'error_type']).apply(lambda x: x.shape[0] / x['hadm_id'].nunique()).reset_index(name='Errors per HADM')

        # Combine results
        errors_per_model = errors_per_model.merge(errors_per_model_hadm, on='Model').merge(errors_per_hadm_model, on='Model')
        errors_per_model_and_phenotype = errors_per_model_and_phenotype.merge(errors_per_model_and_phenotype_hadm, on=['Model', 'Phenotype']).merge(errors_per_hadm_model_phenotype, on=['Model', 'Phenotype'])
        errors_per_config = errors_per_config.merge(errors_per_config_hadm, on=['Phenotype', 'Configuration']).merge(errors_per_hadm_config, on=['Phenotype', 'Configuration'])

        # Merge new error type analysis
        errors_per_model = errors_per_model.merge(errors_per_model_error_type, on='Model', how='outer').merge(errors_per_model_error_type_hadm, on=['Model', 'error_type'], how='outer').merge(errors_per_hadm_model_error_type, on=['Model', 'error_type'], how='outer')

        return errors_per_model, errors_per_model_and_phenotype, errors_per_config

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_error_report(errors_per_model, errors_per_config, output_dir='error_reports'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        if errors_per_model is not None:
            errors_per_model.to_csv(os.path.join(output_dir, 'errors_per_model.csv'), index=False)
        if errors_per_config is not None:
            errors_per_config.to_csv(os.path.join(output_dir, 'errors_per_config.csv'), index=False)
        print("Error reports saved in", output_dir)
    except Exception as e:
        print("Failed to save error report.", e)

def convert_to_datetime(df, col):
    if isinstance(df, pd.DataFrame):
        df[col] = pd.to_datetime(df[col])
    elif isinstance(df, dd.DataFrame):
        df[col] = dd.to_datetime(df[col])
    else:
        raise TypeError("The input must be a pandas DataFrame or a Dask DataFrame.")
    return df

# LOAD ALL MIMIC IV TABLES IN MEMORY (warning: High memory lengthy process)
def load_mimiciv(core_mimiciv_path: object, note_mimiciv_path: object, n_sample: object
= None, cohort_ids = None, tables_to_return: object = None,
                     compute: object = False) -> object:
    """
    Load MIMIC-IV core and note data samples or full datasets with specified preprocessing.
    Note that the n_sample parameter is optional, if None, all data will be loaded.
    Sampling is based on subject_id but training/testing is based on hadm_id.

    Parameters:
    core_mimiciv_path (object): Path to the core MIMIC-IV CSV files.
    note_mimiciv_path (object): Path to the note MIMIC-IV CSV files.
    n_sample (object, optional): Number of unique subject_ids to sample. If None, load all data.
    cohort_ids (object, optional): List of hadm_ids to load. If None, load all data.
    tables_to_return (object, optional): List of tables to return. If None, return all tables.
    compute (object, optional): Whether to compute the resulting dataframes. Loading dataframes in memory:converting DD to pandas dataframes.

    Returns:
    object: A dictionary of preprocessed and sorted pandas dataframes specified in tables_to_return.
    """

    # If no specific tables requested, return all
    if tables_to_return is None:
        tables_to_return = ['admissions', 'patients', 'transfers', 'diagnoses_icd', 'drgcodes', 'emar', 'emar_detail',
                            'hcpcsevents', 'labevents', 'microbiologyevents', 'poe', 'poe_detail', 'prescriptions',
                            'procedures_icd', 'services', 'd_icd_diagnoses', 'd_icd_procedures', 'd_hcpcs',
                            'd_labitems',
                            'procedureevents', 'outputevents', 'inputevents', 'icustays', 'datetimeevents',
                            'chartevents',
                            'd_items', 'dsnotes', 'radnotes']

    # Helper function to load and optionally sample a CSV file
    def load_csv_dd(file_path, dtype_dict=None, sample_ids=None, cohort_ids=None, selected_columns=None,
                 compression='gzip', blocksize='32MB'):
        """
        Loads a CSV file into a Dask DataFrame, optionally filtering by sample_ids or cohort_ids, and selecting specific columns.
        This function supports reading from both local file systems and Google Cloud Storage.

        Parameters:
        - file_path (str): The path to the CSV file. Can be a local path or a GCS URL (e.g., 'gs://bucket-name/path/to/file.csv').
        - dtype_dict (dict): A dictionary specifying the data types for the columns.
        - sample_ids (list): A list of sample IDs to filter the DataFrame by.
        - cohort_ids (list): A list of cohort IDs to filter the DataFrame by.
        - selected_columns (list): A list of column names to load. If None, all columns are loaded.
        - compression (str): The compression type of the CSV file. Default is 'gzip'.
        - blocksize (str): The block size for reading the CSV file. Default is '32MB'.

        Returns:
        - A Dask DataFrame.
        """

        # Specify the columns to load if provided
        if is_gcs_path(file_path):
            fs = setup_gcs_access()
            with fs.open(file_path, 'rb') as f:
                print('Reading as pandas.df')
                if selected_columns:
                    df = pd.read_csv(f, usecols=selected_columns, dtype=dtype_dict, compression=compression)
                else:
                    df = pd.read_csv(f, dtype=dtype_dict, compression=compression)
        else:
            # For local files, directly use pandas.read_csv
            if selected_columns:
                df = dd.read_csv(file_path, usecols=selected_columns, dtype=dtype_dict, compression=compression,
                                 blocksize=blocksize)
            else:
                df = dd.read_csv(file_path, dtype=dtype_dict, compression=compression, blocksize=blocksize)

        # Filter by sample_ids if provided and applicable
        if sample_ids is not None and 'subject_id' in df.columns:
            df = df[df['subject_id'].isin(sample_ids)]

        # Filter by cohort_ids if provided and applicable
        if cohort_ids is not None and 'hadm_id' in df.columns:
            df = df[df['hadm_id'].isin(cohort_ids)]

        return df

    # Helper function for sampling subject IDs from admissions
    def sample_subject_ids(admissions_df, n_sample):
        # Ensure admissions_df is computed if it's a Dask DataFrame
        admissions_df = ensure_pandas_df(admissions_df)

        # Get the unique subject IDs
        unique_subject_ids = admissions_df['subject_id'].drop_duplicates().tolist()

        # If n_sample is greater than the number of unique subject IDs, sample all
        if n_sample >= len(unique_subject_ids):
            return unique_subject_ids

        # Otherwise, sample n_sample items directly
        sampled_subject_ids = admissions_df['subject_id'].drop_duplicates().sample(n=n_sample, random_state=1).tolist()

        return sampled_subject_ids
    # Helper function for processing dataframes
    def preprocess_dataframe(df, table_name):
        if table_name in ['admissions', 'transfers', 'prescriptions', 'emar', 'labevents', 'microbiologyevents', 'poe',
                          'services', 'procedureevents', 'outputevents', 'inputevents', 'icustays', 'datetimeevents',
                          'chartevents', 'dsnotes', 'radnotes']:
            date_time_columns = {
                'admissions': ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'],
                'transfers': ['intime', 'outtime'],
                'prescriptions': ['starttime', 'stoptime'],
                'emar': ['charttime', 'scheduletime', 'storetime'],
                #'labevents': ['charttime', 'storetime'],
                'microbiologyevents': ['chartdate', 'charttime', 'storedate'],
                'poe': ['ordertime'],
                'services': ['transfertime'],
                'procedureevents': ['starttime', 'endtime'],
                'outputevents': ['charttime'],
                'inputevents': ['starttime', 'endtime'],
                'icustays': ['intime', 'outtime'],
                'datetimeevents': ['charttime', 'storetime'],
                'chartevents': ['charttime'],
                'dsnotes': ['charttime'],
                'radnotes': ['charttime'],
            }
            for col in date_time_columns.get(table_name, []):
                df = convert_to_datetime(df, col)

        if table_name in ['diagnoses_icd', 'd_icd_diagnoses', 'procedures_icd', 'd_icd_procedures', 'hcpcsevents',
                          'd_hcpcs']:
            strip_columns = {
                'diagnoses_icd': ['icd_code', 'icd_version'],
                'd_icd_diagnoses': ['icd_code', 'icd_version'],
                'procedures_icd': ['icd_code', 'icd_version'],
                'd_icd_procedures': ['icd_code', 'icd_version'],
                'hcpcsevents': ['hcpcs_cd'],
                'd_hcpcs': ['code'],
            }
            for col in strip_columns.get(table_name, []):
                df[col] = df[col].str.strip()

        return df

    # Load admissions to sample subject_ids if needed
    admissions_path = core_mimiciv_path + 'hosp/admissions.csv.gz'
    admissions_dtype = {'admission_location': 'object', 'deathtime': 'object', 'edouttime': 'object',
                        'edregtime': 'object'}
    df_admissions = load_csv_dd(admissions_path, dtype_dict=admissions_dtype, compression='gzip')
    sampled_subject_ids = sample_subject_ids(df_admissions, n_sample) if n_sample is not None else None
    cohort_ids if cohort_ids is not None else None

    # Initialize dictionary to hold dataframes
    dfs = {}

    # Load and process each requested table
    for table in tables_to_return:
        try:
            print(f'''### Processing table {table} ###''')
            if table == 'admissions':
                dfs['admissions'] = df_admissions if n_sample is None else df_admissions[
                    df_admissions['subject_id'].isin(sampled_subject_ids)]
                dfs['admissions'] = preprocess_dataframe(dfs['admissions'], table)

            else:
                if table not in ['dsnotes', 'radnotes']:
                    file_path = core_mimiciv_path + {
                        'patients': 'hosp/patients.csv.gz',
                        'transfers': 'hosp/transfers.csv.gz',
                        'diagnoses_icd': 'hosp/diagnoses_icd.csv.gz',
                        'drgcodes': 'hosp/drgcodes.csv.gz',
                        'emar': 'hosp/emar.csv.gz',
                        'emar_detail': 'hosp/emar_detail.csv.gz',
                        'hcpcsevents': 'hosp/hcpcsevents.csv.gz',
                        'labevents': 'hosp/labevents.csv.gz',
                        'microbiologyevents': 'hosp/microbiologyevents.csv.gz',
                        'poe': 'hosp/poe.csv.gz',
                        'poe_detail': 'hosp/poe_detail.csv.gz',
                        'prescriptions': 'hosp/prescriptions.csv.gz',
                        'procedures_icd': 'hosp/procedures_icd.csv.gz',
                        'services': 'hosp/services.csv.gz',
                        'd_icd_diagnoses': 'hosp/d_icd_diagnoses.csv.gz',
                        'd_icd_procedures': 'hosp/d_icd_procedures.csv.gz',
                        'd_hcpcs': 'hosp/d_hcpcs.csv.gz',
                        'd_labitems': 'hosp/d_labitems.csv.gz',
                        'procedureevents': 'icu/procedureevents.csv.gz',
                        'outputevents': 'icu/outputevents.csv.gz',
                        'inputevents': 'icu/inputevents.csv.gz',
                        'icustays': 'icu/icustays.csv.gz',
                        'datetimeevents': 'icu/datetimeevents.csv.gz',
                        'chartevents': 'icu/chartevents.csv.gz',
                        'd_items': 'icu/d_items.csv.gz',
                        'dsnotes': 'note/discharge.csv.gz',
                        'radnotes': 'note/radiology.csv.gz'
                    }.get(table, '')  # Default to empty string if table not found
                else:
                    file_path = note_mimiciv_path + {
                        'dsnotes': 'note/discharge.csv.gz',
                        'radnotes': 'note/radiology.csv.gz'
                    }.get(table, '')  # Default to empty string if table not found


                dtype_dicts = {
                    'patients': {'dod': 'object'},
                    'transfers': {'careunit': 'object', 'hadm_id': 'float64'},
                    'd_labitems': {'loinc_code': 'object'},
                    'd_icd_procedures': {'icd_code': 'object', 'icd_version': 'object'},
                    'd_icd_diagnoses': {'icd_code': 'object', 'icd_version': 'object'},
                    'd_hcpcs': {'long_description': 'object'},
                    'diagnoses_icd': {'icd_code': 'object', 'icd_version': 'object'},
                    'drgcodes': {},
                    'emar': {'enter_provider_id': 'object', 'charttime': 'object',
                             'storetime': 'object', 'scheduletime': 'object'},
                    'emar_detail': {
                        'completion_interval': 'object', 'dose_due': 'object', 'dose_given': 'object',
                        'infusion_complete': 'object', 'infusion_rate_adjustment': 'object',
                        'infusion_rate_unit': 'object', 'new_iv_bag_hung': 'object',
                        'product_description_other': 'object', 'reason_for_no_barcode': 'object',
                        'restart_interval': 'object', 'route': 'object', 'side': 'object',
                        'site': 'object', 'continued_infusion_in_other_location': 'object',
                        'infusion_rate': 'object', 'non_formulary_visual_verification': 'object',
                        'prior_infusion_rate': 'object', 'product_amount_given': 'object',
                        'infusion_rate_adjustment_amount': 'object'
                    },
                    'hcpcsevents': {'hcpcs_cd': 'object'},
                    'labevents': {'subject_id': 'int32',
                            'itemid'    : 'int32',
                            'charttime' : 'object',
                            'storetime' : 'object',
                            'valuenum'  : 'float64',
                            'valueuom'  : 'object',
                            'value': 'object', 'flag': 'object',
                            'priority': 'object', 'comments': 'object'
                    },
                    'microbiologyevents': {'comments': 'object',
                                           'quantity': 'object',
                                           'ab_name': 'object',
                                           'dilution_comparison': 'object',
                                           'dilution_text': 'object',
                                           'interpretation': 'object',
                                           'org_name': 'object'
                                           },
                    'poe': {
                        'discontinue_of_poe_id': 'object',
                        'discontinued_by_poe_id': 'object',
                        'order_status': 'object'
                    },
                    'poe_detail': {},
                    'prescriptions': {'form_rx': 'object',
                                      'gsn': 'object',
                                      'dose_val_rx': 'object',
                                      'ndc': 'float64',
                                      'poe_seq': 'float64',
                                      'form_val_disp': 'object'},
                    'procedures_icd': {'icd_code': 'object', 'icd_version': 'object',
                                       'dose_val_rx': 'object', 'form_val_disp': 'object'
                                       },
                    'services': {'prev_service': 'object'},
                    'd_items': {},
                    'procedureevents': {
                        'value': 'object', 'secondaryordercategoryname': 'object',
                        'totalamountuom': 'object'
                    },
                    'outputevents': {'value': 'object'},
                    'inputevents': {
                        'value': 'object', 'secondaryordercategoryname': 'object',
                        'totalamountuom': 'object'
                    },
                    'icustays': {},
                    'datetimeevents': {'value': 'object'},
                    'chartevents': {'subject_id': 'int32',
                                      'hadm_id'    : 'int32',
                                      'itemid': 'int32',
                                      'stay_id': 'int32',
                                      'charttime' : 'object',
                                      'value': 'object',
                                      'valuenum'  : 'float64',
                                      'valueuom'  : 'object'
                                      },
                    'dsnotes': {'charttime': 'object', 'storetime': 'object', 'text': 'object'},
                    'radnotes': {'charttime': 'object', 'storetime': 'object', 'text': 'object'}
                }

                if file_path:  # If a file path was found for the table
                    # Initialize selected_columns
                    selected_columns = None
                    if 'labevents' in file_path:
                        selected_columns = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum',
                                            'valueuom']
                    elif 'prescriptions' in file_path:
                        selected_columns = ['subject_id', 'hadm_id', 'starttime', 'stoptime', 'ndc', 'gsn',
                                            'drug', 'drug_type']
                    elif 'chartevents' in file_path:
                        selected_columns = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value',
                                            'valuenum', 'valueuom']

                    if selected_columns:
                        dfs[table] = load_csv_dd(file_path, dtype_dict=dtype_dicts.get(table, None),
                                              sample_ids=sampled_subject_ids, cohort_ids=cohort_ids,
                                              selected_columns=selected_columns)
                    else:
                        # Handle the case where selected_columns is not set (None)
                        print("Reading all columns")


                    dfs[table] = load_csv_dd(file_path, dtype_dict=dtype_dicts.get(table, None),
                                          sample_ids=sampled_subject_ids, cohort_ids=cohort_ids, selected_columns=selected_columns)

                    dfs[table] = preprocess_dataframe(dfs[table], table)

                    if 'subject_id' in dfs[table].columns:
                        sort_columns = ['subject_id']  # Default sort column
                        if table in ['admissions', 'transfers', 'diagnoses_icd', 'drgcodes', 'emar', 'hcpcsevents',
                                     'labevents', 'microbiologyevents', 'poe', 'prescriptions', 'procedures_icd',
                                     'services', 'procedureevents', 'outputevents', 'inputevents', 'icustays',
                                     'datetimeevents', 'chartevents', 'dsnotes', 'radnotes']:
                            # Adjust sort_columns based on the specific needs of each table
                            sort_columns.extend(['hadm_id'])
                            if table in ['procedureevents', 'outputevents', 'inputevents', 'icustays', 'datetimeevents',
                                         'chartevents']:
                                sort_columns.append('stay_id')
                            elif table in ['dsnotes', 'radnotes']:
                                sort_columns.append('note_id')

                        # Sort by the first column
                        dfs[table] = dfs[table].sort_values(by=sort_columns[0])

                        # If there are more columns to sort by, sort by the next column
                        if len(sort_columns) > 1:
                            dfs[table] = dfs[table].sort_values(by=sort_columns[1])
                    else:
                        dfs[table] = dfs[table]
                if compute:
                    print(f'Computing table:', {table})
                    dfs[table] = ensure_pandas_df(dfs[table])
        except Exception as e:
            print(f"Failed processing table {table}: {e}")

    # Return dataframes in the order specified by tables_to_return in a dictionary
    return dfs


def cleanup_merged_dataframe(df_merged):
    """
    Cleans up a DataFrame after a merge operation by removing columns with '_x' suffixes
    and renaming columns with '_y' suffixes to their original names.

    Parameters:
    - df_merged: The merged DataFrame to clean up.

    Returns:
    - The cleaned DataFrame.
    """
    # Drop columns with '_y' suffix
    drop_cols = [col for col in df_merged.columns if col.endswith('_y')]
    df_merged.drop(drop_cols, axis=1, inplace=True)

    # Rename columns with '_y' suffix to original names (remove suffix)
    rename_dict = {col: col.rstrip('_x') for col in df_merged.columns if
                   col.endswith('_x')}
    df_merged.rename(columns=rename_dict, inplace=True)

    return df_merged.compute() if isinstance(df_merged, dd.DataFrame) else df_merged



def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def get_table(table_name: str, cohort = None, compute = False) -> pd.DataFrame:
    """
    Load a table from MIMIC-IV and return it as a pandas dataframe, providing cohort to filter table according to
    cohort's hadm_ids. For 'd_labitems' and 'labevents', the function returns a Dask DataFrame without computing it.

    Parameters:
    table_name (str): Name of the table to load
    cohort (pd.DataFrame): Pandas dataframe containing subject_id column

    Returns:
    pd.DataFrame: The requested table filtered by the cohort. For 'd_labitems' and 'labevents', a Dask DataFrame is returned.

    Raises:
    ValueError: If the table_name is not a valid MIMIC-IV table
    """
    all_tables = [
        "admissions",
        "patients",
        "transfers",
        "diagnoses_icd",
        "drgcodes",
        "emar",
        "emar_detail",
        "hcpcsevents",
        "labevents",
        "microbiologyevents",
        "poe",
        "poe_detail",
        "prescriptions",
        "procedures_icd",
        "services",
        "d_icd_diagnoses",
        "d_icd_procedures",
        "d_hcpcs",
        "d_labitems",
        "procedureevents",
        "outputevents",
        "inputevents",
        "icustays",
        "datetimeevents",
        "chartevents",
        "d_items",
        "dsnotes",
        "radnotes",
    ]

    if table_name not in all_tables:
        raise ValueError(f"Table name not in list of tables: {all_tables}")

    # load table using load_mimiciv
    df_dict = load_mimiciv(
        CORE_MIMICIV_PATH,
        NOTE_MIMICIV_PATH,
        tables_to_return=[table_name],
        cohort_ids=cohort["hadm_id"].unique() if cohort is not None else None,
        compute=compute
    )

    # return the table from the dictionary
    df = df_dict[table_name]

    # Debugging: Print the type of df to ensure it's a Dask DataFrame
    print(f"Type of df: {type(df)}")

    # Check if the table is 'd_labitems' or 'labevents' and return a Dask DataFrame without computing
    # TODO: if script run locally remove condition
    def off():
        if table_name in ['d_labitems', 'labevents', 'chartevents', 'radnotes', 'dsnotes']:
            df = df if isinstance(df, dd.DataFrame) else dd.from_pandas(df, npartitions=100)
            print(f"Returning {isinstance(df, dd.DataFrame)} Dask DataFrame for table {table_name}")
            return df
        else:
            try:
                # Compute all the small datframes
                return ensure_pandas_df(df)
            except Exception as e:
                print(f"Error computing DataFrame: {e}")
                return None
    return ensure_pandas_df(df)


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame from a local path or a Google Cloud Storage URL.

    Parameters:
    file_path (str): Path to the CSV file. Can be a local path or a GCS URL (e.g., 'gs://bucket-name/path/to/file.csv').

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """

    # Check if the file path is a GCS URL
    if is_gcs_path(file_path):
        fs = setup_gcs_access()
        # Use gcsfs to open the GCS file
        with fs.open(file_path, 'rb') as f:
            if file_path.endswith('.gz'):
                try:
                    # Attempt 1: Read as gzipped file
                    df = pd.read_csv(f, compression='gzip', header=0)
                except gzip.BadGzipFile:
                    print(
                        f"Warning: File {file_path} has .gz extension but is not a valid gzipped file. Attempting to read as plain CSV.")
                    f.seek(0)  # Reset stream position
                    df = pd.read_csv(f, header=0,
                                     compression=None)  # Attempt 2: Read as plain CSV, explicitly no compression

            else:
                df = pd.read_csv(f)
    else:
        # Load normally if it's a local file
        df = pd.read_csv(file_path)
    return df



def read_gzip_or_parts(path, use_gcs=True, chunk_size=None):
    fs = None
    if use_gcs and is_gcs_path(path):
        try:
            fs = setup_gcs_access()
        except Exception as e:
            print(f"Failed to setup GCS access: {e}")
            return pd.DataFrame()

    if not path_exists(path): # path_exists needs use_gcs argument
        print(f"File not found: {path}")
        return pd.DataFrame()

    # Strategy:
    # 1. Try to read as GZIP. This handles .csv.gz and .csv files that are actually gzipped.
    # 2. If that fails with BadGzipFile, it's not gzipped, so try to read as plain CSV.
    # 3. Handle other errors.

    try:
        # Attempt to read as GZIP first
        file_object_binary = None
        if use_gcs and fs:
            file_object_binary = fs.open(path, mode='rb')
        elif not is_gcs_path(path): # Local file
            file_object_binary = open(path, mode='rb')
        else: # GCS path but use_gcs is False or fs failed to initialize
            print(f"Cannot open GCS path {path} without GCS access.")
            return pd.DataFrame()

        with file_object_binary:
            if chunk_size:
                dfs_list = [chunk for chunk in pd.read_csv(file_object_binary, compression='gzip', chunksize=chunk_size, header=0)]
                if dfs_list:
                    print(f"Successfully read as GZIP (chunked): {path}")
                    return pd.concat(dfs_list, ignore_index=True)
                else: # Empty gzipped file
                    print(f"File read as GZIP but was empty (chunked): {path}")
                    return pd.DataFrame()
            else: # Not chunking
                df = pd.read_csv(file_object_binary, compression='gzip', header=0)
                print(f"Successfully read as GZIP: {path}")
                return df
    except gzip.BadGzipFile:
        print(f"File {path} is not a valid GZIP file. Attempting as plain CSV.")
        # Proceed to plain CSV reading attempt
    except (UnicodeDecodeError, pd.errors.ParserError) as e_content:
        # This means it was likely valid gzip, but the *uncompressed content* is not valid UTF-8 CSV or is malformed.
        print(f"Error decoding GZIP content or parsing CSV for {path}: {type(e_content).__name__} - {e_content}. Attempting as plain CSV.")
    except FileNotFoundError: # Specifically for local files if open() fails before with statement
        print(f"File not found (during GZIP open attempt for local file): {path}")
        return pd.DataFrame()
    except Exception as e_gzip_other: 
        print(f"Unexpected error during GZIP read of {path}: {type(e_gzip_other).__name__} - {e_gzip_other}. Attempting as plain CSV.")

    # Attempt to read as plain CSV (if GZIP attempt failed appropriately)
    try:
        file_object_text = None
        if use_gcs and fs:
            file_object_text = fs.open(path, mode='rt', encoding='utf-8')
        elif not is_gcs_path(path): # Local file
            file_object_text = open(path, mode='rt', encoding='utf-8')
        else: # GCS path but use_gcs is False or fs failed to initialize
            print(f"Cannot open GCS path {path} for plain text read without GCS access.")
            return pd.DataFrame() # Should have been caught by path_exists or GZIP block

        with file_object_text:
            if chunk_size:
                dfs_list = [chunk for chunk in pd.read_csv(f_obj_text, chunksize=chunk_size, header=0)]
                if dfs_list:
                    print(f"Successfully read as plain text CSV (chunked): {path}")
                    return pd.concat(dfs_list, ignore_index=True)
                else: # Empty plain file
                    print(f"File read as plain text CSV but was empty (chunked): {path}")
                    return pd.DataFrame()
            else: # Not chunking
                df = pd.read_csv(file_object_text, header=0)
                print(f"Successfully read as plain text CSV: {path}")
                return df
    except FileNotFoundError:  # Specifically for local files if open() fails
        print(f"File not found (during plain text open attempt for local file): {path}")
        return pd.DataFrame()
    except Exception as e_plain_csv:
        print(f"Failed to read {path} as plain text CSV. Error: {type(e_plain_csv).__name__} - {e_plain_csv}")
        return pd.DataFrame()

def save_gzip(df, path):
    """
    Saves a DataFrame to a .gz file, either locally or to Google Cloud Storage, replacing a directory of part files if it exists.

    Args:
    df (pandas.DataFrame): DataFrame to save.
    path (str): Path where the DataFrame should be saved.
    """

    def ensure_gcs_dir(directory):
        if not fs.exists(directory):
            fs.mkdir(directory)

    if is_gcs_path(path):
        fs = setup_gcs_access()
        base_dir = '/'.join(path.split('/')[:-1])  # Correctly handling GCS path for directories

        if fs.exists(path) and fs.isfile(path):
            fs.rm(path)
        ensure_gcs_dir(base_dir)

        with fs.open(path, 'wb') as f:
            df.to_csv(f, compression='gzip', index=False)
    else:
        base_dir = os.path.dirname(path)
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
        os.makedirs(base_dir, exist_ok=True)
        df.to_csv(path, compression='gzip', index=False)


def save_csv(df, path, type = 'csv'):
    """
    Saves a DataFrame to a .gz file, either locally or to Google Cloud Storage, replacing a directory of part files if it exists.

    Args:
    df (pandas.DataFrame): DataFrame to save.
    path (str): Path where the DataFrame should be saved.
    """

    def ensure_gcs_dir(directory):
        if not fs.exists(directory):
            fs.mkdir(directory)

    if is_gcs_path(path):
        fs = setup_gcs_access()
        base_dir = '/'.join(path.split('/')[:-1])

        if fs.exists(path) and fs.isfile(path):
            fs.rm(path)
        ensure_gcs_dir(base_dir)

        if type == 'csv':
            with fs.open(path, 'wb') as f:
                df.to_csv(f, index=False)
            logging.info(f"File saved to GCS: {path}")
        else:
            with fs.open(path, 'wb') as f:
                df.to_csv(f, compression='gzip', index=False)
            logging.info(f"File saved to GCS: {path}")
    else:
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
        if type == 'csv':
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, compression='gzip', index=False)
            logging.info(f"File saved locally: {path}")


def clean_json_etp(text):
    if not text:
        logging.error("No text provided for cleaning.")
        return None

    # Ensure the string is enclosed in curly braces
    if not text.strip().startswith('{'):
        text = '{' + text.strip().rstrip('}') + '}'

    # Use a more robust regex to match key-value pairs
    pairs = re.findall(r'"([^"]+)"\s*:\s*("[^"]*"|\d+)', text)

    cleaned_pairs = []
    for key, value in pairs:
        # Clean the value to ensure it is either a quoted string or a numeric value
        cleaned_value = re.sub(r'[^\w\s.-]', '', value.strip('"'))
        cleaned_value = f'"{cleaned_value}"' if not cleaned_value.isdigit() else cleaned_value
        cleaned_pairs.append(f'"{key}": {cleaned_value}')

    # Reconstruct the JSON string with cleaned values
    cleaned_json_str = '{' + ', '.join(cleaned_pairs) + '}'

    # Try to parse the cleaned JSON string to ensure it's valid
    try:
        json.loads(cleaned_json_str)
        return cleaned_json_str
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from cleaned string: {cleaned_json_str}. Error: {e}. Raw string {text}")
        return None


def check_double_bracket(text):
    # This function checks for nested JSON and tries to extract the innermost JSON.
    match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
    if match:
        # Extract the innermost JSON
        inner_match = re.search(r'\{[^{}]*\}', match.group(), re.DOTALL)
        if inner_match:
            return inner_match.group()
        else:
            print("No inner JSON found")
            return None
    else:
        # If no nested JSON is found, attempt to fix and validate the JSON
        fixed_text = fix_json(text)
        if validate_json(fixed_text):
            return fixed_text
        else:
            return text

def fix_json(text):
    # Add a single opening bracket if missing
    if text.strip() and not text.strip().startswith('{'):
        text = '{' + text

    # Add a single closing bracket if missing
    if text.strip() and not text.strip().endswith('}'):
        text += '}'

    return text


def validate_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def clean_and_validate_etp_json(json_data):
    """
    Cleans and validates the JSON data to ensure each value is numeric.
    Non-numeric values are converted to '0'.
    """

    for key, value in json_data.items():
        # Convert value to string, default to '0' if None
        value_str = str(value) if value is not None else '0'
        match = re.search(r'\d+', value_str)
        if match:
            # Update the value with the numeric part only
            json_data[key] = match.group()
        else:
            json_data[key] = '0'

    # Double check to ensure all values are strictly numeric
    for key, value in json_data.items():
        # Ensure value is a string, default to '0' if None
        value_str = str(value) if value is not None else '0'
        if not re.match(r'^\d+$', value_str):
            json_data[key] = '0'

    logging.debug(f"This is the validated JSON {json_data}") # Configure logging

    return json_data


def clean_json_classic(text):
    if not text:
        logging.error("No text provided for cleaning.")
        return None

    # Normalize quotes and remove non-standard characters
    text = re.sub(r'[""]', '"', text)
    text = text.replace('`', '"').replace('\\_', '_')

    # Remove comments (anything that starts with // to the end of the line)
    text = re.sub(r'//.*', '', text)

    # Handle escaped quotes and other escape characters
    text = text.replace(r'\"', '"').replace("\\\\", "\\")

    # Extract the JSON-like object from the string
    try:
        json_start = text.index('{')
        json_end = text.rindex('}') + 1
        json_str = text[json_start:json_end]
    except ValueError:
        logging.error(f"Failed to locate JSON boundaries in text : {text}")
        return None

    # Ensure that all keys and string values are correctly quoted
    json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)  # Quote keys
    json_str = re.sub(r':\s*([a-zA-Z]+)', r': "\1"', json_str)  # Quote unquoted string values


    # Validate and return JSON string
    try:
        json.loads(json_str)  # Try parsing the JSON
        return json_str
    except json.JSONDecodeError as e:
        logging.debug(f"Failed to decode JSON from cleaned string: {json_str}. Error: {e}")
        try:
            # Extract the justification field and clean it
            match = re.search(r'"justification"\s*:\s*"([^}]+)"', json_str)
            if match:
                # Get the justification text and clean it
                justification_text = match.group(1)

                # Remove all quotes and normalize spaces
                cleaned_justification = re.sub(r'[""]', '', justification_text)
                cleaned_justification = ' '.join(cleaned_justification.split())

                # Create new JSON string with cleaned justification
                json_parts = json_str.split('"justification"')
                new_json_str = (json_parts[0] +
                                '"justification": "' +
                                cleaned_justification +
                                '"' +
                                (json_parts[1][json_parts[1].find('}'):] if '}' in json_parts[1] else '}'))

                # Validate the cleaned JSON
                json.loads(new_json_str)
                logging.info("Successfully cleaned and validated JSON after justification cleaning")
                return new_json_str

            else:
                logging.error("Could not find justification field to clean")
                return None

        except (json.JSONDecodeError, Exception) as e2:
            logging.error(f"Failed to clean and parse JSON after justification cleaning: {e2}")
            return None


def check_double_bracket_v2(text):
    """
    Checks for nested JSON and extracts the most valid JSON structure.
    Handles multiple levels of nesting and multiple JSON objects.

    Args:
        text (str): Input text potentially containing JSON

    Returns:
        str: Most valid JSON structure found, or original text if none found
    """
    # Find all potential JSON objects (including nested ones)
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    matches = list(re.finditer(json_pattern, text, re.DOTALL))

    if not matches:
        return text

    # Try each match to find valid JSON, prioritizing larger structures
    matches = sorted(matches, key=lambda x: len(x.group()), reverse=True)
    for match in matches:
        try:
            potential_json = match.group()
            json.loads(potential_json)  # Validate JSON structure
            return potential_json
        except json.JSONDecodeError:
            continue

    # If no valid JSON found, return the largest match
    return matches[0].group() if matches else text


def is_valid_prediction_list(predictions, k=3):
    """
    Check if the predictions list has at least k valid numerical entries.

    Args:
        predictions: List or string representation of a list containing predictions
        k (int): Minimum number of valid predictions required (default: 3)

    Returns:
        bool: True if there are at least k valid numerical predictions, False otherwise

    Raises:
        ValueError: If input format is invalid
        TypeError: If input type is not supported
    """

    def is_numeric(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    try:
        # Handle different input types
        if isinstance(predictions, str):
            # Remove brackets and split by comma if string input
            cleaned = predictions.strip('[]').split(',')
            predictions_list = [s.strip().strip("'\"") for s in cleaned if s.strip()]
        elif isinstance(predictions, (list, tuple)):
            predictions_list = predictions
        else:
            raise TypeError(f"Unsupported input type: {type(predictions)}")

        # Convert valid numbers to float
        valid_predictions = [
            float(pred) for pred in predictions_list
            if pred is not None and is_numeric(pred)
        ]

        return len(valid_predictions) >= k

    except Exception as e:
        raise ValueError(f"Invalid input format: {str(e)}")

def clean_json_classic_v2(text):
    """
    Cleans and normalizes JSON text to handle common formatting issues.
    Designed to be generalizable for any JSON structure.

    Args:
        text (str): Text containing JSON to clean

    Returns:
        str: Cleaned JSON string or None if cleaning fails
    """
    if not text:
        logging.error("No text provided for cleaning.")
        return None

    try:
        # Normalize quotes and remove non-standard characters
        text = re.sub(r'["""]', '"', text)
        text = text.replace('`', '"').replace('\\_', '_')

        # Remove comments and extra whitespace
        text = re.sub(r'//.*?(\n|$)', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Extract JSON structure
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None
        text = json_match.group()

        # Fix issues with multiple quotes in justification field
        text = re.sub(r':\s*"([^"]*)"([^"]*)"([^"]*)"(?=\s*,|\s*})', r': "\1\2\3"', text)
        
        # Fix issues with confidence field having extra quotes
        text = re.sub(r'"confidence""\s*:', r'"confidence":', text)
        
        # Pre-process step: normalize certain patterns that often cause issues
        text = re.sub(r':\s*"([^"]*)"(?=\s*[,}])', r': "\1"', text)  # Normalize spacing around quoted values
        text = re.sub(r':\s*([^",{\[\s][^,}\]]*?)(?=\s*[,}])', r': "\1"', text)  # Quote unquoted values
        text = re.sub(r'([{,])\s*([^"\s][^:]*?)(?=\s*:)', r'\1 "\2"', text)  # Quote unquoted keys

        # Handle arrays
        text = re.sub(r'\[\s*([^"\[\]{},]+(?:\s*,\s*[^"\[\]{},]+)*)\s*\]', 
                     lambda m: '[' + ','.join(f'"{x.strip()}"' for x in m.group(1).split(',')) + ']', 
                     text)

        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Final cleanup
        text = re.sub(r'"\s*:\s*', '": ', text)  # Normalize spacing around colons
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.replace('}.', '}')  # Remove trailing periods
        
        # Remove any remaining double quotes within values
        text = re.sub(r'(?<="[^"]*)"([^"]*)"(?=[^"]*"(?:\s*[,}]))', r'\1', text)
        
        # Validate JSON structure
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            # If validation fails, try one more aggressive cleaning
            text = re.sub(r'([^"\\])\\([^"\\])', r'\1\2', text)  # Remove invalid escapes
            text = re.sub(r'\\+([^"\\])', r'\1', text)  # Clean up multiple escapes
            json.loads(text)  # Final validation
            return text

    except json.JSONDecodeError as e:
        logging.error(f"JSON validation failed: {e}\nText: {text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error cleaning JSON: {e}\nText: {text}")
        return None



def etp_grammar(phenotype, json_output):
    def normalize_key(key):
        key = re.sub(r'[^a-zA-Z0-9_]', ' ', key)
        key = key.lower()
        return key

    def normalize_value(value):
        # Convert value to int if it's a string of digits, then check if it's 1
        try:
            value = int(value)
        except ValueError:
            pass
        return 1 if value == 1 else 0

    try:
        valid_keys = PHENOTYPES_ETP_KEYS.get(phenotype, [])
        synonyms = ETP_ABBREVIATIONS.get(phenotype, [])
        corrected_json = {}
        validated_keys_count = 0

        normalized_valid_keys = [normalize_key(k) for k in valid_keys]

        for key, value in json_output.items():
            normalized_key = normalize_key(key)
            value = normalize_value(value)  # Normalize value once here

            # Check against valid keys
            if normalized_key in normalized_valid_keys:
                corrected_key = valid_keys[normalized_valid_keys.index(normalized_key)]
                corrected_json[corrected_key] = value
                validated_keys_count += 1
            else:
                # Check against synonyms
                matched = False
                for synonym in synonyms:
                    if normalized_key == normalize_key(synonym):
                        corrected_key = phenotype
                        corrected_json[corrected_key] = value
                        validated_keys_count += 1
                        matched = True
                        break
                if not matched:
                    # Check for close matches
                    corrected_key = get_close_matches(normalized_key, normalized_valid_keys, n=1, cutoff=0.8)
                    if corrected_key:
                        corrected_key = valid_keys[normalized_valid_keys.index(corrected_key[0])]
                        corrected_json[corrected_key] = value
                        validated_keys_count += 1

        total_keys = len(json_output)
        validation_percentage = validated_keys_count / total_keys if total_keys > 0 else 0

        return corrected_json, validation_percentage

    except KeyError as e:
        print(f"Error: Key not found in PHENOTYPES_ETP_KEYS: {str(e)}")
        return None, 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return None, 0

def load_baseline(phenotype, train, noisy_label, use_shared_paths=False):
    """
    Load baseline features and corrector columns for a given phenotype.
    If use_shared_paths is True, loads from shared baseline file and deduplicates rows by hadm_id.
    Returns (y, features, corrector) if use_shared_paths is False, else (features, corrector).
    """
    # TODO: change back to generate_file_path for prod _local
    baseline_path = generate_file_path(phenotype, 'BASELINE', train, noisy_label, long=True, use_shared_paths=use_shared_paths)
    if not path_exists(baseline_path):
        print(f"[SKIPPING BASELINE DATA] File does not exist: {baseline_path}")
        raise ValueError(f"Baseline data not found for {phenotype}, must stop training here")
    baseline = read_gzip_or_parts(baseline_path)
    print(f"Loaded BASELINE data, shape: {baseline.shape}")

    # Import harmonize_race only when needed to avoid circular import
    from scripts.model.feature_utils import harmonize_race

    if use_shared_paths == False:
        # Assuming 'hadm_id' and 'y' are columns in the baseline dataframe
        y = baseline[['hadm_id', 'y']].set_index('hadm_id')
        features = baseline[['hadm_id', 'marital_status', 'race', 'age', 'gender']].set_index('hadm_id')
        features['race'] = features['race'].apply(harmonize_race)
        corrector = baseline[['hadm_id', 'admissions_number', 'los']].set_index('hadm_id')
        return y, features, corrector
    else:
        # Add a temporary column to track missing values per row
        baseline['missing_count'] = baseline.isna().sum(axis=1)
        baseline_sorted = baseline.sort_values(['hadm_id', 'missing_count'])
        baseline_clean = baseline_sorted.drop_duplicates(subset='hadm_id', keep='first')
        baseline = baseline_clean.drop(columns='missing_count')
        features = baseline[['hadm_id', 'marital_status', 'race', 'age', 'gender']].set_index('hadm_id')
        features['race'] = features['race'].apply(harmonize_race)
        corrector = baseline[['hadm_id', 'admissions_number', 'los']].set_index('hadm_id')
        return features, corrector

