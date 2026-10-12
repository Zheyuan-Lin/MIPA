###########################################################################################################
#                      Utils.py Basic dataframe processing and wrangling functions
###########################################################################################################
#

import os
import sys
import gc
import tempfile
from scipy.sparse import coo_matrix, csr_matrix, vstack
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
    if use_shared_paths:
        # Paths for shared data (identical hadm_id universe across phenotypes)
        if train:
            base_path = COHORT_SHARED_TRAIN_LONG_PATH if long else COHORT_SHARED_TRAIN_ROOT_PATH
            suffix = SHARED_TRAIN_DATA_FILE_SUFFIX
        else:
            base_path = COHORT_SHARED_TEST_VAL_LONG_PATH if long else COHORT_SHARED_TEST_VAL_ROOT_PATH
            suffix = SHARED_DATA_FILE_SUFFIX
        # For shared paths, phenotype is not part of the directory structure,
        # and noisy_label/validation flags might be standardized (e.g., always False for golden test/val data)
        file_path = f"{base_path}{df_type}{suffix}.csv.gz"
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
    if use_shared_paths:
        # Shared NLP path (identical hadm_id universe across phenotypes)
        if train:
            base_path = COHORT_SHARED_TRAIN_NLP_PATH
            suffix = SHARED_TRAIN_DATA_FILE_SUFFIX
        else:
            base_path = COHORT_SHARED_TEST_VAL_NLP_PATH
            suffix = SHARED_DATA_FILE_SUFFIX
        file_path = f"{base_path}{df_type}{suffix}.csv.gz"
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
    if use_shared_paths:
        # For shared summaries, phenotype is not part of the directory structure.
        if train:
            base_path = COHORT_SHARED_TRAIN_SUMMARY_PATH
            suffix = SHARED_TRAIN_DATA_FILE_SUFFIX
        else:
            base_path = COHORT_SHARED_TEST_VAL_SUMMARY_PATH
            suffix = SHARED_DATA_FILE_SUFFIX
        file_path = f"{base_path}{df_type}{suffix}_summary.csv"

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
    if use_shared_paths:
        # For shared dictionaries, phenotype is not part of the directory structure.
        if train:
            base_path = COHORT_SHARED_TRAIN_DICT_PATH
            suffix = SHARED_TRAIN_DATA_FILE_SUFFIX
        else:
            base_path = COHORT_SHARED_TEST_VAL_DICT_PATH
            suffix = SHARED_DATA_FILE_SUFFIX
        file_path = f"{base_path}{df_type}{suffix}_dict.csv"
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
    if use_shared_paths:
        # For shared pivot tables, phenotype is not part of the directory structure.
        # noisy_label might be standardized (e.g., always False for golden test/val data)
        if train:
            base_path = COHORT_SHARED_TRAIN_PIVOT_PATH
            suffix = SHARED_TRAIN_DATA_FILE_SUFFIX
        else:
            base_path = COHORT_SHARED_TEST_VAL_PIVOT_PATH
            suffix = SHARED_DATA_FILE_SUFFIX
        file_path = f"{base_path}{df_type}{suffix}_pivot.csv.gz"
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
            # gzip can't be split into multiple partitions (dask always falls back to a
            # single partition for it -- see the "gzip compression does not support
            # breaking apart files" warning), so there is no parallelism to lose by
            # computing here. Doing so BEFORE the isin() filters below matters: applying
            # isin() against a large (100K+ element) cohort_ids/sample_ids array while the
            # frame is still a lazy dask-expr graph makes query planning pathologically
            # slow (many minutes, observed on procedures_icd/prescriptions-sized tables)
            # even though the same filter is near-instant in plain pandas once materialized.
            df = df.compute()

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
                    'd_items': {'unitname': 'object'},
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
                    'radnotes': {'charttime': 'object', 'storetime': 'object', 'text': 'object',
                                 'hadm_id': 'float64'}
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

                    if not selected_columns:
                        print("Reading all columns")

                    # (selected_columns is passed straight through below -- no need to call
                    # load_csv_dd twice, once per branch, and discard the first result.)
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


def load_csv_filtered_chunked(file_path, cohort_ids=None, dtype_dict=None, selected_columns=None,
                              datetime_columns=None, chunksize=1_000_000,
                              dropna_subset=None, dedup_subset=None):
    """
    Stream a gzip CSV in fixed-size row chunks, keeping only rows whose 'hadm_id' is in
    cohort_ids, without ever materializing the whole file in memory.

    Needed for tables where the compressed file is one unsplittable gzip stream: gzip
    cannot be partitioned, so dd.read_csv(...).compute() must decompress the ENTIRE table
    into RAM before any cohort row-filter can shrink it. On chartevents/labevents
    (10GB+ compressed, hundreds of millions of rows) that OOM-kills a machine with limited
    RAM -- the OS kills the process directly, so no Python exception/traceback is raised.
    Reading with pandas' own chunksize instead bounds peak memory to roughly one chunk's
    worth of rows.

    If dedup_subset is given (e.g. ['hadm_id', 'itemid']), the accumulated result is
    compacted via drop_duplicates(subset=dedup_subset, keep='first') after every chunk,
    bounding memory to roughly the number of unique combinations rather than the raw row
    count. This matters a lot for a table like chartevents: the hadm_id-only cohort filter
    barely shrinks it (chartevents' ~85K total hadm_ids are nearly all within a broad
    training-population cohort), and vitals get recorded every few minutes for the length of
    an ICU stay, so the eventual per-(hadm_id, itemid) dedup that preproc_chart does at the
    very end can collapse the row count by 100x+ -- doing that dedup here, during streaming,
    instead of after reading everything, is the difference between finishing in minutes and
    the filtered-but-not-yet-deduped intermediate alone approaching the size of the 12GB
    source file (confirmed: it reached ~7.8GB, growing, before being stopped).
    Trade-off: this dedup happens before preproc_chart's own stay_id-merge and
    event-time-from-admit >= 0 filtering, so if a given (hadm_id, itemid) pair has both a
    valid and an invalid (wrong stay_id, or negative time delta) raw measurement, keeping an
    arbitrary "first" one here could occasionally keep the invalid one where the original
    (materialize-everything-then-dedup) order might have kept the valid one. Both orderings
    are equally arbitrary about *which* duplicate survives when several valid ones exist, so
    this only changes behavior in the rare case a pair has a mix of valid/invalid readings.
    dropna_subset additionally drops rows with NaN in those columns per-chunk (e.g. ['value']
    for chartevents), reducing volume further before the dedup step.

    Without dedup_subset, filtered chunks are appended straight to a temp gzip CSV on disk
    instead of being kept in a Python list for a final pd.concat -- with a broad cohort
    filter the result can still be tens of millions of rows, and holding "list of chunks" +
    "concat copy" at once doubles the peak. Writing incrementally bounds the peak to about
    one chunk plus the final single read-back below.
    """
    cohort_ids_set = set(cohort_ids) if cohort_ids is not None else None
    reader = pd.read_csv(file_path, compression='gzip', dtype=dtype_dict,
                         usecols=selected_columns, chunksize=chunksize)

    if dedup_subset:
        accumulated = None
        for chunk in reader:
            if cohort_ids_set is not None and 'hadm_id' in chunk.columns:
                chunk = chunk[chunk['hadm_id'].isin(cohort_ids_set)]
            if dropna_subset:
                chunk = chunk.dropna(subset=dropna_subset)
            if chunk.empty:
                continue
            for col in (datetime_columns or []):
                chunk[col] = pd.to_datetime(chunk[col])
            chunk = chunk.drop_duplicates(subset=dedup_subset, keep='first')
            accumulated = chunk if accumulated is None else pd.concat(
                [accumulated, chunk], ignore_index=True
            ).drop_duplicates(subset=dedup_subset, keep='first')
        return accumulated if accumulated is not None else pd.DataFrame(columns=selected_columns or [])

    tmp_path = tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False).name
    wrote_any_rows = False
    try:
        for chunk in reader:
            if cohort_ids_set is not None and 'hadm_id' in chunk.columns:
                chunk = chunk[chunk['hadm_id'].isin(cohort_ids_set)]
            if dropna_subset:
                chunk = chunk.dropna(subset=dropna_subset)
            if chunk.empty:
                continue
            for col in (datetime_columns or []):
                chunk[col] = pd.to_datetime(chunk[col])
            # Each to_csv(mode='a', compression='gzip') call appends a new gzip member;
            # Python's gzip reader (and thus pandas' later read_csv) transparently
            # concatenates members back into one stream on read.
            chunk.to_csv(tmp_path, mode='a', header=not wrote_any_rows, index=False, compression='gzip')
            wrote_any_rows = True

        if not wrote_any_rows:
            return pd.DataFrame(columns=selected_columns or [])

        result = pd.read_csv(tmp_path, compression='gzip', dtype=dtype_dict)
        for col in (datetime_columns or []):
            result[col] = pd.to_datetime(result[col])
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# File specs for tables whose compressed size (10GB+) makes full in-memory materialization
# unsafe -- routed through load_csv_filtered_chunked() in get_table() instead of Dask.
_LARGE_TABLE_SPECS = {
    'labevents': {
        'path': CORE_MIMICIV_PATH + 'hosp/labevents.csv.gz',
        'selected_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'],
        'dtype_dict': {'subject_id': 'int32', 'itemid': 'int32', 'valuenum': 'float64',
                       'value': 'object', 'valueuom': 'object', 'charttime': 'object'},
        'datetime_columns': ['charttime'],
        # See load_csv_filtered_chunked's dedup_subset docstring: labs are drawn repeatedly
        # over a stay, and preproc_labs itself only keeps one row per (hadm_id, itemid) in
        # the end (time=False) -- doing that dedup during streaming instead of after
        # materializing everything keeps peak memory bounded.
        'dropna_subset': ['value'],
        'dedup_subset': ['hadm_id', 'itemid'],
    },
    'chartevents': {
        'path': CORE_MIMICIV_PATH + 'icu/chartevents.csv.gz',
        'selected_columns': ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'],
        'dtype_dict': {'subject_id': 'int32', 'hadm_id': 'int32', 'itemid': 'int32', 'stay_id': 'int32',
                       'valuenum': 'float64', 'value': 'object', 'valueuom': 'object', 'charttime': 'object'},
        # Same rationale as labevents above -- confirmed necessary by measurement: without
        # this, the filtered-but-not-yet-deduped intermediate reached ~7.8GB (and climbing)
        # against a 12GB source file, because the hadm_id-only cohort filter barely shrinks
        # chartevents (its ~85K total hadm_ids are nearly all within a broad training
        # cohort) while vitals are recorded every few minutes for the length of an ICU stay.
        'dropna_subset': ['value'],
        'dedup_subset': ['hadm_id', 'itemid'],
        'datetime_columns': ['charttime'],
    },
}


# Maps every get_table()-recognized table name to its MIMIC-IV 2.2 BigQuery dataset
# (physionet-data.mimiciv_2_2_hosp / mimiciv_2_2_icu). Table names themselves match the local
# csv.gz basenames exactly (confirmed by querying the actual schemas) -- only the dataset
# prefix differs by hosp/ vs icu/ in the original local file layout.
_BIGQUERY_TABLE_DATASET = {
    'admissions': 'mimiciv_2_2_hosp', 'patients': 'mimiciv_2_2_hosp', 'transfers': 'mimiciv_2_2_hosp',
    'diagnoses_icd': 'mimiciv_2_2_hosp', 'drgcodes': 'mimiciv_2_2_hosp', 'emar': 'mimiciv_2_2_hosp',
    'emar_detail': 'mimiciv_2_2_hosp', 'hcpcsevents': 'mimiciv_2_2_hosp', 'labevents': 'mimiciv_2_2_hosp',
    'microbiologyevents': 'mimiciv_2_2_hosp', 'poe': 'mimiciv_2_2_hosp', 'poe_detail': 'mimiciv_2_2_hosp',
    'prescriptions': 'mimiciv_2_2_hosp', 'procedures_icd': 'mimiciv_2_2_hosp', 'services': 'mimiciv_2_2_hosp',
    'd_icd_diagnoses': 'mimiciv_2_2_hosp', 'd_icd_procedures': 'mimiciv_2_2_hosp', 'd_hcpcs': 'mimiciv_2_2_hosp',
    'd_labitems': 'mimiciv_2_2_hosp',
    'procedureevents': 'mimiciv_2_2_icu', 'outputevents': 'mimiciv_2_2_icu', 'inputevents': 'mimiciv_2_2_icu',
    'icustays': 'mimiciv_2_2_icu', 'datetimeevents': 'mimiciv_2_2_icu', 'chartevents': 'mimiciv_2_2_icu',
    'd_items': 'mimiciv_2_2_icu',
}

# Same date-column lists load_mimiciv's preprocess_dataframe() uses for the local-csv path --
# duplicated here (rather than refactoring that nested function out) to keep this addition
# isolated from the already-working local-file code path.
_BIGQUERY_DATETIME_COLUMNS = {
    'admissions': ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'],
    'transfers': ['intime', 'outtime'],
    'prescriptions': ['starttime', 'stoptime'],
    'emar': ['charttime', 'scheduletime', 'storetime'],
    'microbiologyevents': ['chartdate', 'charttime', 'storedate'],
    'poe': ['ordertime'],
    'services': ['transfertime'],
    'procedureevents': ['starttime', 'endtime'],
    'outputevents': ['charttime'],
    'inputevents': ['starttime', 'endtime'],
    'icustays': ['intime', 'outtime'],
    'datetimeevents': ['charttime', 'storetime'],
    'chartevents': ['charttime'],
    'labevents': ['charttime'],
}
_BIGQUERY_ICD_STRIP_COLUMNS = {
    'diagnoses_icd': ['icd_code', 'icd_version'], 'd_icd_diagnoses': ['icd_code', 'icd_version'],
    'procedures_icd': ['icd_code', 'icd_version'], 'd_icd_procedures': ['icd_code', 'icd_version'],
    'hcpcsevents': ['hcpcs_cd'], 'd_hcpcs': ['code'],
}
# Columns where BigQuery's native column type doesn't match what the local-csv dtype_dicts in
# load_mimiciv() assume -- confirmed for prescriptions.ndc: BigQuery stores it as STRING
# (preserving leading zeros), but the local path reads it as float64 (the local csv.gz's ndc
# values already lost their leading zeros when the file was written, which is the exact quirk
# ndc_meds()'s to_str() padding-back logic exists to undo) -- so ndc_meds()'s
# med.ndc.astype("Int64") call expects a float/int-like column, not a string, and raises
# "cannot safely cast non-equivalent object to int64" on the BigQuery version unless coerced
# here first.
_BIGQUERY_NUMERIC_COERCE = {
    'prescriptions': ['ndc'],
}


def _get_bigquery_client():
    from google.cloud import bigquery
    return bigquery.Client(project=BIGQUERY_PROJECT)


def get_table_bigquery(table_name: str, cohort=None) -> pd.DataFrame:
    """
    Load a table from physionet-data's MIMIC-IV 2.2 BigQuery release instead of the local 3.1
    csv.gz files -- see MIMIC_DATA_SOURCE in scripts/constants.py for why. Mirrors
    load_mimiciv()'s local-file behavior closely enough to be a drop-in: pulls the full table
    (admissions/patients aren't filtered -- they define the cohort), applies the same
    cohort_ids.isin() filter by hadm_id for everything else, and the same datetime
    conversion / ICD code stripping load_mimiciv's preprocess_dataframe() does.

    Not used for labevents/chartevents -- see get_table_bigquery_large() for those (full-table
    pull would be hundreds of millions of rows; this does the cohort filter AND the
    one-row-per-(hadm_id,itemid) dedup server-side in SQL instead).
    """
    dataset = _BIGQUERY_TABLE_DATASET[table_name]
    client = _get_bigquery_client()
    query = f"SELECT * FROM `physionet-data.{dataset}.{table_name}`"
    print(f"[BigQuery 2.2] Querying {dataset}.{table_name}...")
    df = client.query(query).to_dataframe()

    if cohort is not None and table_name not in ('admissions', 'patients') and 'hadm_id' in df.columns:
        cohort_ids = cohort['hadm_id'].dropna().unique()
        df = df[df['hadm_id'].isin(cohort_ids)]

    for col in _BIGQUERY_DATETIME_COLUMNS.get(table_name, []):
        if col in df.columns:
            df = convert_to_datetime(df, col)
    for col in _BIGQUERY_ICD_STRIP_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in _BIGQUERY_NUMERIC_COERCE.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"[BigQuery 2.2] {table_name}: {len(df)} rows after cohort filter")
    return df


def get_table_bigquery_large(table_name: str, cohort: pd.DataFrame, max_attempts: int = 3) -> pd.DataFrame:
    """
    labevents/chartevents via BigQuery: uploads cohort's hadm_ids to a temp staging table in
    BIGQUERY_PROJECT, then JOINs against physionet-data's 2.2 table and does the same
    one-row-per-(hadm_id, itemid) dedup _LARGE_TABLE_SPECS' local-chunked-streaming path does
    (load_csv_filtered_chunked's dedup_subset) -- but server-side in SQL via QUALIFY, so we
    only ever download the already-filtered-and-deduped result instead of streaming through
    hundreds of millions of source rows client-side.

    Both the staging-table load job and the main query are given bounded timeouts: observed a
    hang where load_table_from_dataframe(...).result() never returned even though the job had
    already succeeded server-side in ~1s (confirmed via `bq ls -j`), so a plain blocking
    .result() with no timeout can silently wait forever. On a timeout we retry with a brand-new
    client/job rather than waiting indefinitely.
    """
    from google.cloud import bigquery
    import concurrent.futures

    spec = _LARGE_TABLE_SPECS[table_name]
    dataset = _BIGQUERY_TABLE_DATASET[table_name]
    cohort_ids = pd.DataFrame({'hadm_id': cohort['hadm_id'].dropna().unique().astype('int64')})
    staging_table = f"{BIGQUERY_PROJECT}.mipa_staging.cohort_hadm_ids_{table_name}"
    cols = ', '.join(spec['selected_columns'])
    query = f"""
        SELECT {cols} FROM (
            SELECT {cols}, ROW_NUMBER() OVER (
                PARTITION BY hadm_id, itemid ORDER BY charttime DESC
            ) AS rn
            FROM `physionet-data.{dataset}.{table_name}` t
            JOIN `{staging_table}` c USING (hadm_id)
            WHERE value IS NOT NULL
        )
        WHERE rn = 1
    """

    df = None
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        client = _get_bigquery_client()
        try:
            print(f"[BigQuery 2.2] Uploading {len(cohort_ids)} cohort hadm_ids to {staging_table}... (attempt {attempt}/{max_attempts})")
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            load_job = client.load_table_from_dataframe(cohort_ids, staging_table, job_config=job_config)
            load_job.result(timeout=180)

            print(f"[BigQuery 2.2] Streaming filtered+deduped {table_name} (this can take a while)...")
            query_job = client.query(query)
            query_job.result(timeout=900)
            df = query_job.to_dataframe()
            break
        except (concurrent.futures.TimeoutError, TimeoutError) as e:
            last_exc = e
            print(f"[BigQuery 2.2] Attempt {attempt}/{max_attempts} for {table_name} timed out client-side ({e}); retrying with a fresh client/job." if attempt < max_attempts else f"[BigQuery 2.2] All {max_attempts} attempts for {table_name} timed out client-side.")
    if df is None:
        raise RuntimeError(f"get_table_bigquery_large({table_name}) timed out after {max_attempts} attempts") from last_exc
    for col in spec.get('dtype_dict', {}):
        if col in df.columns:
            try:
                df[col] = df[col].astype(spec['dtype_dict'][col])
            except (ValueError, TypeError):
                pass  # e.g. NaNs in an int column -- leave as the BigQuery-inferred dtype
    for col in spec.get('datetime_columns', []):
        if col in df.columns:
            df = convert_to_datetime(df, col)

    print(f"[BigQuery 2.2] {table_name}: {len(df)} rows after cohort filter + dedup")
    return df


def get_table(table_name: str, cohort = None, compute = False) -> pd.DataFrame:
    """
    Load a table from MIMIC-IV and return it as a pandas dataframe, providing cohort to filter table according to
    cohort's hadm_ids.

    'labevents' and 'chartevents' are streamed in bounded chunks (see load_csv_filtered_chunked)
    instead of through Dask, since their compressed files are too large to safely materialize
    in memory whole -- see that function's docstring for why Dask can't help here either.

    Parameters:
    table_name (str): Name of the table to load
    cohort (pd.DataFrame): Pandas dataframe containing subject_id column

    Returns:
    pd.DataFrame: The requested table filtered by the cohort.

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

    if MIMIC_DATA_SOURCE == 'bigquery_2_2' and table_name not in ('dsnotes', 'radnotes'):
        # Notes are excluded here -- they're already MIMIC-IV-Note 2.2 locally regardless of
        # this setting (there's no 3.1-numbered note release to mismatch against), so there's
        # nothing to gain from re-sourcing them via BigQuery.
        if table_name in _LARGE_TABLE_SPECS:
            if cohort is None:
                raise ValueError(f"A cohort is required to stream large table: {table_name}")
            return get_table_bigquery_large(table_name, cohort)
        return get_table_bigquery(table_name, cohort)

    if table_name in _LARGE_TABLE_SPECS:
        if cohort is None:
            raise ValueError(f"A cohort is required to stream large table: {table_name}")
        spec = _LARGE_TABLE_SPECS[table_name]
        cohort_ids = cohort["hadm_id"].dropna().unique()
        print(f"Streaming {table_name} in chunks for {len(cohort_ids)} admissions")
        return load_csv_filtered_chunked(
            spec['path'], cohort_ids=cohort_ids, dtype_dict=spec['dtype_dict'],
            selected_columns=spec['selected_columns'], datetime_columns=spec['datetime_columns'],
            chunksize=250_000,
            dropna_subset=spec.get('dropna_subset'), dedup_subset=spec.get('dedup_subset'),
        )

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


def generate_pivot_sparse_path(csv_path):
    """
    Derive the sparse-pickle sibling path for a pivot file from its (legacy) .csv.gz path
    returned by generate_pivot_path(). See save_pivot()/load_pivot_sparse() for why pivot
    files need this second format.
    """
    if csv_path.endswith('.csv.gz'):
        return csv_path[: -len('.csv.gz')] + '.pkl'
    return csv_path + '.pkl'


def save_pivot(df, csv_path):
    """
    Save a pivot DataFrame, written by pivot_cohort(). If it has pandas Sparse-dtype columns
    (the normal case: pivot_cohort builds these directly from a scipy COO/CSR matrix instead
    of a dense pivot_table to avoid materializing a mostly-zero rows x columns array that can
    run into the tens of billions of cells for high-cardinality tables like DIAG), save as a
    pickle at the .pkl sibling path (see generate_pivot_sparse_path) instead of CSV: writing a
    sparse DataFrame out as CSV text still requires expanding every cell (including all the
    implicit zeros) into a string, which is the same cost class as densifying it in memory,
    defeating the entire point of staying sparse. Non-sparse pivots (e.g. from
    pivot_cohort_timeseries or the NLP CUI pivot, unchanged by this) still go through the
    original save_csv() path. Returns the path actually written to.
    """
    if any(isinstance(t, pd.SparseDtype) for t in df.dtypes):
        path = generate_pivot_sparse_path(csv_path)
        if not is_gcs_path(path):
            base_dir = os.path.dirname(path)
            if base_dir and not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
        df.to_pickle(path)
        return path
    save_csv(df, csv_path)
    return csv_path


def load_pivot_sparse(path, id_col_name='hadm_id', target_index=None):
    """
    Load a pivot file saved by save_pivot(), indexed by id_col_name, with numeric feature
    columns as pandas Sparse[float64]. Returns None if id_col_name is missing from the file
    (caller should treat this the same as the old "skip this df_type" branch).

    Checks for the .pkl sibling path (generate_pivot_sparse_path) first -- that's what
    save_pivot() writes for the normal case (pivot_cohort's sparse construction) and
    pd.read_pickle round-trips Sparse dtype exactly as saved, no conversion needed. Falls back
    to the plain .csv.gz path for pivots that were saved dense (pivot_cohort_timeseries/NLP
    CUI pivot, or pre-existing files from before this), converting their numeric columns to
    Sparse[float64] after reading.

    Pivot tables are mostly-zero wide matrices (DIAG alone can be tens of thousands of
    columns) -- load_and_filter_train/val (scripts/model/modeling.py) and
    load_and_filter_test (scripts/model/evaluate.py) each hold several of these
    simultaneously in `all_feature_dfs` before pd.concat(axis=1, join='outer'). Confirmed on
    pandas 2.2.3: axis=1 concat preserves SparseDtype columns -- BUT ONLY stays cheap if every
    frame already shares the same row index. Each df_type's pivot only covers the admissions
    that actually have that kind of event (e.g. CHART only covers ICU stays), so a plain
    outer-join concat has to introduce new rows for the ones it's missing -- and those show up
    as NaN, which is a different value from the SparseDtype's fill_value (0.0) and so can't be
    stored implicitly. Confirmed by measurement: concatenating a 5,000-row sparse frame into a
    50,000-row outer join ballooned it from 210MB back up to ~4,000MB even though dtype still
    reported as Sparse. Passing target_index here reindexes with fill_value=0 (matching the
    SparseDtype's own fill_value) BEFORE that concat ever happens, which keeps it at ~210MB --
    always pass the caller's already-known master/core hadm_id index.

    impute_features_data() is safe to call directly on the result: these columns never have
    NaN (pivot generation already fills with 0, and the target_index reindex above uses
    fill_value=0, not NaN), so the imputer naturally skips them. process_numerical_columns()
    is NOT safe as-is though -- its per-row divide-by-los/admissions_number step uses
    Series.div(), which silently densifies a Sparse column in this pandas version -- see
    sparse_series_div() for the fix applied there.
    """
    sparse_path = generate_pivot_sparse_path(path)
    if path_exists(sparse_path):
        df = pd.read_pickle(sparse_path)
        if id_col_name not in df.columns:
            return None
        df = df.set_index(id_col_name)
    else:
        df = read_gzip_or_parts(path)
        if id_col_name not in df.columns:
            return None
        df = df.set_index(id_col_name)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(pd.SparseDtype('float64', fill_value=0.0))
    if target_index is not None:
        df = reindex_sparse_df(df, target_index)
    return df


def reindex_sparse_df(df, target_index):
    """
    Reindex a DataFrame whose columns are all pandas Sparse dtype to target_index, filling
    new rows with each column's own fill_value (0.0) -- a fast, vectorized equivalent of
    df.reindex(target_index, fill_value=0).

    Plain DataFrame.reindex() reindexes a wide sparse DataFrame column by column internally,
    each one redoing an index alignment lookup -- confirmed by measurement to take over 750
    seconds for a real pivot table (PROC, 165,588 -> 504,763 rows, 8,591 columns) despite the
    result itself being small (the earlier-confirmed fill_value=0 reindex IS still necessary --
    see load_pivot_sparse's docstring for why a plain outer-join concat can't do this instead).
    This does the row-index alignment ONCE (a single pd.Index.get_indexer call) and applies it
    to the columns' combined data in one vectorized scipy.sparse operation instead -- confirmed
    equivalent output, and the same case went from ~789s to ~0.25s.
    """
    old_index = df.index
    n_rows, n_cols = len(df), len(df.columns)

    # Build one big COO matrix from every column's SparseArray directly (same technique
    # pivot_cohort uses to build these in the first place) rather than pd.DataFrame.sparse.
    # to_coo(), which this pandas version gates behind a newer scipy than this repo pins.
    rows, cols, data = [], [], []
    for j, col in enumerate(df.columns):
        arr = df[col].values
        positions = arr.sp_index.indices
        if len(positions):
            rows.append(positions)
            cols.append(np.full(len(positions), j))
            data.append(arr.sp_values)
    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    data = np.concatenate(data) if data else np.array([])
    csr = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    # indexer[i] = row position in `df` that should land at position i of target_index, or -1
    # if target_index[i] isn't in df.index at all. Append one all-zero row to the matrix and
    # point every -1 at it, so a single fancy-index select handles both cases at once.
    indexer = old_index.get_indexer(target_index)
    csr_padded = vstack([csr, csr_matrix((1, n_cols))]).tocsr()
    new_csr = csr_padded[np.where(indexer == -1, n_rows, indexer)]

    return pd.DataFrame.sparse.from_spmatrix(new_csr, index=target_index, columns=df.columns)


def sparse_series_div(series, divisor):
    """
    Divide a pandas Series with Sparse[float64, 0.0] dtype by a per-row divisor (array-like,
    aligned by position to `series`), without ever densifying.

    Series.div() silently converts a Sparse column to a regular dense float64 Series in
    pandas 2.2.3 (confirmed by measurement) -- for a wide pivot-derived column (hundreds of
    thousands of rows) that single call can require tens of gigabytes, which is exactly the
    per-column normalization process_numerical_columns does (dividing every DIAG/MEDS/PROC/
    CHART/LABS count column by 'los'/'admissions_number'). This instead operates directly on
    the underlying SparseArray's stored (non-fill) values: since fill_value is 0.0 and the
    divisor is never 0 here (caller replaces 0s with 1 first), 0 / divisor == 0 always, so the
    result can reuse the exact same sparse index and fill_value -- cost is O(stored values),
    not O(rows).
    """
    arr = series.values
    if not isinstance(arr, pd.arrays.SparseArray) or arr.fill_value != 0:
        raise ValueError("sparse_series_div requires a Series with Sparse dtype and fill_value == 0")
    divisor_values = np.asarray(divisor)
    positions = arr.sp_index.indices
    new_sp_values = arr.sp_values / divisor_values[positions]
    new_arr = pd.arrays.SparseArray(new_sp_values, sparse_index=arr.sp_index, fill_value=0.0, dtype='float64')
    return pd.Series(new_arr, index=series.index, name=series.name)


def sparse_df_div(df, sparse_columns, divisor):
    """
    Divide every column in sparse_columns (all pandas Sparse[float64, 0.0] dtype, same index
    as df) by a per-row divisor (array-like, aligned by position to df.index), in one
    vectorized pass instead of calling sparse_series_div() once per column.

    Calling sparse_series_div() in a Python for-loop over thousands of pivot-derived columns
    is correct but still has per-call overhead that adds up -- confirmed by measurement at
    ~197 seconds for 20,990 columns. This instead gathers every column's stored (row, value)
    pairs into one combined array, divides all of them by their row's divisor in a single
    numpy operation, and rebuilds the sparse columns from that -- same technique as
    reindex_sparse_df. Returns a new DataFrame with sparse_columns replaced; other columns in
    df are left untouched.
    """
    if not sparse_columns:
        return df
    n_rows = len(df)
    divisor_values = np.asarray(divisor)
    rows, cols, data = [], [], []
    for j, col in enumerate(sparse_columns):
        arr = df[col].values
        positions = arr.sp_index.indices
        if len(positions):
            rows.append(positions)
            cols.append(np.full(len(positions), j))
            data.append(arr.sp_values)
    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    cols_arr = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    data = np.concatenate(data) if data else np.array([])
    new_data = data / divisor_values[rows]
    csr = coo_matrix((new_data, (rows, cols_arr)), shape=(n_rows, len(sparse_columns))).tocsr()
    result = pd.DataFrame.sparse.from_spmatrix(csr, index=df.index, columns=sparse_columns)

    df = df.copy()
    df[sparse_columns] = result
    return df


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

