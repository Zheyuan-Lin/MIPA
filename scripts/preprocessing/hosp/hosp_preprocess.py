# Filename: hosp+preprocess.py
# Description: This file contains functions and utilities for preprocessing data from
#              the hospital database.
#              before it is used in analysis or modeling. It includes methods for
#              cleaning data, handling missing values, encoding categorical variables,
#              and other necessary preprocessing steps to prepare the dataset for
#              subsequent analysis or machine learning modeling. The functions within
#              are designed to be modular and reusable across different data processing
#              workflows.

# Import necessary libraries
import numpy as np
import os
import re
from tqdm import tqdm
from scripts.constants import *
from scripts.utils import *
from scripts.preprocessing.hosp.labs_preprocess_util import *
from scripts.subtask.uom_conversion import *
from scripts.subtask.outlier_removal import *


########################## DIAGNOSES ##########################
import numpy as np
import pandas as pd
import dask.dataframe as dd

def standardize_icd(mapping: pd.DataFrame, cohort: pd.DataFrame, root_icd9: bool = False, group_icd10: bool = True) -> pd.DataFrame:
    """
    Standardizes ICD codes from ICD9 to ICD10 using a mapping table.

    Parameters:
    - mapping: A DataFrame containing the ICD9 to ICD10 mapping.
    - cohort: The cohort DataFrame with ICD codes to be standardized.
    - root_icd9: If True, only map an ICD9 code to ICD10 based on the ICD9's root (first 3 digits).
    - group_icd10: If True, groups ICD10 codes based on the first 3 digits of the ICD10.

    Returns:
    - The cohort DataFrame with added columns for the standardized ICD codes.
    """
    def icd_9to10(icd: str) -> str:
        # Truncate ICD9 code to its root if root_icd9 is True
        if root_icd9:
            icd = icd[:3]
        try:
            # Attempt to map ICD9 code to ICD10
            return mapping.loc[mapping['diagnosis_code'] == icd].icd10cm.iloc[0]
        except IndexError:
            # Return NaN if mapping not found
            return np.nan

    # Determine column name for the standardized ICD10 codes
    icd_map_colname = 'icd10_convert'
    if root_icd9:
        icd_map_colname = 'icd9root_' + icd_map_colname

    # Apply mapping function to the cohort DataFrame
    cohort[icd_map_colname] = cohort['icd_code'].apply(icd_9to10)

    # Group ICD10 codes based on the first 3 digits if group_icd10 is True
    if group_icd10:
        cohort['root'] = cohort.apply(
                lambda row: row[icd_map_colname][:3] if pd.notnull(row[icd_map_colname]) else (
                        row['icd_code'][:3] if pd.notnull(row['icd_code']) and row['icd_version'] == '10' else np.nan),
                axis=1
        )
    return cohort


def preproc_diagnoses(cohort: pd.DataFrame, icd_map_path: str = None, root_icd9: bool = True, group_icd10: bool = True, options: str = 'ICD10') -> pd.DataFrame:
    """
    Processes a cohort dataset with ICD codes, mapping them to ICD-10 codes using a mapping table.
    Removing all codes for which mapping is not found.

    Parameters:
    - cohort: The cohort DataFrame.
    - icd_map_path: The path to the ICD mapping table.
    - root_icd9: If True, maps ICD-9 to ICD-10 using only the first three characters of the ICD9 code for a better fuzzy match.
    - group_icd10: If True, groups ICD10 codes based on the first 3 digits of the ICD10.
    - options: Determines the output format. Can be 'ICD10', 'ICD10_grouped', or 'Original'.

    Returns:
    - The processed cohort DataFrame.
    """
    def get_cohort_diagnosis(cohort: pd.DataFrame) -> pd.DataFrame:
        # Retrieve diagnoses_icd and d_icd_diagnoses tables
        df_diagnoses_icd = get_table('diagnoses_icd', cohort)
        df_diagnoses_icd_dict = get_table('d_icd_diagnoses')
        # Merge and clean up the DataFrame
        result_df = df_diagnoses_icd.merge(df_diagnoses_icd_dict, on='icd_code', how='inner', suffixes=('', '_drop'))
        result_df = result_df.drop(columns=['seq_num', 'icd_version_drop'])
        return result_df

    cohort_df = get_cohort_diagnosis(cohort)

    if icd_map_path:
        # Load the ICD mapping table
        icd_map = read_icd_mapping(icd_map_path)
        # Standardize ICD codes
        cohort_df = standardize_icd(icd_map, cohort_df, root_icd9, group_icd10)

    # Validate options parameter
    if options not in ['ICD10', 'ICD10_grouped', 'Original']:
        raise ValueError("options must be either 'ICD10', 'ICD10_grouped', or 'Original'")

    # Logging for unique ICD codes before and after conversion
    print("# Admissions:", cohort_df.hadm_id.nunique())
    print("# unique ICD-9 codes:", cohort_df[cohort_df['icd_version'].astype(str) == '9']['icd_code'].nunique())
    print("# unique ICD-10 codes:", cohort_df[cohort_df['icd_version'].astype(str) == '10']['icd_code'].nunique())
    if root_icd9:
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10):", cohort_df['icd9root_icd10_convert'].nunique())
    else:
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10 and grouping ICD-9 codes):", cohort_df[
            'icd10_convert'].nunique())
    if group_icd10:
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10 and grouping ICD-10 codes):",
              cohort_df['root'].nunique())



    # Determine the 'new_icd_code' column based on options
    if options == 'Original':
        cohort_df['new_icd_code'] = cohort_df['icd_code'] # Returns original ICD codes
    elif options == 'ICD10':
        if root_icd9:
            cohort_df['new_icd_code'] = cohort_df.apply(
                    lambda row: row['icd9root_icd10_convert'] if pd.notna(row['icd9root_icd10_convert'])
                    else (row['icd_code'] if row['icd_code'] == '10' else None),
                    axis=1
            )
        else:
            cohort_df['new_icd_code'] = cohort_df.apply(
                    lambda row: row['icd10_convert'] if pd.notna(row['icd10_convert'])
                    else (row['icd_code'] if row['icd_code'] == '10' else None),
                    axis=1
            )
    elif options == 'ICD10_grouped':
        cohort_df['new_icd_code'] = cohort_df['root']

    # Count the number of na new_icd_code and print to user
    null_new_icd_code_df = cohort_df[cohort_df['new_icd_code'].isna()]
    count_null_new_icd_code = null_new_icd_code_df['icd_code'].nunique()

    print("Number of non transformed ICD codes that will be ignored: ", {count_null_new_icd_code})

    # Drop rows where 'new_icd_code' is NaN
    cohort_df = cohort_df.drop(columns=['icd_code'])

    # Final renaming
    cohort_df = cohort_df.dropna(subset=['new_icd_code'])
    cohort_df = cohort_df.rename(columns={'new_icd_code': 'icd_code'})

    return cohort_df[['subject_id', 'hadm_id', 'icd_code', 'long_title']]


########################## LABS ##########################
def read_labevents_table(cohort: pd.DataFrame) -> pd.DataFrame:
    df_labevents = get_table('labevents', cohort)
    return df_labevents[
        ['subject_id', 'itemid', 'hadm_id', 'charttime','value', 'valuenum','valueuom']]

def read_d_labitems_table():
    df_d_labitems = get_table('d_labitems')
    return df_d_labitems[['itemid', 'label', 'category']]


def read_labs(cohort: pd.DataFrame) -> dd.DataFrame:
    """
    Reads lab events and lab items tables and merges them.
    This function assumes that the tables are read as Dask DataFrames for efficient processing.
    """
    # Assuming read_labevents_table and read_d_labitems_table return Dask DataFrames
    labevents = read_labevents_table(cohort)
    labitems = read_d_labitems_table()

    # Merge the DataFrames
    merged_df = labevents.merge(labitems, how='inner', left_on=['itemid'], right_on=['itemid'])
    return merged_df

def preproc_labs(cohort: pd.DataFrame, time = False) -> pd.DataFrame:
    """
    Optimized function for getting hospital observations pertaining to a cohort using Dask for parallel processing.
    This function is structured to save memory when reading and transforming data.
    """
    # Read and preprocess labs data using Dask
    labs_df = read_labs(cohort)
    print(f'Initial Labs dataframe shape: {labs_df.shape}')

    # Drop rows with NaN values in 'value'
    labs_df = labs_df.dropna(subset=['value'])
    print(f'Labs dataframe shape after dropping NaN values in "value": {labs_df.shape}')

    # Fill NaN in 'valueuom' with 0
    labs_df['valueuom'] = labs_df['valueuom'].fillna(0)

    # Merge with cohort data to filter by admissions
    labs_df = labs_df.merge(cohort[['hadm_id', 'admittime', 'dischtime']], how='inner', left_on='hadm_id',
                            right_on='hadm_id')

    # Check for missing 'hadm_id' and impute if necessary
    missing_hadm_ids = labs_df[labs_df['hadm_id'].isna()]
    missing_hadm_ids = ensure_pandas_df(missing_hadm_ids)
    if missing_hadm_ids.shape[0] > 0:
        print(f'Missing HADM ids to be imputed: {missing_hadm_ids.shape}')
        imputed_hadm_ids = impute_hadm_ids(missing_hadm_ids[['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom']], cohort)
        imputed_hadm_ids = imputed_hadm_ids.rename(columns={'hadm_id_new': 'hadm_id'})
        labs_df = labs_df.dropna(subset=['hadm_id']) # Drop rows with NaN 'hadm_id' after imputation
        labs_df = labs_df.append(imputed_hadm_ids, ignore_index=True)
        print(f'Imputed HADM ids: {imputed_hadm_ids.shape}')
    else:
        print("No missing HADM ids found. Skipping imputation.")



    # Convert 'charttime' to datetime and calculate time from admission
    if isinstance(labs_df, dd.DataFrame):
        labs_df['charttime'] = dd.to_datetime(labs_df['charttime'])
        labs_df['lab_time_from_admit'] = labs_df['charttime'] - labs_df['admittime']
        labs_df = labs_df[labs_df['lab_time_from_admit'] >= pd.Timedelta(0)]

    elif isinstance(labs_df, pd.DataFrame):
        labs_df['charttime'] = pd.to_datetime(labs_df['charttime'])
        labs_df['lab_time_from_admit'] = labs_df['charttime'] - labs_df['admittime']
        labs_df = labs_df[labs_df['lab_time_from_admit'] >= pd.Timedelta(0)]


    # Remove labs for which delta is negative
    labs_df = labs_df.dropna() # Drop rows with NaN values after datetime conversion

    # Logging for unique item IDs, admissions, and total rows
    print("# Itemid: ", labs_df.itemid.nunique())
    print("# Admissions: ", labs_df.hadm_id.nunique())
    print("Total number of rows: ", labs_df.shape[0])


    # Time series correction
    if time == False:
        labs_df = labs_df.drop_duplicates(subset=['hadm_id', 'item_id'])
        labs_df = labs_df.drop(columns=['lab_time_from_admit'])
    if time == True:
        pass

    labs_df = labs_df.drop(columns=['category','charttime', 'admittime', 'dischtime'])
    # Convert Dask DataFrame back to Pandas DataFrame for final output
    return labs_df


########################## MEDICATIONS ##########################
def read_prescriptions_table(cohort: pd.DataFrame) -> pd.DataFrame:
    df_prescriptions = get_table('prescriptions', cohort)
    df_prescriptions = df_prescriptions.reset_index()
    return df_prescriptions[['subject_id', 'hadm_id', 'starttime', 'stoptime', 'ndc', 'gsn', 'drug',
                 'drug_type']]

def get_generic_drugs(mapping, df):
    """Takes NDC product table and prescriptions dataframe; adds column with NDC
    table's corresponding generic name"""

    def brand_to_generic(ndc):
        # We only want the first 2 sections of the NDC code: xxxx-xxxx-xx
        matches = list(re.finditer(r"-", ndc))
        if len(matches) > 1:
            ndc = ndc[:matches[1].start()]
        try:
            return mapping.loc[mapping.PRODUCTNDC == ndc].NONPROPRIETARYNAME.iloc[0]
        except:
            print("Error: ", ndc)
            return np.nan

    df['generic_drug_name'] = df['ndc'].apply(brand_to_generic)

def ndc_meds(med, mapping = MAP_PATH+'ndc_product.txt') -> pd.DataFrame:
    # Convert any nan values to a dummy value
    mapping = mapping
    med.ndc = med.ndc.fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med.ndc = med.ndc.astype("Int64")

    # The NDC codes in the prescription dataset is the 11-digit NDC code, although
    # codes are missing
    # their leading 0's because the column was interpreted as a float then integer;
    # this function restores
    # the leading 0's, then obtains only the PRODUCT and MANUFACTURER parts of the NDC
    # code (first 9 digits)
    def to_str(ndc):
        if ndc < 0:  # dummy values are < 0
            return np.nan
        ndc = str(ndc)
        return (("0" * (11 - len(ndc))) + ndc)[0:-2]

    # The mapping table is ALSO incorrectly formatted for 11 digit NDC codes. An 11
    # digit NDC is in the
    # form of xxxxx-xxxx-xx for manufactuerer-product-dosage. The hyphens are in the
    # correct spots, but
    # the number of digits within each section may not be 5-4-2, in which case we add
    # leading 0's to each
    # to restore the 11 digit format. However, we only take the 5-4 sections, just like
    # the to_str function
    def format_ndc_table(ndc):
        parts = ndc.split("-")
        return ("0" * (5 - len(parts[0])) + parts[0]) + (
                    "0" * (4 - len(parts[1])) + parts[1])


    def read_ndc_mapping(map_path):
        ndc_map = pd.read_csv(map_path, header=0, delimiter='\t', encoding='latin1')
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
        ndc_map.columns = list(map(str.lower, ndc_map.columns))
        return ndc_map

    # Read in NDC mapping table
    ndc_map = read_ndc_mapping(mapping)[
        ['productndc', 'nonproprietaryname', 'pharm_classes']]

    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map['new_ndc'] = ndc_map.productndc.apply(format_ndc_table)
    ndc_map.drop_duplicates(subset=['new_ndc', 'nonproprietaryname'], inplace=True)
    med['new_ndc'] = med.ndc.apply(to_str)

    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how='inner', left_on='new_ndc', right_on='new_ndc')

    # In NDC mapping table, the pharm_class col is structured as a text string,
    # separating different pharm classes from eachother
    # This can be [PE], [EPC], and others, but we're interested in EPC. Luckily,
    # between each commas, it states if a phrase is [EPC]
    # So, we just string split by commas and keep phrases containing "[EPC]"
    def get_EPC(s):
        """Gets the Established Pharmacologic Class (EPC) from the mapping table"""
        if type(s) != str:
            return np.nan
        words = s.split(",")
        return [x for x in words if "[EPC]" in x]

    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med['EPC'] = med.pharm_classes.apply(get_EPC)

    return med

def preproc_meds(cohort: pd.DataFrame, time = False, group = True) -> pd.DataFrame:
    adm = cohort[['hadm_id', 'admittime', 'dischtime']]
    med = read_prescriptions_table(adm)
    med = med.merge(adm, left_on='hadm_id', right_on='hadm_id', how='inner')
    med['start_hours_from_admit'] = med['starttime'] - med['admittime']
    med['stop_hours_from_admit'] = med['stoptime'] - med['admittime']

    # Normalize drug strings and remove potential duplicates
    med.drug = med.drug.fillna("").astype(str)
    med.drug = med.drug.apply(
        lambda x: x.lower().strip().replace(" ", "_") if not "" else "")
    med.drug = med.drug.dropna().apply(lambda x: x.lower().strip())

    # meds.to_csv(output_path, compression='gzip', index=False)
    med = ndc_meds(med)

    print("Number of unique type of drug: ", med.drug.nunique())
    print("Number of unique type of drug (after grouping to use Non propietary names): ",
          med.nonproprietaryname.nunique())
    print("Number of unique type of drug classes: ",
          med.pharm_classes.nunique())
    print("Total number of rows: ", med.shape[0])
    print("# Admissions:  ", med.hadm_id.nunique())

    columns_to_remove = ['gsn', 'ndc', 'drug_type', 'admittime', 'dischtime', 'stoptime', 'starttime', 'productndc',
                         'pharm_classes']
    med = med.drop(columns=columns_to_remove, errors='ignore')
    med = med.rename(columns={'new_ndc': 'ndc'})

    # Remove duplicates
    med = med.drop_duplicates(subset=['hadm_id', 'ndc', 'start_hours_from_admit', 'stop_hours_from_admit'])
    # Drop rows with negative delta
    med = med[med['start_hours_from_admit'] >= pd.Timedelta(0)]
    med = med[med['stop_hours_from_admit'] >= pd.Timedelta(0)]

    # Removing columns
    columns_to_remove = ['gsn', 'drug_type', 'admittime', 'dischtime', 'stoptime', 'starttime', 'productndc',
                         'pharm_classes']
    med = med.drop(columns=columns_to_remove, errors='ignore')

    # Time series correction
    # Returns data with unique rows for each drug, with start and stop times; timeseries manipulation in another function
    if time:
        # Keep each drug with unique start and stop times
        if group:
            med = med.drop_duplicates(subset=['hadm_id', 'nonproprietaryname', 'start_hours_from_admit', 'stop_hours_from_admit'])  # Alternative is group by EPC
        else:
            med = med.drop_duplicates(subset=['hadm_id', 'drug', 'start_hours_from_admit', 'stop_hours_from_admit'])
        pass
    if not time:
        # Remove drug with different start and stop times
        if group:
            med = med.drop_duplicates(subset=['hadm_id', 'nonproprietaryname']) #Alternative is group by EPC for
            # less granular filtration
        else:
            med = med.drop_duplicates(subset=['hadm_id', 'drug'])
        med = med.drop(columns=['start_hours_from_admit', 'stop_hours_from_admit'])
    return med


########################## PROCEDURES ##########################
def read_procedures_table(cohort: pd.DataFrame) -> pd.DataFrame:
    df_procedures_icd = get_table('procedures_icd', cohort)
    df_procedures_icd.reset_index(inplace=True)
    return df_procedures_icd


def read_d_icd_procedures_table() -> pd.DataFrame:
    df_d_icd_procedures = get_table('d_icd_procedures')
    df_d_icd_procedures.reset_index(inplace=True)
    return df_d_icd_procedures[['icd_code', 'long_title']]


def read_procedures(cohort: pd.DataFrame) -> pd.DataFrame:
    df_procedures_icd = read_procedures_table(cohort)
    df_d_icd_procedures = read_d_icd_procedures_table()
    return df_procedures_icd.merge(
            df_d_icd_procedures, how='inner', left_on=['icd_code'],
            right_on=['icd_code']
    )


def preproc_proc(cohort, time=False) -> pd.DataFrame:
    """Function for getting procedures ICD codes for cohort
     No mapping done for procedures ICD codes
    """

    adm = cohort[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'anchor_year']]
    proc = read_procedures(adm)

    df_cohort = proc.merge(adm[['hadm_id', 'admittime', 'dischtime']], how='inner',
                 left_on='hadm_id', right_on='hadm_id')

    df_cohort['chartdate'] = pd.to_datetime(df_cohort['chartdate'])
    df_cohort['admittime'] = pd.to_datetime(df_cohort['admittime'])
    df_cohort['proc_time_from_admit'] = df_cohort['chartdate'] - df_cohort['admittime']

    # Drop rows with negative delta
    df_cohort = df_cohort[df_cohort['proc_time_from_admit'] >= pd.Timedelta(0)]
    df_cohort = df_cohort.dropna()


    # Print unique counts and value_counts
    print("# Unique ICD9 Procedures:  ",
          df_cohort.loc[df_cohort.icd_version == 9].icd_code.dropna().nunique())
    print("# Unique ICD10 Procedures: ",
          df_cohort.loc[df_cohort.icd_version == 10].icd_code.dropna().nunique())
    print("\nValue counts of each ICD version:\n", df_cohort.icd_version.value_counts())
    print("# Admissions:  ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    if time == False:
        df_cohort = df_cohort.drop_duplicates(subset=['hadm_id', 'icd_code'])
        df_cohort = df_cohort.drop(columns=['chartdate', 'admittime', 'dischtime', 'proc_time_from_admit'])
    if time == True:
        pass

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort.drop(columns=['index', 'seq_num'])


########################## RADIOLOGY NOTES ##########################

def read_radiology_notes(cohort_df) -> pd.DataFrame:
    df_radnotes= get_table('radnotes', cohort=cohort_df)
    df_radnotes.reset_index()
    return df_radnotes[['hadm_id', 'charttime', 'text', 'note_type']]


def preproc_radnotes(cohort) -> dd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function
    is structured to save memory when reading and transforming data."""

    # Extract necessary columns
    adm = cohort[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
    radnotes = read_radiology_notes(adm)

    # Merge with checks for alignment on 'hadm_id'
    df_cohort = radnotes.merge(adm, how='inner', on='hadm_id')

    # Convert 'charttime' and 'admittime' to datetime
    if isinstance(df_cohort, dd.DataFrame):
        df_cohort['charttime'] = dd.to_datetime(df_cohort['charttime'])
        df_cohort['admittime'] = dd.to_datetime(df_cohort['admittime'])

    elif isinstance(df_cohort, pd.DataFrame):
        df_cohort['charttime'] = pd.to_datetime(df_cohort['charttime'])
        df_cohort['admittime'] = pd.to_datetime(df_cohort['admittime'])




    # Calculate time from admission
    df_cohort['rad_time_from_admit'] = df_cohort['charttime'] - df_cohort['admittime']

    # Drop rows with negative delta
    df_cohort = df_cohort[df_cohort['rad_time_from_admit'] >= pd.Timedelta(0)]

    # Drop rows with no significant text
    df_cohort = df_cohort[df_cohort['text'].str.len() >= 200]

    # Drop rows with any NaN values in critical columns
    df_cohort = df_cohort.dropna(subset=['text','charttime', 'admittime', 'rad_time_from_admit'])
    # filter for note_type == 'RR'
    df_cohort = df_cohort[df_cohort['note_type'] == 'RR']

    # Add a column for the number of rad_note per hadm_id
    df_cohort['radnote_count'] = df_cohort.groupby('hadm_id')['text'].transform('count')

    # Since Dask does not execute until compute(), we prepare the data and return the Dask DataFrame
    return df_cohort[['subject_id','hadm_id', 'text', 'rad_time_from_admit']]

########################## DISCHARGE SUMMARY NOTES ##########################

def read_ds_notes(cohort_df) -> pd.DataFrame:
    df_dsnotes= get_table('dsnotes', cohort=cohort_df)
    df_dsnotes.reset_index()
    return df_dsnotes[['hadm_id', 'charttime', 'text']]

def read_ds_notes(cohort_df) -> dd.DataFrame:
    df_dsnotes = get_table('dsnotes', cohort=cohort_df)  # Assuming get_table returns a Dask DataFrame
    df_dsnotes = df_dsnotes.reset_index()  # Reset index in Dask
    return df_dsnotes[['subject_id', 'hadm_id', 'charttime', 'text']]

def preproc_dsnotes(cohort) -> dd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function
    is structured to save memory when reading and transforming data."""

    # Extract necessary columns
    adm = cohort[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'anchor_year']]
    dsnotes = read_ds_notes(cohort)

    # Merge with checks for alignment on 'hadm_id'
    df_cohort = dsnotes.merge(adm[['hadm_id', 'admittime', 'dischtime']], how='inner', on='hadm_id')
    df_cohort = df_cohort.drop_duplicates(subset=['hadm_id'])

    # Convert 'charttime' and 'admittime' to datetime
    if isinstance(df_cohort, dd.DataFrame):
        df_cohort['charttime'] = dd.to_datetime(df_cohort['charttime'])
        df_cohort['admittime'] = dd.to_datetime(df_cohort['admittime'])

    elif isinstance(df_cohort, pd.DataFrame):
        df_cohort['charttime'] = dd.to_datetime(df_cohort['charttime'])
        df_cohort['admittime'] = pd.to_datetime(df_cohort['admittime'])

    # Calculate time from admission
    df_cohort['ds_time_from_admit'] = df_cohort['charttime'] - df_cohort['admittime']

    # Drop rows with negative delta
    df_cohort = df_cohort[df_cohort['ds_time_from_admit'] >= pd.Timedelta(0)]

    # Filtering based on length
    df_cohort = df_cohort[df_cohort['text'].str.len() >= 300]

    # Add a column for number of notes per hadm_id
    df_cohort['dsnote_count'] = df_cohort.groupby('hadm_id')['text'].transform('count')

    # Drop rows with any NaN values in critical columns
    df_cohort = df_cohort.dropna()

    # Since Dask does not execute until compute(), we prepare the data and return the Dask DataFrame
    return df_cohort[['subject_id','hadm_id', 'text', 'ds_time_from_admit']]


########################## ICU CHART EVENTS ##########################
def read_icu_chartevents_dict() -> pd.DataFrame:
    df_icu_chartevents_dict= get_table('d_items')
    df_icu_chartevents_dict.reset_index()
    return df_icu_chartevents_dict[['itemid', 'label']]

def read_icu_chartevents(cohort_df) -> pd.DataFrame:
    df_chartevents= get_table('chartevents', cohort=cohort_df)
    df_chartevents.reset_index()
    return df_chartevents[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom']]

def read_chartevents(cohort_df) -> pd.DataFrame:
    """
    Reads the ICU chartevents data and merges it with the dictionary of items.

    Parameters:
    None

    Returns:
    pd.DataFrame: A DataFrame containing the merged chartevents and dictionary of items data.

    Raises:
    None

    Warns:
    None

    Example:
    >>> df_chartevents = read_chartevents()
    >>> print(df_chartevents.head())
    """
    dict_df = read_icu_chartevents_dict()
    chart_df = read_icu_chartevents(cohort_df)

    chartevents_df = chart_df.merge(dict_df, how='left', left_on='itemid', right_on='itemid')
    chartevents_df.reset_index()

    return chartevents_df


def preproc_chart(cohort, time=False) -> dd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    df_chartevents = read_chartevents(cohort)  # Assuming this returns a Dask DataFrame

    # Ensure 'valuenum' is not NaN
    df_chartevents = df_chartevents.dropna(subset=['value'])

    # Merge with cohort data
    cohort_data = cohort[['stay_id', 'intime']]
    df_chartevents = df_chartevents.merge(cohort_data, how='inner', on='stay_id')

    # Calculate event time from admission
    time_col = 'charttime'  # Assuming 'charttime' is the column with event timestamps
    df_chartevents['event_time_from_admit'] = df_chartevents[time_col] - df_chartevents['intime']

    # Drop rows with negative delta
    df_chartevents = df_chartevents[df_chartevents['event_time_from_admit'] >= pd.Timedelta(0)]

    # Drop unnecessary columns
    df_chartevents = df_chartevents.drop(columns=[time_col, 'intime'])

    # Drop duplicates and any remaining NaN values
    df_chartevents = df_chartevents.dropna().drop_duplicates()

    # Optionally, trigger a single computation here if the dataset is manageable
    # df_chartevents = df_chartevents.compute()  # Uncomment if manageable and necessary

    # Instead of computing here, return the Dask DataFrame to be computed later++
    #df_cohort_computed = df_chartevents.compute()  # Compute once
    #print("# Unique Events:  ", df_cohort_computed['itemid'].nunique())
    #print("# Admissions:  ", df_cohort_computed['stay_id'].nunique())
    #print("Total rows", df_cohort_computed.shape[0])
    if time == False:
        df_chartevents = df_chartevents.drop_duplicates(subset=['hadm_id', 'itemid'])
        df_chartevents = df_chartevents.drop(columns=['event_time_from_admit'])
    if time == True:
        pass

    return df_chartevents



########################## MAPPING ##########################
def read_icd_mapping(map_path=MAP_PATH):
    mapping = pd.read_csv(map_path, header=0, delimiter='\t')
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def read_ndc_mapping(map_path=MAP_PATH):
    ndc_map = pd.read_csv(map_path, header=0, delimiter='\t')
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
    ndc_map.columns = list(map(str.lower, ndc_map.columns))
    return ndc_map


########################## PREPROCESSING ##########################
def get_range(df: pd.DataFrame, time_col: str, anchor_col: str,
              measure='days') -> pd.Series:
    """Uses array arithmetic to find the ranges an observation time could be in based
    on the patient's anchor info"""

    def get_year_timedelta(group: tuple) -> str:
        """Applied to a Series of tuples in the form (start year, end year) that
        represent the range of years a labevent could occur according to a patient's
        anchor info"""
        # Recall that we only want info in a 3 year time range anywhere between 2008
        # and 2019.
        # All possible ranges are:  ['2008 - 2010', '2009 - 2011', '2010 - 2012',
        # '2011 - 2013',
        # '2012 - 2014', '2013 - 2015','2014 - 2016',  '2015 - 2017', '2016 - 2018',
        # '2017 - 2019']
        if group[0] >= 2008 and group[0] <= 2017 and group[1] >= 2010 and group[
            1] <= 2019:
            return group[0] - 2008
        else:
            return np.nan

    if measure == 'years':
        # Array arithmetic to obtain a tuple of (start year, end year) for each labevent
        # After getting list of tuples, apply get_timedelta
        shift = (df[time_col] - df[anchor_col])
        return pd.Series(
            list(zip(df.min_year_group + shift, df.max_year_group + shift))).apply(
            get_year_timedelta)
    elif measure == 'days':
        # Convert base_year into a datetime and find the difference in days between the
        # time_col and the beginning of the base_year
        base_dt = df[anchor_col].apply(lambda x: dt.datetime(year=x, month=1, day=1))
        return (df[time_col] - base_dt).dt.days
    else:
        raise Exception('\'measure\' argument must be either \'years\' or \'days\'.')
