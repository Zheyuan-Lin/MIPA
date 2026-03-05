import pandas as pd
import warnings

from scripts.utils import load_csv, load_json
from scripts.constants import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def phenotype_list():
 return sorted(["c_diff_inpatient", "c_diff_past", "dementia", "depression",
 "diabetes_type_1",
         "diabetes_type_2",
         "hfref", "hfpef", "hypertension", "sle", "metastatic_cancer", "obesity",
         "rheumatoid_arthritis", "roh_abuse", "vte_inpatient", "vte_past"])

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
            'HISPANIC OR LATINO' if x == 'HISPANIC' else x,
            meta=('race', 'object')
    )
    #### Replace '?' in language column with null
    df_core['language'] = df_core['language'].replace('?', None)

    ### Feature selection
    # Sanity check
    ### Remove duplicates hadm_id
    df_core = df_core.drop_duplicates(subset=['hadm_id']) # Ensure compute() is called when the DataFrame is ready
    ### Remove patients with age less than 18 or greater than 89
    df_core = df_core[(df_core['age'] > 18) & (df_core['age'] <= 89)]

    # NULL values not handled
    return df_core.compute()
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


def phenotype_cohort_train(df_core, phenotype, silver_labels=False):
    """
    Filters the core dataframe to include only patients with the specified phenotype.
    If silver_labels is True, returns cohorts of patients with a prespecified list of ICD codes.

    :param df_core: pd.DataFrame - Core patient data.
    :param phenotype: str - The target phenotype.
    :param silver_labels: bool - If True, use silver labels for noisy labeling.
    :return: pd.DataFrame - Filtered and labeled patient data.
    """
    phenotypes = ["c_diff", "dementia", "depression", "diabetes_type_1",
                  "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
                  "metastatic_cancer", "obesity", "rheumatoid_arthritis", "roh_abuse", "vte"]
    if phenotype not in phenotypes:
        raise ValueError(f'Phenotype not in list of phenotypes: {phenotypes}')

    # Exclude test subjects
    to_exclude = pd.read_csv(FILTERS_PATH + 'test_subject_ids.csv')
    df_core = df_core.merge(to_exclude, on='subject_id', how='left', indicator=True)
    df_core = df_core[df_core['_merge'] == 'left_only'].drop(columns=['_merge'])

    if silver_labels:
        phenotypes_silver_labels = load_json(FILTERS_PATH + 'silver_labels.json')
        phenotype_icd_codes = phenotypes_silver_labels[phenotype]

        # Create a DataFrame for phenotype ICD codes for merging
        df_phenotype_codes = pd.DataFrame(phenotype_icd_codes, columns=['icd_code'])
        df_phenotype_codes['y'] = 1

        # Merge to assign 'y' labels
        df_core = df_core.merge(df_phenotype_codes, on='icd_code', how='left')
        df_core['y'].fillna(0, inplace=True)

        # Filter positive and negative samples
        df_positive = df_core[df_core['y'] == 1]
        df_negative = df_core[df_core['y'] == 0]

        # Sample negative hadm_ids
        negative_hadm_ids = df_negative['hadm_id'].drop_duplicates().sample(n=len(df_positive['hadm_id'].unique()),
                                                                            random_state=42)
        df_negative_sampled = df_negative[df_negative['hadm_id'].isin(negative_hadm_ids)]

        # Concatenate positive and sampled negative dataframes
        df_core = pd.concat([df_positive, df_negative_sampled])

    else:
        print(f"You extracted the phenotype {phenotype} without using silver labels. The entire cohort was returned "
              f"excluding the test set. You can define noisy labels for training based on your own criteria.")

    # Return hadm_ids only
    df_core = df_core.drop_duplicates(subset=['subject_id', 'hadm_id'])
    columns_to_remove = ['admission_location', 'insurance', 'icd_code', 'icd_version']
    df_core = df_core.drop(columns=columns_to_remove, errors='ignore')

    return df_core

def phenotype_cohort_test(df_core, phenotype, split=0.5):
    """
    For given pheneotype, provide a test and validation set
    Essentially return 30/70 ratio of labeled dataset based on subject_id
    Test set for gold label training, validation set for measure of performance
    """
    # TODO : implement once final golden_labels.csv finalised data/filters
    phenotypes = ["c_diff", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
         "metastatic_cancer", "obesity","rheumatoid_arthritis", "roh_abuse", "vte"]
    # Verify that the phenotype is in the list of phenotypes
    if phenotype not in phenotypes:
        raise ValueError(f'Phenotype not in list of phenotypes : {phenotypes}')

    # Load dictionary of phenotypes and correspondings subject_ids as test set
    # TODO: must create dictionary of corresponding vetted discharge summaries + must also add distinction for cdiff
    #  and vte
    # TODO: must implement creation of accurate controls for test set - with label for pos/control;
    test_subject_ids = load_json(FILTERS_PATH+'test_subject_ids.json')
    test_subject_ids = test_subject_ids[phenotype]

    # Remove subjects with df_phenotype['subject_id'] not in df_core['subject_id']
    df_core = df_core[df_core['subject_id'].isin(test_subject_ids)]

    # Split the test set into a validation set and a test set
    # 50% of the test set will be used for testing and 50% for validation
    # Splitting based on subject_id
    df_core_test = df_core.sample(frac=split, random_state=42)
    df_core_val = df_core[~df_core['subject_id'].isin(df_core_test['subject_id'])]

    return df_core_test, df_core_val


def generate_cohort_train_test(df_core, phenotype):
    phenotypes = ["c_diff", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
         "metastatic_cancer", "obesity","rheumatoid_arthritis", "roh_abuse", "vte"]
    # Verify that the phenotype is in the list of phenotypes
    if phenotype not in phenotypes:
        raise ValueError(f'Phenotype not in list of phenotypes : {phenotypes}')

    return phenotype_cohort_train(df_core, phenotype, silver_labels=True), phenotype_cohort_test(df_core, phenotype)
