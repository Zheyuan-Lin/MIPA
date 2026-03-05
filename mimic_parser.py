import argparse
import os
import dask.dataframe as dd
import pandas as pd

# Import your modules
from scripts.constants import *
from scripts.preprocessing.cohort import *
from scripts.preprocessing.hosp.hosp_preprocess import *
from scripts.generation.generation import *
from scripts.featureselection.feature_selection import *
from scripts.featureselection.pivot import *
from scripts.utils import *

import os


PHENOTYPES_NEW = ["c_diff_complication", "c_diff_past", "alcohol_abuse", "vte_complication", "vte_past"]


def set_project_root():
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_script_path)
    os.chdir(project_root_path)

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--dataset_type', type=str, default='train', help='Type of dataset')
    parser.add_argument('--noisy_label', type=bool, default=True, help='Use noisy labels')
    parser.add_argument('--time', type=bool, default=True, help='Include time')
    parser.add_argument('--notes', type=bool, default=True, help='Include notes')
    parser.add_argument('--train', type=bool, default=True, help='Training mode')
    parser.add_argument('--NLP', type=bool, default=False, help='Use NLP')
    parser.add_argument('--pivot_type', type=str, default='simple', choices=['simple', 'timeseries'], help='Type of pivot')
    parser.add_argument('--agg_window', type=str, default='D', help='Aggregation window')
    parser.add_argument('--run_notes_nlp_parsing', default=False, help='Run notes NLP parsing instead of the full pipeline')
    parser.add_argument('--debug_type', type=str, choices=['DIAG', 'PROC', 'MEDS', 'CHART', 'LABS', 'RADNOTES', 'DSNOTES'], help='Debug type (refers to the database of interest)')
    parser.add_argument('--debug_phenotype', type=str, choices=["c_diff_complication", "c_diff_past", "dementia", "depression",
                                                                "diabetes_type_1", "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
                                                                "metastatic_cancer", "obesity", "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"], help='Debug phenotype (refers to the phenotype parsed)')

    # Grouping parameters
    # Diagnosis ICD code diagnosis grouping
    parser.add_argument('--root_icd9', type=bool, default=True, help='Use root ICD9')
    parser.add_argument('--group_icd10', type=bool, default=True, help='Group ICD10')
    parser.add_argument('--options', type=str, default='ICD10', help='Options for ICD')
    # Medication grouping
    parser.add_argument('--med_grouping', type=bool, default=True, help='Use medication grouping')

    # Feature selection lower quantile
    parser.add_argument('--lower_labs', type=float, default=0.25, help='Lower threshold for labs')
    parser.add_argument('--lower_chart', type=float, default=0.25, help='Lower threshold for chart')
    parser.add_argument('--lower_icd', type=float, default=0.25, help='Lower threshold for ICD codes (diagnosis and '
                                                                      'procedures)')
    parser.add_argument('--lower_meds', type=float, default=0.25, help='Lower threshold for medications')
    parser.add_argument('--lower_nlp', type=float, default=0.20, help='Lower threshold for NLP codes')


    return parser.parse_args()


def notes_nlp_parsing(lower_quantile=0.20):
    """
    Process and parse notes using NLP techniques.

    This function processes the notes for each phenotype and data type (RADNOTES, DSNOTES),
    generates NLP features, applies feature selection, and saves the pivoted data.
    """
    train = False
    noisy_label = False
    for phenotype in ['c_diff_past']:
        for df_type in ['RADNOTES', 'DSNOTES']:
            print(f"[PROCESSING {phenotype} and {df_type}]")
            #path = generate_file_path(phenotype, df_type, train=False, noisy_label=False, long=True, validation=False, use_shared_paths=True)
            #df = read_gzip_or_parts(path)
            #cat = generate_cat()
            # Generate NLP features
            nlp_path = generate_nlp_file_path(phenotype, df_type, train=False, noisy_label=False,  use_shared_paths=True)
            #dict_path = generate_dict_path(phenotype, df_type=df_type, train=train, label=noisy_label, use_shared_paths=True)
            #print('FEATURE CREATION')
            #df = process_and_annotate_dataframe(df, cat=cat, text_column="text")
            #print('FEATURE CREATED')
            # Generate dictionary path : NB dictionary is saved in the pivot step for the other data types
            # Feature selection based on CUI frequency + Dictionary Export
            #df = apply_feature_selection_and_export_cuis(df, cat=cat, output_file=dict_path, lower_quantile=lower_quantile)
            #print(f"[NLP FEATURE SELECTION]")
            #df = df[["subject_id", "hadm_id", "nlp_codes"]]
            #save_gzip(df, nlp_path)
            nlp_path = generate_nlp_file_path(phenotype, df_type, train=False, noisy_label=False,  use_shared_paths=True)
            df = read_gzip_or_parts(nlp_path)
            print('PIVOTING')
            pivoted_df = pivot_and_aggregate(df, id_column='hadm_id', cui_column='nlp_codes')
            save_path = generate_pivot_path(phenotype, df_type, train, noisy_label, use_shared_paths=True)  # getting path
            save_csv(pivoted_df, save_path)
            print(f"[SUCCESSFULLY SAVED DSNOTES PIVOTED DATA] {save_path}")

def debug(TYPE, phenotype):
    """
    Debug function to generate raw CSV, feature selection, summary, and pivot for a specific phenotype.

    Args:
        TYPE (str): Debug type (refers to the database of interest).
        phenotype (str): Debug phenotype (refers to the phenotype tested).
    """
    params = {
            'dataset_type': 'train',
            'debug'       : TYPE,
            'noisy_label' : True,
            'time'        : True,
            'notes'       : False,
            'train'       : True,
            'NLP'         : False,
            'pivot_type'  : 'simple',  # alternative: 'timeseries'
            'agg_window'  : 'D',
    }

    for phenotype in [phenotype]:
        generate_raw_csv(phenotype, **{k: v for k, v in params.items() if k in ['dataset_type', 'noisy_label',
                                                                                'time', 'notes', 'debug']})
        generate_features_selection(phenotype, **{k: v for k, v in params.items() if k in ['train', 'noisy_label',
                                                                                           'notes', 'NLP', 'debug']})
        generate_summary(phenotype, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes',
                                                                                'NLP', 'debug']})
        generate_pivot(phenotype, **{k: v for k, v in params.items() if k in ['train', 'pivot_type', 'noisy_label',
                                                                              'notes', 'agg_window', 'debug']})

def main(args):
    """
    Main function to process the command-line arguments and execute the appropriate functions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.run_notes_nlp_parsing:
        notes_nlp_parsing(args.lower_nlp)
        return

    if args.debug_type and args.debug_phenotype:
        debug(args.debug_type, args.debug_phenotype)
        return

    # Define the parameter values in a dictionary
    params = {
        'train': args.train,
        'noisy_label': args.noisy_label,
        'time': args.time,
        'notes': args.notes,
        'NLP': args.NLP,
        'pivot_type': args.pivot_type,
        'agg_window': args.agg_window,
        'root_icd9': args.root_icd9,
        'group_icd10': args.group_icd10,
        'options': args.options,
        'med_grouping': args.med_grouping,
        'lower_labs': args.lower_labs,
        'lower_chart': args.lower_chart,
        'lower_icd': args.lower_icd,
        'lower_meds': args.lower_meds,
        'lower_nlp': args.lower_nlp,
    }

    for phenotype in PHENOTYPES:
        generate_raw_csv(phenotype, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'time', 'notes', 'root_icd9', 'group_icd10', 'options', 'med_grouping']})
        generate_features_selection(phenotype, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP', 'lower_labs', 'lower_chart', 'lower_icd', 'lower_meds', 'lower_nlp']})
        generate_summary(phenotype, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP']})
        generate_pivot(phenotype, **{k: v for k, v in params.items() if k in ['train', 'pivot_type', 'noisy_label', 'notes', 'agg_window']})


def main_noarg():
    """
    Main function to process the command-line arguments and execute the appropriate functions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """


    # Define the parameter values in a dictionary
    params = {
        'train': False,
        'noisy_label': False,
        'time': True,
        'notes': True,
        'NLP': True,
        'pivot_type': "simple",
        'agg_window': "D",
        'root_icd9': True,
        'group_icd10': True,
        'diag_options': "ICD10",
        'med_grouping': True,
        'lower_labs': 0.25,
        'lower_chart': 0.25,
        'lower_icd': 0.25,
        'lower_meds': 0.25,
        'lower_nlp': 0.20
    }

    if params['train']:
        # Original behavior for training: process each phenotype individually
        for phenotype in PHENOTYPES:
            try:
                print(f"Processing TRAINING for phenotype: {phenotype}")
                generate_raw_csv(phenotypes_to_process=phenotype,
                                 train=True,
                                 noisy_label=params['noisy_label'],
                                 time=params['time'], notes=params['notes'],
                                 root_icd9=params['root_icd9'], group_icd10=params['group_icd10'],
                                 diag_options=params['diag_options'], med_grouping=params['med_grouping'])

                generate_features_selection(phenotype, train=True, noisy_label=params['noisy_label'],
                                            notes=params['notes'], NLP=params['NLP'],
                                            lower_labs=params['lower_labs'], lower_chart=params['lower_chart'],
                                            lower_icd=params['lower_icd'], lower_meds=params['lower_meds'],
                                            lower_nlp=params['lower_nlp'])

                generate_summary(phenotype, train=True, noisy_label=params['noisy_label'],
                                 notes=params['notes'], NLP=params['NLP'])

                generate_pivot(phenotype, train=True, pivot_type=params['pivot_type'],
                               noisy_label=params['noisy_label'], notes=params['notes'],
                               agg_window=params['agg_window'])
                print(f"Successfully processed TRAINING for phenotype: {phenotype}")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while processing TRAINING for phenotype {phenotype}: {e}")
    else:
        # New behavior for test/validation (train=False)
        # 1. Generate indexes for all phenotypes and process shared raw data types
        print("--- Starting Test/Validation Data Processing (Indexes and Shared Raw Data) ---")
        try:
            generate_raw_csv(phenotypes_to_process=PHENOTYPES,
                             train=False,
                             noisy_label=params['noisy_label'],
                             time=params['time'], notes=params['notes'],
                             root_icd9=params['root_icd9'], group_icd10=params['group_icd10'],
                             diag_options=params['diag_options'], med_grouping=params['med_grouping'])
            print("--- Completed Index Generation and Shared Raw Data Processing ---")

            # 2. Generate feature selection outputs (dictionaries) using shared data - ONCE
            print("--- Starting Feature Selection for Shared Test/Validation Data ---")
            generate_features_selection(phenotype=None,
                                        train=False, noisy_label=params['noisy_label'],
                                        notes=params['notes'], NLP=params['NLP'],
                                        lower_labs=params['lower_labs'], lower_chart=params['lower_chart'],
                                        lower_icd=params['lower_icd'], lower_meds=params['lower_meds'],
                                        lower_nlp=params['lower_nlp'],
                                        use_shared_paths=True)
            print("--- Completed Feature Selection for Shared Test/Validation Data ---")

            # 3. Generate summary files using shared data - ONCE
            print("--- Starting Summary Generation for Shared Test/Validation Data ---")
            generate_summary(phenotype=None,
                             train=False, noisy_label=params['noisy_label'],
                             notes=params['notes'], NLP=params['NLP'],
                             use_shared_paths=True)
            print("--- Completed Summary Generation for Shared Test/Validation Data ---")

            # 4. Generate pivot tables using shared data - ONCE
            print("--- Starting Pivot Table Generation for Shared Test/Validation Data ---")
            generate_pivot(phenotype=None,
                           train=False, pivot_type=params['pivot_type'],
                           noisy_label=params['noisy_label'], notes=params['notes'],
                           agg_window=params['agg_window'],
                           use_shared_paths=True)
            print("--- Completed Pivot Table Generation for Shared Test/Validation Data ---")
            print("--- All Test/Validation Data Processing Successfully Completed ---")

        except Exception as e:
            print(f"ERROR: An unexpected error occurred during SHARED Test/Validation data processing: {e}")

if __name__ == '__main__':
    set_project_root()
    notes_nlp_parsing(0.20)
    args = parse_args()
    main(args)
