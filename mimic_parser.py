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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v!r}')

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--dataset_type', type=str, default='train', help='Type of dataset')
    parser.add_argument('--noisy_label', type=str2bool, default=True, help='Use noisy labels')
    parser.add_argument('--time', type=str2bool, default=True, help='Include time')
    parser.add_argument('--notes', type=str2bool, default=True, help='Include notes')
    parser.add_argument('--train', type=str2bool, default=True, help='Training mode')
    parser.add_argument('--NLP', type=str2bool, default=False, help='Use NLP')
    parser.add_argument('--pivot_type', type=str, default='simple', choices=['simple', 'timeseries'], help='Type of pivot')
    parser.add_argument('--agg_window', type=str, default='D', help='Aggregation window')
    parser.add_argument('--run_notes_nlp_parsing', type=str2bool, default=False, help='Run notes NLP parsing instead of the full pipeline')
    parser.add_argument('--debug_type', type=str, choices=['DIAG', 'PROC', 'MEDS', 'CHART', 'LABS', 'RADNOTES', 'DSNOTES'], help='Debug type (refers to the database of interest)')
    parser.add_argument('--debug_phenotype', type=str, choices=["c_diff_complication", "c_diff_past", "dementia", "depression",
                                                                "diabetes_type_1", "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
                                                                "metastatic_cancer", "obesity", "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"], help='Debug phenotype (refers to the phenotype parsed)')

    # Grouping parameters
    # Diagnosis ICD code diagnosis grouping
    parser.add_argument('--root_icd9', type=str2bool, default=True, help='Use root ICD9')
    parser.add_argument('--group_icd10', type=str2bool, default=True, help='Group ICD10')
    parser.add_argument('--options', type=str, default='ICD10', help='Options for ICD')
    # Medication grouping
    parser.add_argument('--med_grouping', type=str2bool, default=True, help='Use medication grouping')

    # Feature selection lower quantile
    parser.add_argument('--lower_labs', type=float, default=0.25, help='Lower threshold for labs')
    parser.add_argument('--lower_chart', type=float, default=0.25, help='Lower threshold for chart')
    parser.add_argument('--lower_icd', type=float, default=0.25, help='Lower threshold for ICD codes (diagnosis and '
                                                                      'procedures)')
    parser.add_argument('--lower_meds', type=float, default=0.25, help='Lower threshold for medications')
    parser.add_argument('--lower_nlp', type=float, default=0.20, help='Lower threshold for NLP codes')


    return parser.parse_args()


def notes_nlp_parsing(lower_quantile=0.20, train=False, parallel=False, n_workers=3):
    """
    Process and parse notes using NLP techniques.

    This function processes the notes for each phenotype and data type (RADNOTES, DSNOTES),
    generates NLP features, applies feature selection, and saves the pivoted data.

    Parameters:
    - train: False (default) processes the shared test/val notes; True processes the shared
      training notes (see process_and_save_shared_train_data) -- phenotype is irrelevant to
      the actual path in both cases since use_shared_paths=True is used throughout (the
      'c_diff_past' loop below is just a placeholder value for functions that need some
      phenotype argument).
    - parallel: use process_and_annotate_dataframe_parallel (multiprocessing) instead of the
      serial process_and_annotate_dataframe. Serial MedCAT inference was measured at
      ~240ms/note -- at training scale (~1.35M notes across DSNOTES+RADNOTES) that is ~3.7
      days of continuous single-core work, so this should be True for the training run.
    - n_workers: worker process count for the parallel path. Each worker loads its own full
      MedCAT model copy (~1.1GB RSS measured, since CAT.load_model_pack isn't
      picklable/shareable across macOS's 'spawn' multiprocessing start method) -- keep this
      conservative on an 8GB machine rather than matching the full CPU core count.
    """
    noisy_label = False
    for phenotype in ['c_diff_past']:
        for df_type in ['RADNOTES', 'DSNOTES']:
            print(f"[PROCESSING {phenotype} and {df_type}]")
            # generate_raw_csv (notes=True) already wrote the raw preprocessed text
            # (subject_id, hadm_id, text, ...) to this same shared NLP path -- that's what
            # we annotate here, then overwrite in place with the CUI-annotated version.
            nlp_path = generate_nlp_file_path(phenotype, df_type, train=train, noisy_label=False, use_shared_paths=True)
            df = read_gzip_or_parts(nlp_path)
            print(f"Loaded {len(df)} {df_type} notes to annotate")
            cat = generate_cat()
            dict_path = generate_dict_path(phenotype, df_type=df_type, train=train, label=noisy_label, use_shared_paths=True)
            print('FEATURE CREATION')
            if parallel:
                df = process_and_annotate_dataframe_parallel(df, text_column="text", n_workers=n_workers)
            else:
                df = process_and_annotate_dataframe(df, cat=cat, text_column="text")
            print('FEATURE CREATED')
            # Feature selection based on CUI frequency + Dictionary Export
            df = apply_feature_selection_and_export_cuis(df, cat=cat, output_file=dict_path, lower_quantile=lower_quantile)
            print(f"[NLP FEATURE SELECTION]")
            df = df[["subject_id", "hadm_id", "nlp_codes"]]
            save_gzip(df, nlp_path)
            print('PIVOTING')
            pivoted_df = pivot_and_aggregate(df, id_column='hadm_id', cui_column='nlp_codes')
            save_path = generate_pivot_path(phenotype, df_type, train, noisy_label, use_shared_paths=True)  # getting path
            # save_pivot (not save_csv) since pivoted_df is now a pandas Sparse-dtype
            # DataFrame -- see scripts/utils.py::save_pivot's docstring for why writing a
            # sparse DataFrame out as plain CSV defeats the point of staying sparse.
            actual_save_path = save_pivot(pivoted_df, save_path)
            print(f"[SUCCESSFULLY SAVED {df_type} PIVOTED DATA] {actual_save_path}")

def debug(TYPE, phenotype):
    """
    Debug function to generate raw CSV, feature selection, summary, and pivot restricted to a
    single table type (TYPE), for fast local iteration.

    Always runs the train=True path (as before). generate_raw_csv still only needs `phenotype`
    to write that one phenotype's index file -- the actual table generation/feature-selection/
    pivot run once against the shared training cohort (see generate_raw_csv's train=True branch
    and process_and_save_shared_train_data), same as main()'s params['train'] branch.

    Args:
        TYPE (str): Debug type (refers to the database of interest).
        phenotype (str): Debug phenotype (refers to the phenotype tested).
    """
    params = {
            'train'       : True,
            'debug'       : TYPE,
            'noisy_label' : True,
            'time'        : True,
            'notes'       : False,
            'NLP'         : False,
            'pivot_type'  : 'simple',  # alternative: 'timeseries'
            'agg_window'  : 'D',
    }

    generate_raw_csv([phenotype], **{k: v for k, v in params.items() if k in ['train', 'noisy_label',
                                                                              'time', 'notes', 'debug']})
    generate_features_selection(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label',
                                                                                       'notes', 'NLP', 'debug']})
    generate_summary(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes',
                                                                                'NLP', 'debug']})
    generate_pivot(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'pivot_type', 'noisy_label',
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
        'diag_options': args.options,
        'med_grouping': args.med_grouping,
        'lower_labs': args.lower_labs,
        'lower_chart': args.lower_chart,
        'lower_icd': args.lower_icd,
        'lower_meds': args.lower_meds,
        'lower_nlp': args.lower_nlp,
    }

    if params['train']:
        # train=True (silver-labeled training): the hadm_id universe is identical across all
        # phenotypes (only the per-phenotype 'y' label differs -- see
        # process_and_save_phenotype_data / process_and_save_shared_train_data), so
        # generate_raw_csv takes the full phenotype list in one call (writes per-phenotype
        # index files, then computes raw features once against the shared cohort), and
        # downstream steps run once against the shared paths instead of per-phenotype --
        # mirrors the train=False branch below exactly.
        generate_raw_csv(PHENOTYPES, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'time', 'notes', 'root_icd9', 'group_icd10', 'diag_options', 'med_grouping']})
        generate_features_selection(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP', 'lower_labs', 'lower_chart', 'lower_icd', 'lower_meds', 'lower_nlp']})
        generate_summary(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP']})
        generate_pivot(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'pivot_type', 'noisy_label', 'notes', 'agg_window']})
    else:
        # train=False (golden test/val): generate_raw_csv requires the full phenotype
        # list in one call (it builds a single shared cohort across all phenotypes),
        # and downstream steps run once against the shared paths instead of per-phenotype.
        generate_raw_csv(PHENOTYPES, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'time', 'notes', 'root_icd9', 'group_icd10', 'diag_options', 'med_grouping']})
        generate_features_selection(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP', 'lower_labs', 'lower_chart', 'lower_icd', 'lower_meds', 'lower_nlp']})
        generate_summary(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'noisy_label', 'notes', 'NLP']})
        generate_pivot(None, use_shared_paths=True, **{k: v for k, v in params.items() if k in ['train', 'pivot_type', 'noisy_label', 'notes', 'agg_window']})


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
    args = parse_args()
    main(args)
