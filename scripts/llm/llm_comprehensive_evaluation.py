import logging
import os
import pickle
import traceback
from itertools import product
from collections import defaultdict
import asyncio
import time
import gc
from cachetools import LRUCache
import math
import random

import ast
import pandas as pd
import numpy as np

from scripts.llm.llm_module import *
from scripts.utils import *
from scripts.llm.reasoning_metrics import reasoning_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(module)s - %(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

fs = setup_gcs_access()


def load_phenotype_predictions(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                               show_config_bar=True, show_chunk_bar=True, reasoning=False):
    if os.path.exists(metrics_pickle_path):
        try:
            with open(metrics_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
            logging.debug(f"Loaded phenotype metrics from {metrics_pickle_path}")
            return phenotype_predictions
        except Exception as e:
            logging.error(f"Failed to load phenotype metrics from {metrics_pickle_path}: {e}")

    logging.debug(f"Phenotype metrics not found at {metrics_pickle_path}. Running llm_parsing...")
    return generate_predictions(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                                show_config_bar, show_chunk_bar, reasoning)


def generate_predictions(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path, show_config_bar=True,
                         show_chunk_bar=True, reasoning=False):
    phenotype_predictions = llm_parsing(phenotype, model_name, configurations, cleaned_data,
                                        show_config_bar=show_config_bar, show_chunk_bar=show_chunk_bar,
                                        reasoning=reasoning)

    directory = os.path.dirname(metrics_pickle_path)
    os.makedirs(directory, exist_ok=True)
    try:
        with open(metrics_pickle_path, 'wb') as file:
            pickle.dump(phenotype_predictions, file)
        logging.info(f"Saved phenotype metrics to {metrics_pickle_path}")
    except Exception as e:
        logging.error(f"Failed to save phenotype metrics to {metrics_pickle_path}: {e}")

    return phenotype_predictions


def llm_parsing(phenotype, model_name, configurations, cleaned_data, show_config_bar=True, show_chunk_bar=True,
                reasoning=False):
    phenotype_parsed = {}
    phenotype_preparsed = {}

    config_pbar = tqdm(
        total=len(configurations),
        desc=f"Configs for {phenotype} with {model_name}",
        unit="config", colour="green", position=1, leave=True, dynamic_ncols=True, miniters=1
    ) if show_config_bar else None

    try:
        for chunking_method, prompt_strategy, aggregation_method in configurations:
            if config_pbar:
                config_pbar.set_description(f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}")
                config_pbar.refresh()

            key = (chunking_method, prompt_strategy, phenotype, model_name)
            post_aggregation_key = (chunking_method, prompt_strategy, aggregation_method, phenotype, model_name)

            # -- FIRST STAGE: PRE-AGGREGATION (Parsing/Extracting) --
            if key not in phenotype_preparsed:
                try:
                    # Attempt chunking
                    try:
                        chunked_data = cleaned_data.chunk_text(method=chunking_method)
                        if chunked_data is None or chunked_data.empty:
                            logging.error(
                                f"No data was returned from chunk_text for method '{chunking_method}'. "
                                f"Skipping this config for {phenotype} with {model_name}."
                            )
                            continue
                    except TypeError as e:
                        logging.error(
                            f"TypeError while chunking data with method '{chunking_method}' for "
                            f"{phenotype} with {model_name}: {e}. Skipping this configuration."
                        )
                        continue
                    except KeyError as e:
                        logging.error(
                            f"KeyError while chunking data with method '{chunking_method}' for "
                            f"{phenotype} with {model_name}: Missing key {e}. Skipping this configuration."
                        )
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error while chunking data: {e}")
                        traceback.print_exc()
                        continue

                    # Set up prompt engineering and parser
                    prompt_engineering = PromptEngineering(prompt_type=prompt_strategy)
                    llm_parser = LLMParser(
                        model_name=model_name,
                        prompt_engineering=prompt_engineering,
                        api_url=OPENROUTER_URL,
                        api_key=OPENROUTER_API_KEY
                    )

                    # Choose the parsing function
                    parsing_bar = tqdm(
                        total=len(chunked_data),
                        desc="Parsing chunks",
                        unit="chunk",
                        leave=True,
                        colour="blue",
                        position=2
                    ) if show_chunk_bar else None

                    try:
                        if reasoning:
                            # parse with reasoning
                            pre_aggregation_results, errors_df = llm_parser.extract_then_phenotype_reasoning(
                                chunked_data, phenotype, parsing_bar
                            )
                            pre_agg_path = generate_parse_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )
                        else:
                            # parse without reasoning
                            if prompt_strategy != 'etp':
                                pre_aggregation_results, errors_df = llm_parser.parse(
                                    chunked_data, phenotype, parsing_bar
                                )
                            else:
                                pre_aggregation_results, errors_df = llm_parser.extract_then_phenotype(
                                    chunked_data, phenotype, parsing_bar
                                )
                            pre_agg_path = generate_parse_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )

                    except KeyError as e:
                        logging.error(
                            f"KeyError while parsing for {phenotype} with {model_name}, "
                            f"config {chunking_method}-{prompt_strategy}: Missing key {e}. Skipping."
                        )
                        traceback.print_exc()
                        if parsing_bar:
                            parsing_bar.close()
                        continue
                    except TypeError as e:
                        logging.error(
                            f"TypeError while parsing for {phenotype} with {model_name}, "
                            f"config {chunking_method}-{prompt_strategy}: {e}. Skipping."
                        )
                        traceback.print_exc()
                        if parsing_bar:
                            parsing_bar.close()
                        continue
                    except Exception as e:
                        logging.error(
                            f"Error during parsing for {phenotype} with {model_name}, "
                            f"Config: {chunking_method}-{prompt_strategy}: {e}"
                        )
                        traceback.print_exc()
                        if parsing_bar:
                            parsing_bar.close()
                        continue
                    finally:
                        if parsing_bar:
                            parsing_bar.close()

                    # Save results
                    phenotype_preparsed[key] = pre_aggregation_results
                    save_csv(pre_aggregation_results, pre_agg_path)
                    if errors_df is not None:
                        save_csv(errors_df, errors_path)

                except requests.exceptions.RequestException as e:
                    logging.error(f"Network error during parsing: {e}")
                    continue
                except IOError as e:
                    logging.error(f"File operation failed: {e}")
                    continue
                except Exception as e:
                    logging.error(
                        f"Error during parsing for {phenotype} with {model_name}, "
                        f"Config: {chunking_method}-{prompt_strategy}: {e}"
                    )
                    traceback.print_exc()
                    continue

            # -- SECOND STAGE: POST-AGGREGATION (Aggregating) --
            if post_aggregation_key not in phenotype_parsed:
                try:
                    result_aggregator = ResultAggregator(aggregation_method=aggregation_method)
                    logging.info('Aggregating results...')
                    if aggregation_method == 'etp':
                        result_aggregator.set_phenotype(phenotype)
                        # dynamic if learned threshold, here we fix it
                        result_aggregator.set_etp_rule_type('fixed')
                        result_aggregator.set_etp_threshold(0.4)

                    final_results = result_aggregator.aggregate(phenotype_preparsed[key])

                    post_agg_path = (
                        generate_parse_reasoning_path(
                            phenotype,
                            f"{chunking_method}_{aggregation_method}_{prompt_strategy}_{model_name}_post",
                            False
                        ) if reasoning else generate_parse_path(
                            phenotype,
                            f"{chunking_method}_{aggregation_method}_{prompt_strategy}_{model_name}_post",
                            False
                        )
                    )
                    save_csv(final_results, post_agg_path)
                    phenotype_parsed[post_aggregation_key] = final_results
                except KeyError as e:
                    logging.error(
                        f"KeyError during aggregation for {phenotype} with {model_name}, "
                        f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}: Missing key {e}. "
                        "Skipping aggregation."
                    )
                    traceback.print_exc()
                except TypeError as e:
                    logging.error(
                        f"TypeError during aggregation for {phenotype} with {model_name}, "
                        f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}: {e}. "
                        "Skipping aggregation."
                    )
                    traceback.print_exc()
                except Exception as e:
                    logging.error(
                        f"Error during post-aggregation for {phenotype} with {model_name}, "
                        f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}: {e}"
                    )
                    traceback.print_exc()

            if config_pbar:
                config_pbar.update(1)

    except Exception as e:
        logging.error(f"An error occurred while processing {phenotype} with {model_name}: {e}")
        traceback.print_exc()

    finally:
        if config_pbar:
            config_pbar.close()

    return phenotype_parsed


def evaluation(phenotype, model_name, configurations, evaluator, phenotype_parsed):
    phenotype_metrics = {}
    misclassifications = {}
    properclassifications = {}

    # Evaluation manager already contains the gold standard df
    evaluation_manager = EvaluationManager(evaluator=evaluator)

    for chunking_method, prompt_strategy, aggregation_method in configurations:
        key = (chunking_method, prompt_strategy, aggregation_method, phenotype, model_name)

        if key in phenotype_parsed:
            final_results = phenotype_parsed[key]

            config = ExperimentConfig(phenotype=phenotype, chunking_method=chunking_method,
                                      aggregation_method=aggregation_method, prompt_strategy=prompt_strategy,
                                      model_name=model_name)

            # Computing evaluation results for each config/phenotype
            metrics, errors_hadm_ids_df, proper_hadm_ids_df = evaluation_manager.run_experiment(final_results, config,
                                                                                                phenotype)

            phenotype_metrics[key] = metrics

            errors_hadm_ids_df['Phenotype'] = phenotype
            errors_hadm_ids_df['ModelName'] = model_name
            errors_hadm_ids_df['ChunkingMethod'] = chunking_method
            errors_hadm_ids_df['PromptStrategy'] = prompt_strategy
            errors_hadm_ids_df['AggregationMethod'] = aggregation_method

            proper_hadm_ids_df['Phenotype'] = phenotype
            proper_hadm_ids_df['ModelName'] = model_name
            proper_hadm_ids_df['ChunkingMethod'] = chunking_method
            proper_hadm_ids_df['PromptStrategy'] = prompt_strategy
            proper_hadm_ids_df['AggregationMethod'] = aggregation_method

            misclassifications[key] = errors_hadm_ids_df
            properclassifications[key] = proper_hadm_ids_df

            metrics_path = generate_metrics_path(phenotype, config)
            save_csv(metrics, metrics_path)

    return phenotype_metrics, misclassifications, properclassifications


def safe_literal_eval(s):
    """Safely evaluate a string as a dictionary."""
    try:
        return ast.literal_eval(s) if isinstance(s, str) else s
    except (ValueError, SyntaxError):
        return {}


def aggregate_and_save_dataframes(df: pd.DataFrame, output_dir: str = 'output/classic/metrics/analysis') -> None:
    """
    Aggregate summary metrics by configuration and save one CSV file per configuration.
    This function allows to interpret the results more easily and compare different configurations.

    Args:
        df (pd.DataFrame): Input DataFrame with evaluation results
        output_dir (str): Directory to save output files
    """
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert numeric columns that might be strings to float
    numeric_columns = ['n', 'valid_n', 'accuracy', 'precision', 'recall', 'f1_score',
                       'true_negatives', 'false_positives', 'false_negatives', 'true_positives']

    # Convert to numeric, replacing errors with NaN
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Define dictionary columns and parse them
    dict_columns = ['n_chunks', 'chunk_tokens_mean', 'input_tokens_mean',
                    'output_tokens_mean', 'processing_time_mean']

    for col in dict_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)

    # Define metrics with appropriate aggregation methods
    metrics = {
        'n': 'sum',
        'valid_n': 'sum',
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'true_negatives': 'sum',
        'false_positives': 'sum',
        'false_negatives': 'sum',
        'true_positives': 'sum',
        'n_chunks': 'mean',
        'chunk_tokens_mean': 'mean',
        'input_tokens_mean': 'mean',
        'output_tokens_mean': 'mean',
        'processing_time_mean': 'mean'
    }

    try:
        # Group by configuration columns
        # config_columns = ['model']  # model-level analysis
        config_columns = ['model', 'prompt_strategy', 'chunking_method', 'aggregation_method']  # config-level analysis

        # Group by configuration and aggregate
        grouped = df.groupby(config_columns)

        config_df_list = []
        # For each unique configuration
        for config_values, group_df in grouped:
            # Aggregate metrics for this configuration
            agg_metrics = pd.DataFrame()

            # Aggregate each metric separately
            for col, method in metrics.items():
                if col not in group_df.columns:
                    continue  # Skip if column is missing

                if method == 'sum':
                    agg_metrics[col] = [group_df[col].sum()]
                elif method == 'mean':
                    agg_metrics[col] = [group_df[col].mean()]
                elif method == 'mean_dict':
                    # Collect all values from dictionaries in the group
                    all_values = []
                    for d in group_df[col]:
                        if isinstance(d, dict):
                            all_values.extend(d.values())
                    # Compute mean if there are values, otherwise set to NaN
                    mean_val = pd.NA
                    if all_values:
                        mean_val = sum(all_values) / len(all_values)
                    agg_metrics[col] = [mean_val]

            # Add configuration columns
            for col, value in zip(config_columns, config_values):
                agg_metrics[col] = value

            # Round numeric values (excluding dict columns which are now aggregated)
            numeric_cols = [col for col in agg_metrics.columns if pd.api.types.is_numeric_dtype(agg_metrics[col])]
            agg_metrics[numeric_cols] = agg_metrics[numeric_cols].round(4)

            config_df_list.append(agg_metrics)

        # Define custom orders
        chunking_method_order = ['none', 'naive', 'document']
        aggregation_method_order = ['any', 'two', 'majority', 'etp']
        prompt_strategy_order = ['zero_shot', 'few_shot', 'COT', 'etp']

        # Convert columns to categorical with custom orders
        config_dfs = pd.concat(config_df_list, ignore_index=True)

        columns_to_convert = [
            ('chunking_method', chunking_method_order),
            ('aggregation_method', aggregation_method_order),
            ('prompt_strategy', prompt_strategy_order)
        ]

        for column, order in columns_to_convert:
            try:
                config_dfs[column] = pd.Categorical(
                    config_dfs[column],
                    categories=order,
                    ordered=True
                )
            except KeyError:
                print(f"Warning: Column '{column}' not found in the DataFrame.")

        # Sort with model alphabetically and other columns by categorical order
        sort_columns = ['model'] + [col for col, _ in columns_to_convert if col in config_dfs.columns]

        try:
            config_dfs = config_dfs.sort_values(
                by=sort_columns,
                ascending=[True] * len(sort_columns)
            )
        except KeyError as e:
            print(f"Warning: Error during sorting. Column not found: {e}")

        file_name = "all_configs.csv"
        file_path = os.path.join(output_dir, file_name)

        # Save configuration results
        config_dfs.to_csv(file_path, index=False)
        print(f"Saved aggregated results for all configurations")

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        print(f"DataFrame info:")
        print(df.info())


async def llm_phenotyping_async_reasoning(run_parsing=True, run_evaluation=True, run_json_parsing_error_analysis=True,
                                          show_overall_bar=True, show_config_bar=True, show_chunk_bar=True,
                                          phenotypes=PHENOTYPES, models=MODELS_OPEN, chunking_methods=CHUNKING_METHODS,
                                          prompt_strategies=PROMPT_STRATEGIES, aggregation_methods=AGGREGATION_METHODS,
                                          batch_size=1, reasoning_df_path=None):
    """
    Async version of llm_phenotyping that uses batch processing for API calls.
    Task unit is each phenotype-model combination.
    Reasoning version only parses a subset of data that is misclassified by both models as reasoning_df
    Additional parameter:
        batch_size (int): Number of chunks to process in parallel. Default is 100.
    """
    fs = setup_gcs_access()

    chunking_methods = chunking_methods
    prompt_strategies = prompt_strategies
    aggregation_methods = aggregation_methods

    all_combinations = product(chunking_methods, prompt_strategies, aggregation_methods)

    configurations = [
        (chunking, prompt, aggregation)
        for chunking, prompt, aggregation in all_combinations
        if (prompt == 'etp' and aggregation == 'etp') or
           (chunking == 'none' and prompt != 'etp' and aggregation == 'any') or
           (chunking == 'none' and prompt == 'etp' and aggregation == 'etp') or
           (chunking != 'none' and prompt != 'etp' and aggregation != 'etp')
    ]

    phenotypes = phenotypes
    models = models

    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    cleaned_data = data.clean_text(expand_abbreviations=True)
    total_discharge_summary = len(data)

    evaluator = Evaluator(GOLDEN_LABELS_PATH)

    all_metrics = {}
    misclassifications_concat = {}
    properclassifications_concat = {}

    overall_iterator = list(product(phenotypes, models))
    overall_pbar = tqdm(total=len(overall_iterator), desc="Overall Progress", unit="task", colour="red",
                        position=0, leave=True, dynamic_ncols=True, miniters=1) if show_overall_bar else None

    for phenotype, model_name in overall_iterator:
        # Filtering data to evaluate only misclassified data
        reasoning_df = pd.read_csv(reasoning_df_path) if reasoning_df_path is not None else None
        if reasoning_df is not None:
            # Convert hadm_ids to a set for potentially faster lookups
            hadm_ids = set(reasoning_df[
                               (reasoning_df['both_models_misclassified'] == True) &
                               (reasoning_df['phenotype'] == phenotype)
                               ]['hadm_id'].unique())

            # Filter using the set of hadm_ids
            cleaned_data_filtered = cleaned_data[cleaned_data['hadm_id'].isin(hadm_ids)]
            logger.info(f"Filtered data #, {len(cleaned_data_filtered)}")

        if show_overall_bar:
            overall_pbar.set_postfix_str(f"Processing {phenotype} with {model_name}", refresh=True)

        key = (phenotype, model_name)
        parsed_pickle_path = f"{PICKLE_PATH}{phenotype}_{model_name}_parsed.pkl"

        # Parsing all notes and predicting phenotypes according to each configuration
        if run_parsing:
            phenotype_predictions = await load_phenotype_predictions_async(
                phenotype, model_name, configurations, cleaned_data_filtered,
                parsed_pickle_path, show_config_bar, show_chunk_bar,
                batch_size=batch_size
            )

            phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)


        else:
            with open(parsed_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
                phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)

        # Evaluating all predictions and saving metrics
        if run_evaluation:
            try:
                clean_phenotype_metrics_df, misclassifications_df, properclassifications_df = evaluation(
                    phenotype, model_name, configurations, evaluator, phenotype_predictions
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")

            all_metrics[key] = clean_phenotype_metrics_df
            misclassifications_concat[key] = misclassifications_df
            properclassifications_concat[key] = properclassifications_df

        if show_overall_bar:
            overall_pbar.update(1)

    if show_overall_bar:
        overall_pbar.close()

    if run_evaluation:
        all_metrics_list = []
        misclassifications_list = []
        properclassifications_list = []
        for phenotype in phenotypes:
            for model_name in models:
                key = (phenotype, model_name)
                if key in all_metrics:
                    all_metrics_list.append(all_metrics[key])
                if key in misclassifications_concat:
                    misclassifications_list.append(misclassifications_concat[key])
                    properclassifications_list.append(properclassifications_concat[key])

        try:
            if all_metrics_list:
                # Initialize list for cleaned DataFrames
                all_metrics_dfs = []

                for i, metric_dict in enumerate(all_metrics_list):
                    for column in metric_dict.values():
                        if isinstance(column, pd.DataFrame):
                            # Remove duplicate columns
                            column = column.loc[:, ~column.columns.duplicated()]

                            # Reset index to ensure clean concatenation
                            df_reset = column.reset_index(drop=True)

                            # Add to list
                            all_metrics_dfs.append(df_reset)
                        else:
                            logging.warning(f"Skipping non-DataFrame object: {type(column)}")

                if all_metrics_dfs:
                    # Concatenate with clean indices
                    clean_all_metrics_df = pd.concat(all_metrics_dfs, axis=0, ignore_index=True)

                    # Remove any duplicate columns that might have been created during concatenation
                    clean_all_metrics_df = clean_all_metrics_df.loc[:, ~clean_all_metrics_df.columns.duplicated()]

                    # Save the concatenated DataFrame
                    overall_summary_path = generate_metrics_summary_path(level='overall')
                    save_csv(clean_all_metrics_df, overall_summary_path)

                    # Analyzing the data
                    aggregate_and_save_dataframes(clean_all_metrics_df)
                    logging.info(f"All evaluation metrics concatenated and saved to {overall_summary_path}")
                    logging.info(f"Final DataFrame shape: {clean_all_metrics_df.shape}")
                else:
                    logging.warning("No metric DataFrames to concatenate.")
            else:
                logging.warning("No metrics to concatenate.")
        except Exception as e:
            logging.error(f"Could not concatenate all metrics: {e}")
            logging.error(f"Error type: {type(e).__name__}")
            if 'all_metrics_dfs' in locals():
                for i, df in enumerate(all_metrics_dfs):
                    try:
                        # Log column names without duplicates
                        unique_cols = list(dict.fromkeys(df.columns))
                        logging.error(f"DataFrame {i} unique columns: {unique_cols}")
                        logging.error(f"DataFrame {i} shape: {df.shape}")
                    except Exception as inner_e:
                        logging.error(f"Error inspecting DataFrame {i}: {inner_e}")

        try:
            if misclassifications_list:
                misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in
                                          misclass_dict.values()]
                properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in
                                             properclass_dict.values()]
                if misclassifications_dfs:
                    misclassifications_df = pd.concat(misclassifications_dfs, ignore_index=True)
                    misclassification_path = generate_misclassification_path(status='raw')
                    save_csv(misclassifications_df, misclassification_path)

                    properclassifications_df = pd.concat(properclassifications_dfs, ignore_index=True)
                    properclassifications_path = generate_properclassification_path(status='raw')
                    save_csv(properclassifications_df, properclassifications_path)

                    misclassification_report_path = generate_misclassification_path(status='reports')
                    results = analyze_misclassification_errors(misclassification_path, 2)
                    export_results(results, misclassification_report_path)
                    logging.info("All misclassifications analyzed and saved.")
                else:
                    logging.warning("No misclassification DataFrames to process.")
            else:
                logging.warning("No misclassifications to process.")
        except Exception as e:
            logging.error(f"Error processing misclassifications: {e}")

    if run_json_parsing_error_analysis:
        errors_df = aggregate_error_csvs(phenotypes)
        if not errors_df.empty:
            try:
                errors_per_model, errors_per_model_and_phenotype, errors_per_config = generate_error_report(errors_df)
                errors_per_model_path = generate_errors_report_path("errors_per_model")
                errors_per_model_phenotype_path = generate_errors_report_path("errors_per_model_and_phenotype")
                errors_per_config_path = generate_errors_report_path("errors_per_config")

                save_csv(errors_per_model, errors_per_model_path)
                save_csv(errors_per_model_and_phenotype, errors_per_model_phenotype_path)
                save_csv(errors_per_config, errors_per_config_path)
                logging.info("JSON parsing error report successfully saved")

            except Exception as e:
                logging.error(f"Could not generate an error report: {e}")


async def llm_phenotyping_async(run_parsing=True, run_evaluation=True, run_json_parsing_error_analysis=True,
                                show_overall_bar=True, show_config_bar=True, show_chunk_bar=True,
                                phenotypes=PHENOTYPES, models=MODELS_OPEN, chunking_methods=CHUNKING_METHODS,
                                prompt_strategies=PROMPT_STRATEGIES, aggregation_methods=AGGREGATION_METHODS,
                                batch_size=100, uncertainty=False):
    """
    Async version of llm_phenotyping that uses batch processing for API calls.
    Task unit is each phenotype-model combination.
    Additional parameter:
        batch_size (int): Number of chunks to process in parallel. Default is 100.
    """
    fs = setup_gcs_access()

    chunking_methods = chunking_methods
    prompt_strategies = prompt_strategies
    aggregation_methods = aggregation_methods

    all_combinations = product(chunking_methods, prompt_strategies, aggregation_methods)

    configurations = [
        (chunking, prompt, aggregation)
        for chunking, prompt, aggregation in all_combinations
        if (prompt == 'etp' and aggregation == 'etp') or
           (chunking == 'none' and prompt != 'etp' and aggregation == 'any') or
           (chunking == 'none' and prompt == 'etp' and aggregation == 'etp') or
           (chunking != 'none' and prompt != 'etp' and aggregation != 'etp')
    ]

    phenotypes = phenotypes
    models = models

    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    cleaned_data = data.clean_text(expand_abbreviations=True)
    total_discharge_summary = len(data)

    evaluator = Evaluator(GOLDEN_LABELS_PATH)

    # Create dictionaries to collect results for all phenotype-model combinations
    all_metrics = {}
    misclassifications_concat = {}
    properclassifications_concat = {}

    # For uncertainty mode, create additional dictionaries to collect results by confidence level
    if uncertainty:
        confidence_levels = ['all', 'high', 'medium', 'low']
        # Use a nested defaultdict to store metrics by confidence level, then by phenotype-model key
        all_metrics_by_confidence = {level: {} for level in confidence_levels}
        misclassifications_by_confidence = {level: {} for level in confidence_levels}
        properclassifications_by_confidence = {level: {} for level in confidence_levels}

    overall_iterator = list(product(phenotypes, models))
    overall_pbar = tqdm(total=len(overall_iterator), desc="Overall Progress", unit="task", colour="red",
                        position=0, leave=True, dynamic_ncols=True, miniters=1) if show_overall_bar else None

    for phenotype, model_name in overall_iterator:
        if show_overall_bar:
            overall_pbar.set_postfix_str(f"Processing {phenotype} with {model_name}", refresh=True)

        key = (phenotype, model_name)
        parsed_pickle_path = f"{PICKLE_PATH}{phenotype}_{model_name}_parsed.pkl"

        # Parsing all notes and predicting phenotypes according to each configuration
        if run_parsing:
            phenotype_predictions = await load_phenotype_predictions_async(
                phenotype, model_name, configurations, cleaned_data,
                parsed_pickle_path, show_config_bar, show_chunk_bar,
                batch_size=batch_size, uncertainty=uncertainty
            )

            phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)
        else:
            with open(parsed_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
                phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)

        # Evaluating all predictions and saving metrics
        if run_evaluation:
            if uncertainty:
                # Process data separately for each confidence level
                for confidence_level in confidence_levels:
                    # Filter phenotype_predictions by confidence level
                    filtered_predictions = {}
                    
                    for config_key, predictions_df in phenotype_predictions.items():
                        if 'confidence' in predictions_df.columns:
                            if confidence_level == 'all':
                                # Use all predictions
                                filtered_predictions[config_key] = predictions_df
                            else:
                                # Filter by specific confidence level
                                filtered_df = predictions_df[predictions_df['confidence'] == confidence_level].copy()
                                if not filtered_df.empty:
                                    filtered_predictions[config_key] = filtered_df
                        else:
                            # If confidence column is missing, only include in 'all' category
                            if confidence_level == 'all':
                                filtered_predictions[config_key] = predictions_df
                            logging.warning(f"Confidence column missing in predictions for {config_key}")
                    
                    # Skip evaluation if no predictions for this confidence level
                    if not filtered_predictions:
                        logging.info(f"No predictions found for {phenotype}, {model_name} with confidence level: {confidence_level}")
                        continue
                    
                    # Run evaluation for this confidence level
                    metrics_df, misclass_df, properclass_df = evaluation(
                        phenotype, model_name, configurations, evaluator, filtered_predictions
                    )
                    
                    # Add confidence level column to results
                    for k, df in metrics_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    for k, df in misclass_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    for k, df in properclass_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    # Store results for this confidence level in the global dictionaries
                    all_metrics_by_confidence[confidence_level][key] = metrics_df
                    misclassifications_by_confidence[confidence_level][key] = misclass_df
                    properclassifications_by_confidence[confidence_level][key] = properclass_df
            else:
                # Standard evaluation without confidence levels
                clean_phenotype_metrics_df, misclassifications_df, properclassifications_df = evaluation(
                    phenotype, model_name, configurations, evaluator, phenotype_predictions
                )

                all_metrics[key] = clean_phenotype_metrics_df
                misclassifications_concat[key] = misclassifications_df
                properclassifications_concat[key] = properclassifications_df

        if show_overall_bar:
            overall_pbar.update(1)

    if show_overall_bar:
        overall_pbar.close()

    # After processing all phenotype-model combinations, process and save the final results
    if run_evaluation:
        if uncertainty:
            # Process and combine results from all confidence levels
            all_metrics_list = []
            misclassifications_list = []
            properclassifications_list = []
            
            # Combine results from all confidence levels
            for confidence_level in confidence_levels:
                for key, metrics_df in all_metrics_by_confidence[confidence_level].items():
                    all_metrics_list.append(metrics_df)
                    
                for key, misclass_df in misclassifications_by_confidence[confidence_level].items():
                    misclassifications_list.append(misclass_df)
                    
                for key, properclass_df in properclassifications_by_confidence[confidence_level].items():
                    properclassifications_list.append(properclass_df)
            
            try:
                if all_metrics_list:
                    # Process metrics with confidence level
                    all_metrics_dfs = []
                    
                    for metric_dict in all_metrics_list:
                        for column in metric_dict.values():
                            if isinstance(column, pd.DataFrame):
                                # Remove duplicate columns
                                column = column.loc[:, ~column.columns.duplicated()]
                                
                                # Reset index for clean concatenation
                                df_reset = column.reset_index(drop=True)
                                
                                # Add to list
                                all_metrics_dfs.append(df_reset)
                            else:
                                logging.warning(f"Skipping non-DataFrame object: {type(column)}")
                    
                    if all_metrics_dfs:
                        # Concatenate with clean indices
                        clean_all_metrics_df = pd.concat(all_metrics_dfs, axis=0, ignore_index=True)
                        
                        # Remove any duplicate columns
                        clean_all_metrics_df = clean_all_metrics_df.loc[:, ~clean_all_metrics_df.columns.duplicated()]
                        
                        # Save the consolidated results for all confidence levels
                        overall_summary_path = generate_metrics_summary_path(level='overall_confidence')
                        save_csv(clean_all_metrics_df, overall_summary_path)
                        
                        # Also save individual confidence level files
                        for confidence_level in confidence_levels:
                            level_df = clean_all_metrics_df[clean_all_metrics_df['confidence_level'] == confidence_level]
                            if not level_df.empty:
                                level_path = generate_metrics_summary_path(level=f'confidence_{confidence_level}')
                                save_csv(level_df, level_path)
                        
                        # Analyzing the data - with confidence grouping
                        aggregate_and_save_dataframes(clean_all_metrics_df, 
                                                     output_dir='output/classic/metrics/analysis_uncertainty')
                        
                        logging.info(f"All uncertainty-based evaluation metrics saved to {overall_summary_path}")
                        logging.info(f"Final DataFrame shape: {clean_all_metrics_df.shape}")
                    else:
                        logging.warning("No uncertainty metric DataFrames to concatenate.")
                else:
                    logging.warning("No uncertainty metrics to concatenate.")
            except Exception as e:
                logging.error(f"Could not concatenate uncertainty metrics: {e}")
                logging.error(f"Error type: {type(e).__name__}")
                if 'all_metrics_dfs' in locals():
                    for i, df in enumerate(all_metrics_dfs):
                        try:
                            unique_cols = list(dict.fromkeys(df.columns))
                            logging.error(f"DataFrame {i} unique columns: {unique_cols}")
                            logging.error(f"DataFrame {i} shape: {df.shape}")
                        except Exception as inner_e:
                            logging.error(f"Error inspecting DataFrame {i}: {inner_e}")
            
            try:
                if misclassifications_list:
                    misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in
                                              misclass_dict.values()]
                    properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in
                                                 properclass_dict.values()]
                    
                    if misclassifications_dfs:
                        misclassifications_df = pd.concat(misclassifications_dfs, ignore_index=True)
                        misclassification_path = generate_misclassification_path(status='confidence_all')
                        save_csv(misclassifications_df, misclassification_path)
                        
                        properclassifications_df = pd.concat(properclassifications_dfs, ignore_index=True)
                        properclassifications_path = generate_properclassification_path(status='confidence_all')
                        save_csv(properclassifications_df, properclassifications_path)
                        
                        # Save individual confidence level files
                        for confidence_level in confidence_levels:
                            level_misclass = misclassifications_df[misclassifications_df['confidence_level'] == confidence_level]
                            if not level_misclass.empty:
                                level_path = generate_misclassification_path(status=f'confidence_{confidence_level}')
                                save_csv(level_misclass, level_path)
                            
                            level_proper = properclassifications_df[properclassifications_df['confidence_level'] == confidence_level]
                            if not level_proper.empty:
                                level_path = generate_properclassification_path(status=f'confidence_{confidence_level}')
                                save_csv(level_proper, level_path)
                        
                        # Analyze misclassifications with uncertainty info
                        misclassification_report_path = generate_misclassification_path(status='confidence_report')
                        results = analyze_misclassification_errors(misclassification_path, 2)
                        export_results(results, misclassification_report_path)
                        
                        logging.info("All uncertainty-based misclassifications analyzed and saved.")
                    else:
                        logging.warning("No uncertainty misclassification DataFrames to process.")
                else:
                    logging.warning("No uncertainty misclassifications to process.")
            except Exception as e:
                logging.error(f"Error processing uncertainty misclassifications: {e}")
        else:
            # Standard evaluation processing (no confidence levels)
            all_metrics_list = []
            misclassifications_list = []
            properclassifications_list = []
            for phenotype in phenotypes:
                for model_name in models:
                    key = (phenotype, model_name)
                    if key in all_metrics:
                        all_metrics_list.append(all_metrics[key])
                    if key in misclassifications_concat:
                        misclassifications_list.append(misclassifications_concat[key])
                        properclassifications_list.append(properclassifications_concat[key])

            try:
                if all_metrics_list:
                    # Initialize list for cleaned DataFrames
                    all_metrics_dfs = []

                    for i, metric_dict in enumerate(all_metrics_list):
                        for column in metric_dict.values():
                            if isinstance(column, pd.DataFrame):
                                # Remove duplicate columns
                                column = column.loc[:, ~column.columns.duplicated()]

                                # Reset index to ensure clean concatenation
                                df_reset = column.reset_index(drop=True)

                                # Add to list
                                all_metrics_dfs.append(df_reset)
                            else:
                                logging.warning(f"Skipping non-DataFrame object: {type(column)}")

                    if all_metrics_dfs:
                        # Concatenate with clean indices
                        clean_all_metrics_df = pd.concat(all_metrics_dfs, axis=0, ignore_index=True)

                        # Remove any duplicate columns that might have been created during concatenation
                        clean_all_metrics_df = clean_all_metrics_df.loc[:, ~clean_all_metrics_df.columns.duplicated()]

                        # Save the concatenated DataFrame
                        overall_summary_path = generate_metrics_summary_path(level='overall')
                        save_csv(clean_all_metrics_df, overall_summary_path)

                        # Analyzing the data
                        aggregate_and_save_dataframes(clean_all_metrics_df)
                        logging.info(f"All evaluation metrics concatenated and saved to {overall_summary_path}")
                        logging.info(f"Final DataFrame shape: {clean_all_metrics_df.shape}")
                    else:
                        logging.warning("No metric DataFrames to concatenate.")
                else:
                    logging.warning("No metrics to concatenate.")
            except Exception as e:
                logging.error(f"Could not concatenate all metrics: {e}")
                logging.error(f"Error type: {type(e).__name__}")
                if 'all_metrics_dfs' in locals():
                    for i, df in enumerate(all_metrics_dfs):
                        try:
                            # Log column names without duplicates
                            unique_cols = list(dict.fromkeys(df.columns))
                            logging.error(f"DataFrame {i} unique columns: {unique_cols}")
                            logging.error(f"DataFrame {i} shape: {df.shape}")
                        except Exception as inner_e:
                            logging.error(f"Error inspecting DataFrame {i}: {inner_e}")

            try:
                if misclassifications_list:
                    misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in
                                              misclass_dict.values()]
                    properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in
                                                 properclass_dict.values()]
                    if misclassifications_dfs:
                        misclassifications_df = pd.concat(misclassifications_dfs, ignore_index=True)
                        misclassification_path = generate_misclassification_path(status='raw')
                        save_csv(misclassifications_df, misclassification_path)

                        properclassifications_df = pd.concat(properclassifications_dfs, ignore_index=True)
                        properclassifications_path = generate_properclassification_path(status='raw')
                        save_csv(properclassifications_df, properclassifications_path)

                        misclassification_report_path = generate_misclassification_path(status='reports')
                        results = analyze_misclassification_errors(misclassification_path, 2)
                        export_results(results, misclassification_report_path)
                        logging.info("All misclassifications analyzed and saved.")
                    else:
                        logging.warning("No misclassification DataFrames to process.")
                else:
                    logging.warning("No misclassifications to process.")
            except Exception as e:
                logging.error(f"Error processing misclassifications: {e}")

    if run_json_parsing_error_analysis:
        if uncertainty:
            # Handle uncertainty error analysis if needed
            errors_df = aggregate_error_csvs(phenotypes)
            if not errors_df.empty:
                try:
                    errors_per_model, errors_per_model_and_phenotype, errors_per_config = generate_error_report(errors_df)
                    
                    # Save with uncertainty suffix
                    errors_per_model_path = generate_errors_report_path("errors_per_model_uncertainty")
                    errors_per_model_phenotype_path = generate_errors_report_path("errors_per_model_and_phenotype_uncertainty")
                    errors_per_config_path = generate_errors_report_path("errors_per_config_uncertainty")

                    save_csv(errors_per_model, errors_per_model_path)
                    save_csv(errors_per_model_and_phenotype, errors_per_model_phenotype_path)
                    save_csv(errors_per_config, errors_per_config_path)
                    logging.info("Uncertainty JSON parsing error report successfully saved")
                except Exception as e:
                    logging.error(f"Could not generate an uncertainty error report: {e}")
        else:
            errors_df = aggregate_error_csvs(phenotypes)
            if not errors_df.empty:
                try:
                    errors_per_model, errors_per_model_and_phenotype, errors_per_config = generate_error_report(errors_df)
                    errors_per_model_path = generate_errors_report_path("errors_per_model")
                    errors_per_model_phenotype_path = generate_errors_report_path("errors_per_model_and_phenotype")
                    errors_per_config_path = generate_errors_report_path("errors_per_config")

                    save_csv(errors_per_model, errors_per_model_path)
                    save_csv(errors_per_model_and_phenotype, errors_per_model_phenotype_path)
                    save_csv(errors_per_config, errors_per_config_path)
                    logging.info("JSON parsing error report successfully saved")

                except Exception as e:
                    logging.error(f"Could not generate an error report: {e}")


async def load_phenotype_predictions_async(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                                           show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    Async version of load_phenotype_predictions that uses batch processing for API calls.
    """
    if os.path.exists(metrics_pickle_path):
        try:
            with open(metrics_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
            logging.debug(f"Loaded phenotype metrics from {metrics_pickle_path}")
            return phenotype_predictions
        except Exception as e:
            logging.error(f"Failed to load phenotype metrics from {metrics_pickle_path}: {e}")

    logging.debug(f"Phenotype metrics not found at {metrics_pickle_path}. Running llm_parsing...")
    return await generate_predictions_async(
        phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
        show_config_bar, show_chunk_bar, reasoning, uncertainty, batch_size
    )


async def generate_predictions_async(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                                     show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    Async version of generate_predictions that uses batch processing for API calls.
    """
    phenotype_predictions = await llm_parsing_async(
        phenotype, model_name, configurations, cleaned_data,
        show_config_bar=show_config_bar, show_chunk_bar=show_chunk_bar,
        reasoning=reasoning, uncertainty=uncertainty, batch_size=batch_size
    )

    directory = os.path.dirname(metrics_pickle_path)
    os.makedirs(directory, exist_ok=True)
    try:
        with open(metrics_pickle_path, 'wb') as file:
            pickle.dump(phenotype_predictions, file)
        logging.info(f"Saved phenotype metrics to {metrics_pickle_path}")
    except Exception as e:
        logging.error(f"Failed to save phenotype metrics to {metrics_pickle_path}: {e}")

    return phenotype_predictions


async def llm_parsing_async(phenotype, model_name, configurations, cleaned_data,
                            show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    Async version of llm_parsing that uses batch processing for API calls.
    """
    phenotype_parsed = {}
    phenotype_preparsed = {}

    config_pbar = tqdm(
        total=len(configurations),
        desc=f"Configs for {phenotype} with {model_name}",
        unit="config", colour="green", position=1, leave=True, dynamic_ncols=True, miniters=1
    ) if show_config_bar else None

    try:
        for chunking_method, prompt_strategy, aggregation_method in configurations:
            if config_pbar:
                config_pbar.set_description(f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}")
                config_pbar.refresh()

            key = (chunking_method, prompt_strategy, phenotype, model_name)
            post_aggregation_key = (chunking_method, prompt_strategy, aggregation_method, phenotype, model_name)

            if key not in phenotype_preparsed:
                try:
                    try:
                        chunked_data = cleaned_data.chunk_text(method=chunking_method)
                        if chunked_data is None or chunked_data.empty:
                            logging.error(
                                f"No data was returned from chunk_text for method '{chunking_method}'. "
                                f"Skipping this config for {phenotype} with {model_name}."
                            )
                            continue
                    except (TypeError, KeyError) as e:
                        logging.error(f"Error while chunking data: {e}")
                        continue

                    prompt_engineering = PromptEngineering(prompt_type=prompt_strategy)
                    llm_parser = LLMParser(
                        model_name=model_name,
                        prompt_engineering=prompt_engineering,
                        api_url=OPENROUTER_URL,
                        api_key=OPENROUTER_API_KEY
                    )

                    parsing_bar = tqdm(
                        total=len(chunked_data),
                        desc="Parsing chunks",
                        unit="chunk",
                        leave=True,
                        colour="blue",
                        position=2
                    ) if show_chunk_bar else None

                    try:
                        if reasoning:
                            # Not implementing async for reasoning yet
                            pre_aggregation_results, errors_df = llm_parser.extract_then_phenotype_reasoning(
                                chunked_data, phenotype, parsing_bar
                            )
                        elif uncertainty:
                            pre_aggregation_results, errors_df = await llm_parser.parse_uncertainty_async(
                                chunked_data, phenotype, parsing_bar, batch_size
                            )
                        else:
                            if prompt_strategy != 'etp':
                                pre_aggregation_results, errors_df = await llm_parser.parse_async(
                                    chunked_data, phenotype, parsing_bar, batch_size
                                )
                            else:
                                pre_aggregation_results, errors_df = await llm_parser.extract_then_phenotype_async(
                                    chunked_data, phenotype, parsing_bar, batch_size
                                )

                        # Save results
                        if reasoning:
                            pre_agg_path = generate_parse_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )
                        else:
                            pre_agg_path = generate_parse_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )

                        phenotype_preparsed[key] = pre_aggregation_results
                        save_csv(pre_aggregation_results, pre_agg_path)
                        if errors_df is not None:
                            save_csv(errors_df, errors_path)

                    except Exception as e:
                        logging.error(f"Error during parsing: {e}")
                        traceback.print_exc()
                    finally:
                        if parsing_bar:
                            parsing_bar.close()

                except Exception as e:
                    logging.error(f"Error during processing: {e}")
                    traceback.print_exc()
                    continue

            if post_aggregation_key not in phenotype_parsed:
                try:
                    result_aggregator = ResultAggregator(aggregation_method=aggregation_method)
                    if aggregation_method == 'etp':
                        result_aggregator.set_phenotype(phenotype)
                        result_aggregator.set_etp_rule_type('fixed')
                        result_aggregator.set_etp_threshold(0.4)

                    final_results = result_aggregator.aggregate(phenotype_preparsed[key])
                    post_agg_path = generate_parse_path(
                        phenotype,
                        f"{chunking_method}_{aggregation_method}_{prompt_strategy}_{model_name}_post",
                        False
                    )
                    save_csv(final_results, post_agg_path)
                    phenotype_parsed[post_aggregation_key] = final_results

                except Exception as e:
                    logging.error(f"Error during aggregation: {e}")
                    traceback.print_exc()

            if config_pbar:
                config_pbar.update(1)

    except Exception as e:
        logging.error(f"An error occurred while processing {phenotype} with {model_name}: {e}")
        traceback.print_exc()

    finally:
        if config_pbar:
            config_pbar.close()

    return phenotype_parsed


def save_entropy_results(detailed_results, semi_aggregated_results):
    detailed_df = pd.DataFrame(detailed_results)

    # Filter predictions
    mask_etp = detailed_df['prompt_strategy'] != 'etp'
    filtered_df = pd.concat([
        detailed_df[~mask_etp],  # All rows where 'prompt_strategy' is 'etp'
        detailed_df[mask_etp & detailed_df['predictions'].apply(is_valid_prediction_list, args=(2,))]
    ])
    filtered_df.reset_index(drop=True, inplace=True)
    detailed_df = filtered_df

    # Create config aggregated DataFrame
    config_aggregated_df = pd.DataFrame(semi_aggregated_results)

    # Group by different combinations including temperature
    prompt_aggregated_df = config_aggregated_df.groupby(
        ['phenotype', 'model', 'temperature', 'prompt_strategy']
    ).agg({'mean_entropy': 'mean'}).reset_index()

    chunk_aggregated_df = config_aggregated_df.groupby(
        ['phenotype', 'model', 'temperature', 'chunking_method']
    ).agg({'mean_entropy': 'mean'}).reset_index()

    aggregation_aggregated_df = config_aggregated_df.groupby(
        ['phenotype', 'model', 'temperature', 'aggregation_method']
    ).agg({'mean_entropy': 'mean'}).reset_index()

    model_aggregated_df = config_aggregated_df.groupby(
        ['phenotype', 'model', 'temperature']
    ).agg({'mean_entropy': 'mean'}).reset_index()

    # Additional temperature-specific aggregation
    temperature_aggregated_df = config_aggregated_df.groupby(
        ['phenotype', 'temperature']
    ).agg({'mean_entropy': 'mean'}).reset_index()

    try:
        # Save all results
        save_csv(detailed_df, ENTROPY_REPORT_PATH + 'detailed_results.csv')

        # Save grouped reports
        save_csv(config_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'config_level_results.csv')
        save_csv(prompt_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'prompt_level_results.csv')
        save_csv(chunk_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'chunking_level_results.csv')
        save_csv(aggregation_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'aggregation_level_results.csv')
        save_csv(model_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'model_level_results.csv')
        save_csv(temperature_aggregated_df, ENTROPY_REPORT_PATH + 'grouped_reports/' + 'temperature_level_results.csv')

        logging.info("Results saved successfully")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


async def entropy_evaluation_async(phenotypes, models, chunking_methods, prompt_strategies, aggregation_methods, k=3,
                                   easy_path=EASY_PATH, hard_path=CHALLENGING_PATH, batch_size=20,
                                   max_retries=3, retry_delay=2):
    """
    Redesigned asynchronous entropy evaluation with a more sequential approach:
    1) Process one phenotype at a time
    2) For each phenotype, process each temperature sequentially
    3) Within each temperature, handle models and configurations in parallel with controlled concurrency
    4) Save results after each phenotype+temperature combination is complete

    The function avoids redundant processing by caching results for the same prompt-chunk configuration
    when testing different aggregation methods. It implements a two-level caching strategy:
    1) Text parsing cache (model and temperature independent)
    2) Prediction cache (model and temperature specific)

    Parameters:
        phenotypes (list): List of phenotypes to evaluate
        models (list): List of models to evaluate
        chunking_methods (list): List of chunking methods to use
        prompt_strategies (list): List of prompt strategies to use
        aggregation_methods (list): List of aggregation methods to use
        k (int): Number of repeated calls for entropy calculation
        easy_path (str): Path to the CSV file with easy cases
        hard_path (str): Path to the CSV file with hard cases
        batch_size (int): Maximum number of concurrent API calls
        max_retries (int): Maximum number of retry attempts for failed API calls
        retry_delay (int): Base delay in seconds between retry attempts (exponential backoff is applied)
    """
    from datetime import datetime
    import random
    import os

    start_time = datetime.now()
    logging.info(f"Starting sequential entropy evaluation with batch_size={batch_size}")

    # Create directories for results
    os.makedirs(ENTROPY_REPORT_PATH, exist_ok=True)
    os.makedirs(ENTROPY_REPORT_PATH + "interim_results", exist_ok=True)

    # Setup main progress bar
    main_pbar = tqdm(
        desc="Progress", 
        unit="phenotype",
        position=0, 
        leave=True, 
        colour="blue",
        total=len(phenotypes)
    )
    
    # Progress bar for data loading
    loading_pbar = tqdm(
        total=3,  # 3 steps: load CSVs, load data, clean data
        desc="Loading Data",
        unit="step",
        position=1,
        leave=False,
        colour="green"
    )

    # Load easy and hard CSV files
    loading_pbar.set_description("Loading classification CSVs")
    easy_df = pd.read_csv(easy_path)
    hard_df = pd.read_csv(hard_path)
    loading_pbar.update(1)

    # Load cleaned data
    loading_pbar.set_description("Loading data")
    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    loading_pbar.update(1)
    
    loading_pbar.set_description("Cleaning text data")
    cleaned_data = data.clean_text(expand_abbreviations=True)
    loading_pbar.update(1)
    loading_pbar.close()

    temperatures = [0.0, 0.5, 1.0]
    detailed_results = []
    semi_aggregated_results = []

    # Initialize all tracking counters
    total_tasks = 0
    completed_tasks = 0
    failed_tasks = 0
    cached_results = 0
    cached_parsed_text = 0  # Count for parsed text reuse
    retried_tasks = 0
    succeeded_retries = 0

    # Helper function to process a single row with chunking method 'none' with retry logic
    async def process_single_row(temp, phenotype, model, prompt_strategy, agg_method, row, k, easy_ids, hard_ids,
                                 semaphore, text_cache, prediction_cache, row_pbar=None, retry_count=0):
        nonlocal cached_results, cached_parsed_text, retried_tasks, succeeded_retries
        
        # Two-level caching:
        # 1. Text parsing cache (model and temperature independent)
        # 2. Prediction cache (model and temperature specific)
        
        # First check if we have already processed this exact configuration
        prediction_cache_key = (temp, model, phenotype, 'none', prompt_strategy, row['hadm_id'])
        
        if prediction_cache_key in prediction_cache:
            # Reuse exact prediction for this model/temp
            predictions = prediction_cache[prediction_cache_key]
            cached_results += 1
            if row_pbar:
                row_pbar.update(1)
            return prediction_cache_key, predictions
        
        # --- Simple Fix: Wrap the entire processing/retry logic for a single row ---
        try:
            # Text parsing cache key - independent of model and temperature
            text_cache_key = (phenotype, 'none', prompt_strategy, row['hadm_id'])

            # Create parser for this temperature and model
            parser = LLMParser(
                model_name=model,
                prompt_engineering=PromptEngineering(prompt_type=prompt_strategy),
                api_url=OPENROUTER_URL,
                api_key=OPENROUTER_API_KEY,
                temperature=temp
            )

            # Get text from row
            text = row['text']

            # Process with appropriate method based on prompt strategy
            SINGLE_API_CALL_TIMEOUT = 120 # Timeout for one API call attempt (seconds)
            async with semaphore:
                try:
                    if prompt_strategy == 'etp' and agg_method == 'etp':
                        if hasattr(parser, 'evaluate_consistency_for_etp_none_async'):
                            # Preferred async path
                            result = await asyncio.wait_for(
                                parser.evaluate_consistency_for_etp_none_async(
                                    text, phenotype, fixed_etp_threshold=0.4, k=k
                                ),
                                timeout=SINGLE_API_CALL_TIMEOUT
                            )
                        else:
                            # Sync fallback executed in a separate thread so the event loop stays responsive
                            logging.warning("Async method 'evaluate_consistency_for_etp_none_async' not found; executing sync version in thread – timeout still enforced.")
                            result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    parser.evaluate_consistency_for_etp_none,
                                    text, phenotype, fixed_etp_threshold=0.4, k=k
                                ),
                                timeout=SINGLE_API_CALL_TIMEOUT
                            )
                    else:
                        if hasattr(parser, 'evaluate_consistency_async'):
                            # Preferred async path
                            result = await asyncio.wait_for(
                                parser.evaluate_consistency_async(text, phenotype, k=k),
                                timeout=SINGLE_API_CALL_TIMEOUT
                            )
                        else:
                            # Sync fallback executed in a thread
                            logging.warning("Async method 'evaluate_consistency_async' not found; executing sync version in thread – timeout still enforced.")
                            result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    parser.evaluate_consistency,
                                    text, phenotype, k=k
                                ),
                                timeout=SINGLE_API_CALL_TIMEOUT
                            )
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout ({SINGLE_API_CALL_TIMEOUT}s) during single API call for {row['hadm_id']}. Will attempt retry.")
                    raise # Re-raise TimeoutError to be caught by the outer retry handler
                except Exception as api_err:
                     logging.warning(f"Error during single API call for {row['hadm_id']}: {api_err}. Will attempt retry.")
                     raise # Re-raise other API errors for retry handler

            # Save predictions for reuse (This part is now outside the async with semaphore)
            # The result variable is available here if no exception was raised above
            if result is not None:
                entropy, predictions = result
                if row_pbar:
                    row_pbar.update(1)

                # Save to both caches
                prediction_cache[prediction_cache_key] = predictions

                # Mark retry as succeeded if it was a retry
                if retry_count > 0:
                     succeeded_retries += 1
                return prediction_cache_key, predictions
            else:
                # Treat None result as an error condition for retry purposes
                raise Exception("Received None result from API or synchronous fallback")

        except Exception as e:
            # Handle retry logic here
            if retry_count < max_retries:
                # Increment retry counter
                if retry_count == 0: # Only count the first time it enters retry for a task
                     retried_tasks += 1

                # Calculate exponential backoff with jitter
                backoff_time = retry_delay * (2 ** retry_count) + random.uniform(0, 1)
                logging.warning(
                    f"Retry {retry_count + 1}/{max_retries} for {row['hadm_id']} ({phenotype}, {model}, {prompt_strategy}) due to: {e}. Waiting {backoff_time:.2f}s", exc_info=False # Keep log minimal
                )

                # Wait before retrying
                await asyncio.sleep(backoff_time)

                # Recursive retry
                return await process_single_row(temp, phenotype, model, prompt_strategy, agg_method,
                                               row, k, easy_ids, hard_ids, semaphore,
                                               text_cache, prediction_cache, row_pbar, retry_count + 1)
            else:
                # Max retries exceeded
                logging.error(f"Max retries exceeded for {row['hadm_id']} in {phenotype} with {model}, {prompt_strategy}. Last error: {e}", exc_info=False)
                if row_pbar:
                    row_pbar.update(1) # Ensure progress bar updates even on final failure
                return None, None # Return failure indicator

        # --- End Simple Fix Wrapper ---
        # This part should theoretically not be reached if the above logic is sound,
        # but added as a safeguard.
        if row_pbar:
            row_pbar.update(1)
        return None, None

    # Process one phenotype and temperature at a time
    async def process_phenotype_temperature(phenotype, temp, models, chunking_methods, prompt_strategies,
                                          aggregation_methods, preselected, easy_ids, hard_ids, k,
                                          text_cache, prediction_cache):

        nonlocal cached_results, cached_parsed_text, total_tasks, completed_tasks, failed_tasks, retried_tasks, succeeded_retries

        temp_pbar = tqdm(
            desc=f"Temperature {temp}",
            unit="config",
            position=1,
            leave=False,
            colour="cyan",
            total=len(models) * len(chunking_methods) * len(prompt_strategies) * len(aggregation_methods)
        )

        # Create semaphore for this temperature
        api_semaphore = asyncio.Semaphore(batch_size)

        local_results = []
        local_summaries = []

        # Process all configurations for this temperature
        config_tasks = []

        # Generate valid configurations first
        valid_configs = []
        for model in models:
            for chunk_method in chunking_methods:
                for prompt_strategy in prompt_strategies:
                    for agg_method in aggregation_methods:
                        # Skip invalid configurations
                        if (prompt_strategy == 'etp' and agg_method != 'etp') or \
                           (prompt_strategy != 'etp' and agg_method == 'etp'):
                            temp_pbar.update(1) # Update pbar for skipped configs
                            continue

                        valid_configs.append((model, chunk_method, prompt_strategy, agg_method))
                        # total_tasks += 1 # Total tasks is calculated globally now

        # Process each configuration
        for model, chunk_method, prompt_strategy, agg_method in valid_configs:
            # --- Add Timeout for the entire configuration processing --- 
            CONFIG_PROCESSING_TIMEOUT = 1800 # Timeout in seconds (e.g., 30 minutes)
            config_processed = False
            try:
                 await asyncio.wait_for(
                     # Inner function to process the config, allowing wait_for
                     process_single_config(phenotype, temp, model, chunk_method, prompt_strategy, agg_method,
                                              preselected, easy_ids, hard_ids, k, text_cache, prediction_cache,
                                              api_semaphore, local_results, local_summaries),
                     timeout=CONFIG_PROCESSING_TIMEOUT
                 )
                 config_processed = True # Mark as processed if no exception/timeout
            except asyncio.TimeoutError:
                 logging.error(f"TIMEOUT: Configuration ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}) exceeded {CONFIG_PROCESSING_TIMEOUT}s. Skipping.")
                 failed_tasks += 1 # Increment failed tasks for this configuration
                 # Continue to the next configuration (progress bar updated in finally)
            except Exception as outer_config_err: # Catch any unexpected errors not caught inside process_single_config
                 logging.error(f"UNEXPECTED OUTER ERROR processing config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}): {outer_config_err}", exc_info=True)
                 failed_tasks += 1
                 # Continue to the next configuration (progress bar updated in finally)
            finally:
                 # --- Ensure progress bar updates once per configuration --- 
                 temp_pbar.update(1)
            # --- End Timeout Wrapper ---

        temp_pbar.close()

        # Save results for this temperature
        try:
            interim_path = ENTROPY_REPORT_PATH + f"interim_results/{phenotype}_temp{temp}.pickle"
            with open(interim_path, 'wb') as f:
                pickle.dump({
                    'detailed': local_results,
                    'summary': local_summaries,
                    'timestamp': datetime.now().isoformat()
                }, f)

            logging.info(f"Saved results for {phenotype} at temperature {temp} to {interim_path}")
        except Exception as e:
            logging.error(f"Failed to save interim results: {e}")

        return local_results, local_summaries

    # --- Helper async function to process a single configuration ---
    async def process_single_config(phenotype, temp, model, chunk_method, prompt_strategy, agg_method,
                                     preselected, easy_ids, hard_ids, k, text_cache, prediction_cache,
                                     api_semaphore, local_results, local_summaries):
        nonlocal completed_tasks, failed_tasks # Allow modification
        # ENSURE THIS LINE IS PRESENT: Add the missing nonlocal declarations
        nonlocal cached_results, cached_parsed_text
        # This function contains the original try...except block for processing one config
        try:
            if chunk_method == 'none':
                # Process rows in parallel
                row_pbar = tqdm(
                    total=len(preselected),
                    desc=f"Rows ({model[:10]}..{prompt_strategy}..t{temp})",
                    position=2,
                    leave=False,
                    colour="yellow"
                )

                row_tasks = []
                for _, row in preselected.iterrows():
                    task = process_single_row(
                        temp, phenotype, model, prompt_strategy, agg_method,
                        row, k, easy_ids, hard_ids, api_semaphore,
                        text_cache, prediction_cache, row_pbar
                    )
                    row_tasks.append(task)

                # Execute row tasks with timeout
                ROW_PROCESSING_TIMEOUT = 600 # Timeout in seconds (e.g., 10 minutes)
                try:
                    results = await asyncio.wait_for(
                         asyncio.gather(*row_tasks, return_exceptions=True),
                         timeout=ROW_PROCESSING_TIMEOUT
                    )

                    # Process results
                    predictions_map = {}
                    successful_rows = 0
                    for i, result in enumerate(results):
                         row_hadm_id = preselected.iloc[i]['hadm_id'] # Get hadm_id for context
                         if isinstance(result, asyncio.TimeoutError):
                            logging.error(f"Timeout processing row {row_hadm_id} for {model}, {prompt_strategy}, temp {temp}")
                            continue
                         if isinstance(result, Exception):
                            logging.error(f"Row processing task for hadm_id {row_hadm_id} failed: {result}", exc_info=False) # Keep log minimal
                            continue

                         if result is not None and result[0] is not None:
                            cache_key, predictions = result
                            hadm_id = cache_key[-1]
                            predictions_map[hadm_id] = predictions
                            successful_rows += 1
                         else:
                            logging.warning(f"Row processing task for hadm_id {row_hadm_id} returned None.")

                    if not predictions_map: # If all rows failed or timed out
                        logging.warning(f"No successful rows processed for config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}). Skipping summary.")
                        failed_tasks += 1
                        row_pbar.close()
                        # Update main temp pbar here as we are skipping the rest of the config logic
                        # Find a way to pass temp_pbar or update globally if necessary
                        # Assuming global update for now:
                        # temp_pbar.update(1) # CANNOT DO THIS DIRECTLY
                        return # Exit this config processing


                    # Calculate entropy and prepare records
                    parser = LLMParser(
                        model_name=model,
                        prompt_engineering=PromptEngineering(prompt_type=prompt_strategy),
                        api_url=OPENROUTER_URL,
                        api_key=OPENROUTER_API_KEY,
                        temperature=temp
                    )

                    records = []
                    entropies = []

                    for hadm_id, preds in predictions_map.items():
                         try: # Add inner try-except for safety during calculation
                             entropy = parser._calculate_entropy(preds)
                             if entropy is None: # Skip if entropy calculation failed
                                  logging.warning(f"Entropy calculation failed for {hadm_id}, preds: {preds}")
                                  continue
                             record = {
                                 'hadm_id': hadm_id,
                                 'phenotype': phenotype,
                                 'model': model,
                                 'temperature': temp,
                                 'chunking_method': 'none',
                                 'prompt_strategy': prompt_strategy,
                                 'aggregation_method': agg_method,
                                 'entropy': entropy,
                                 'predictions': str(preds),
                                 'k': k,
                                 'difficulty': 'easy' if hadm_id in easy_ids else (
                                     'hard' if hadm_id in hard_ids else 'unknown')
                             }
                             records.append(record)
                             entropies.append(entropy)
                         except Exception as calc_err:
                             logging.error(f"Error calculating entropy for {hadm_id}: {calc_err}", exc_info=False)


                    # Create summary record only if entropies were calculated
                    if entropies:
                        mean_entropy = float(np.mean(entropies))
                        summary = {
                            'phenotype': phenotype,
                            'model': model,
                            'temperature': temp,
                            'chunking_method': 'none',
                            'prompt_strategy': prompt_strategy,
                            'aggregation_method': agg_method,
                            'mean_entropy': mean_entropy,
                            'k': k
                        }
                        local_summaries.append(summary)
                    else:
                        logging.warning(f"No valid entropies calculated for config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}). Skipping summary.")

                    # Add detailed results
                    local_results.extend(records)

                    # Update completed tasks count
                    completed_tasks += 1

                except asyncio.TimeoutError:
                    logging.error(f"Overall Row Processing Timeout ({ROW_PROCESSING_TIMEOUT}s) for {model}, {prompt_strategy}, temp {temp}")
                    failed_tasks += 1 # Mark the whole config as failed for this case
                except Exception as e:
                     logging.error(f"Error gathering rows for {model} with {prompt_strategy}, temp {temp}: {e}", exc_info=False)
                     failed_tasks += 1

                row_pbar.close()
            else:
                # For chunked data, process through the parser directly
                parser = LLMParser(
                    model_name=model,
                    prompt_engineering=PromptEngineering(prompt_type=prompt_strategy),
                    api_url=OPENROUTER_URL,
                    api_key=OPENROUTER_API_KEY,
                    temperature=temp
                )

                # Text cache key - independent of model and temperature
                # This allows reusing parsed/chunked text across models AND temperatures
                existing_results = {}
                uncached_hadm_ids = []

                # First check the prediction cache (model+temp specific)
                for hadm_id in preselected['hadm_id'].unique():
                    prediction_cache_key = (temp, model, phenotype, chunk_method, prompt_strategy, hadm_id)
                    if prediction_cache_key in prediction_cache:
                        existing_results[hadm_id] = prediction_cache[prediction_cache_key]
                        cached_results += 1
                    else:
                        # Now check the text cache for other models/temps
                        text_cache_key = (phenotype, chunk_method, prompt_strategy, hadm_id)
                        found = False

                        # Check if this hadm_id with same chunking/prompt is in any model/temp cache
                        for existing_temp in temperatures:
                            for existing_model in models:
                                existing_prediction_key = (existing_temp, existing_model, phenotype,
                                                       chunk_method, prompt_strategy, hadm_id)
                                if existing_prediction_key in prediction_cache:
                                    # We found it, but need to recalculate entropy for this model/temp
                                    existing_results[hadm_id] = prediction_cache[existing_prediction_key]
                                    cached_parsed_text += 1
                                    found = True
                                    break
                            if found:
                                break

                        if not found:
                            uncached_hadm_ids.append(hadm_id)

                # Only process hadm_ids that aren't in either cache
                new_results = existing_results # Start with cached results
                if uncached_hadm_ids:
                    # Some or all rows need processing
                    uncached_data = preselected[preselected['hadm_id'].isin(uncached_hadm_ids)]

                    if not uncached_data.empty:
                         # --- Modified CHUNK_PROCESSING_TIMEOUT ---
                         if model == 'llama_70b' and chunk_method == 'document':
                             CHUNK_PROCESSING_TIMEOUT = 900 # Timeout for llama_70b with document (15 mins)
                             logging.info(f"Using extended CHUNK_PROCESSING_TIMEOUT of {CHUNK_PROCESSING_TIMEOUT}s for llama_70b + document.")
                         else:
                             CHUNK_PROCESSING_TIMEOUT = 180 # Timeout for other chunked data (3 mins)
                         # --- End Modified CHUNK_PROCESSING_TIMEOUT ---
                         try:
                            async with api_semaphore:
                                 if hasattr(parser, 'evaluate_consistency_for_chunked_data_async'):
                                     # Add timeout to the async call if possible, or wrap with wait_for
                                     new_uncached_results = await asyncio.wait_for(
                                            parser.evaluate_consistency_for_chunked_data_async(
                                                uncached_data, phenotype, model, chunk_method, prompt_strategy, agg_method, k
                                            ),
                                            timeout=CHUNK_PROCESSING_TIMEOUT
                                        )
                                 else:
                                     # Fallback to synchronous version – execute in thread so timeout/cancellation works
                                     logging.warning("Async method 'evaluate_consistency_for_chunked_data_async' not found; executing sync version in thread – timeout still enforced.")
                                     new_uncached_results = await asyncio.wait_for(
                                         asyncio.to_thread(
                                             parser.evaluate_consistency_for_chunked_data,
                                             uncached_data, phenotype, model, chunk_method, prompt_strategy, agg_method, k
                                         ),
                                         timeout=CHUNK_PROCESSING_TIMEOUT
                                     )

                                 # Combine with existing results
                                 new_results = {**existing_results, **new_uncached_results}

                                 # Cache new results - both caches
                                 for hadm_id, preds in new_uncached_results.items():
                                     # Add to prediction cache (model+temp specific)
                                     prediction_cache_key = (temp, model, phenotype, chunk_method, prompt_strategy, hadm_id)
                                     prediction_cache[prediction_cache_key] = preds
                         except asyncio.TimeoutError:
                             logging.error(f"Timeout ({CHUNK_PROCESSING_TIMEOUT}s) processing chunked data for {model}, {chunk_method}, {prompt_strategy}, temp {temp}")
                             # Continue with only existing_results if any
                             new_results = existing_results # Ensure we only use cached results if timeout occurred
                         except Exception as chunk_err:
                             logging.error(f"Error during chunked data processing for config ({model}, {chunk_method}, {prompt_strategy}, {agg_method}): {chunk_err}", exc_info=False)
                             # Continue with only existing_results if any
                             new_results = existing_results # Ensure we only use cached results if error occurred

                if not new_results: # Check if after potential failures, we have anything to process
                    logging.warning(f"No results (neither cached nor new) for chunked config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}). Skipping.")
                    failed_tasks += 1
                    # Update main temp pbar here as we are skipping the rest of the config logic
                    # temp_pbar.update(1) # CANNOT DO THIS DIRECTLY
                    return

                # Process results (even if only cached ones)
                records = []
                entropies = []

                for hadm_id, preds in new_results.items():
                     try: # Add inner try-except for safety
                         # Calculate entropy (model-specific calculation)
                         entropy = parser._calculate_entropy(preds)
                         if entropy is None:
                             logging.warning(f"Entropy calculation failed for chunked {hadm_id}, preds: {preds}")
                             continue
                         record = {
                             'hadm_id': hadm_id,
                             'phenotype': phenotype,
                             'model': model,
                             'temperature': temp,
                             'chunking_method': chunk_method,
                             'prompt_strategy': prompt_strategy,
                             'aggregation_method': agg_method,
                             'entropy': entropy,
                             'predictions': str(preds),
                             'k': k,
                             'difficulty': 'easy' if hadm_id in easy_ids else (
                                 'hard' if hadm_id in hard_ids else 'unknown')
                         }
                         records.append(record)
                         entropies.append(entropy)
                     except Exception as calc_err:
                         logging.error(f"Error calculating chunked entropy for {hadm_id}: {calc_err}", exc_info=False)

                # Create summary record only if entropies were calculated
                if entropies:
                    mean_entropy = float(np.mean(entropies))
                    summary = {
                        'phenotype': phenotype,
                        'model': model,
                        'temperature': temp,
                        'chunking_method': chunk_method,
                        'prompt_strategy': prompt_strategy,
                        'aggregation_method': agg_method,
                        'mean_entropy': mean_entropy,
                        'k': k
                    }
                    local_summaries.append(summary)
                else:
                    logging.warning(f"No valid entropies calculated for chunked config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}). Skipping summary.")


                # Add detailed results
                local_results.extend(records)

                # Update counters
                completed_tasks += 1

        # --- End Original Config Processing Logic ---
        except Exception as config_err:
            logging.error(f"FATAL: Uncaught exception processing config ({phenotype}, {temp}, {model}, {chunk_method}, {prompt_strategy}, {agg_method}): {config_err}", exc_info=True) # Log full traceback here
            failed_tasks += 1 # Increment failed tasks for this configuration
            # Continue to the next configuration in the loop (handled by the outer loop)
        finally:
             # Ensure progress bar updates regardless of success/failure within this config
             # temp_pbar.update(1) # CANNOT UPDATE temp_pbar here
             # Need to find a way to signal update to the outer loop or update globally
             pass # Update happens in the outer loop's exception handling or after successful await

    # Main execution function
    async def process_all_phenotypes():
        nonlocal detailed_results, semi_aggregated_results, total_tasks # temp_pbar is local to this function

        # Calculate total tasks based on valid configurations
        total_valid_configs = 0
        for phenotype in phenotypes:
            for temp in temperatures:
                for model in models:
                    for chunk_method in chunking_methods:
                        for prompt_strategy in prompt_strategies:
                            for agg_method in aggregation_methods:
                                # Skip invalid configurations
                                if (prompt_strategy == 'etp' and agg_method != 'etp') or \
                                  (prompt_strategy != 'etp' and agg_method == 'etp'):
                                    continue
                                total_valid_configs += 1

        # Set total tasks
        total_tasks = total_valid_configs
        logging.info(f"Total valid configurations to process: {total_tasks}")

        # Two-level cache system
        # 1. Text cache - shared across models AND temperatures (stores parsed/chunked text)
        # 2. Prediction cache - specific to model and temperature (stores predictions)
        text_cache = {}
        prediction_cache = {}

        # Process each phenotype
        for phenotype in phenotypes:
            logging.info(f"Processing phenotype: {phenotype}")

            # Preselect data for this phenotype
            preselected, easy_ids, hard_ids = preselect_data(phenotype, cleaned_data, easy_df, hard_df)
            if preselected.empty:
                logging.warning(f"No data selected for phenotype {phenotype}")
                main_pbar.update(1)
                continue

            # Process each temperature sequentially
            phenotype_results = []
            phenotype_summaries = []

            temp_pbar = tqdm(
                total=len(temperatures),
                desc=f"Temperature for {phenotype}",
                position=1,
                leave=False,
                colour="magenta"
            )

            for temp in temperatures:
                # --- Add Resume Logic --- 
                interim_path = ENTROPY_REPORT_PATH + f"interim_results/{phenotype}_temp{temp}.pickle"
                # --- Add Debug Logging --- 
                logging.info(f"Checking for interim file at: {interim_path}")
                file_exists = os.path.exists(interim_path)
                logging.info(f"Result of os.path.exists: {file_exists}")
                # --- End Debug Logging ---
                if file_exists:
                    try:
                         logging.info(f"Loading existing results for {phenotype}, temp {temp} from {interim_path}")
                         with open(interim_path, 'rb') as f:
                             loaded_data = pickle.load(f)
                         phenotype_results.extend(loaded_data.get('detailed', []))
                         phenotype_summaries.extend(loaded_data.get('summary', []))
                         temp_pbar.update(1)
                         # Log skipping
                         logging.info(f"SUCCESS: Loaded and SKIPPING processing for {phenotype}, temp {temp}.")
                         # Skip processing this temp since results were loaded
                         # Need to adjust completed_tasks count if we are tracking based on config
                         # For simplicity here, we assume progress is based on phenotype/temp level
                         continue 
                    except Exception as load_err:
                         logging.error(f"Failed to load interim results from {interim_path}: {load_err}. Reprocessing.")
                         # If loading fails, fall through to reprocessing
                else:
                    # Log that the file was not found and processing will occur
                    logging.info(f"No existing interim file found at {interim_path}. Starting processing for {phenotype}, temp {temp}.")
                # --- End Resume Logic ---

                # Process this temperature if interim file didn't exist or failed to load
                results, summaries = await process_phenotype_temperature(
                    phenotype, temp, models, chunking_methods, prompt_strategies,
                    aggregation_methods, preselected, easy_ids, hard_ids, k,
                    text_cache, prediction_cache
                )

                phenotype_results.extend(results)
                phenotype_summaries.extend(summaries)
                temp_pbar.update(1)

                # Force garbage collection
                gc.collect()

            temp_pbar.close()

            # Add results to overall collections
            detailed_results.extend(phenotype_results)
            semi_aggregated_results.extend(phenotype_summaries)

            # Save interim results after each phenotype (This saving part already exists)
            try:
                interim_path_phenotype = ENTROPY_REPORT_PATH + f"interim_results/{phenotype}_complete.pickle"
                with open(interim_path_phenotype, 'wb') as f:
                    pickle.dump({
                        'detailed': phenotype_results,
                        'summary': phenotype_summaries,
                        'timestamp': datetime.now().isoformat()
                    }, f)

                logging.info(f"Saved complete results for {phenotype} to {interim_path_phenotype}")
            except Exception as e:
                logging.error(f"Failed to save phenotype results: {e}")

            # Update main progress bar
            main_pbar.update(1)

            # Clear memory for this phenotype
            phenotype_results = None
            phenotype_summaries = None
            gc.collect()

        main_pbar.close()
        return detailed_results, semi_aggregated_results

    # Function to preselect data for a phenotype
    def preselect_data(phenotype, cleaned_data, easy_df, hard_df):
        easy_ids = set(easy_df[(easy_df['both_models_correct'] == True) & 
                             (easy_df['phenotype'] == phenotype)]['hadm_id'].unique())
        hard_ids = set(hard_df[(hard_df['both_models_misclassified'] == True) & 
                              (hard_df['phenotype'] == phenotype)]['hadm_id'].unique())
        
        # Take first 10 from each if available (adjust as needed)
        easy_ids_limited = list(easy_ids)[:10]
        hard_ids_limited = list(hard_ids)[:10]
        combined_ids = easy_ids_limited + hard_ids_limited
        
        # Filter data
        if combined_ids:
            filtered_data = cleaned_data[cleaned_data['hadm_id'].isin(combined_ids)]
        else:
            filtered_data = cleaned_data
        
        # Select sample rows (adjust sample size as needed)
        preselected = filtered_data.sample(min(20, len(filtered_data)),
                                         random_state=42) if not filtered_data.empty else filtered_data
        
        return preselected, easy_ids_limited, hard_ids_limited

    # Main execution
    try:
        logging.info("Starting sequential processing of phenotypes and temperatures")
        
        # Process all phenotypes
        detailed_results, semi_aggregated_results = await process_all_phenotypes()
        
        # Save final results
        saving_pbar = tqdm(total=1, desc="Saving Results", position=0, leave=True, colour="green")
        save_entropy_results(detailed_results, semi_aggregated_results)
        saving_pbar.update(1)
        saving_pbar.close()
        
        duration = datetime.now() - start_time
        logging.info(f"Sequential entropy evaluation completed in {duration}")
        
        # Final summary
        print("\n===== Execution Summary =====")
        print(f"Total tasks: {total_tasks}")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Failed tasks: {failed_tasks}")
        print(f"Cached results (exact matches): {cached_results}")
        print(f"Cached parsed text (reused across models/temps): {cached_parsed_text}")
        print(f"Total retries: {retried_tasks}")
        print(f"Successful retries: {succeeded_retries}")
        print(f"Total time: {duration}")
        print("============================\n")
        
        return detailed_results, semi_aggregated_results
    
    except Exception as e:
        logging.error(f"Error in entropy evaluation: {e}")
        traceback.print_exc()
        return [], []


def entropy_evaluation(phenotypes, models, chunking_methods, prompt_strategies, aggregation_methods, k=3):
    logging.info("Starting consistency evaluation")
    # Initialize lists to store detailed and semi-aggregated results

    detailed_results = []
    semi_aggregated_results = []

    # Temperatures to evaluates
    temperatures = [0.0, 0.5, 1.0]

    # Load easy and hard CSV files
    easy_df = pd.read_csv(EASY_PATH)
    hard_df = pd.read_csv(CHALLENGING_PATH)

    # LOADING LABELED DATA
    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    cleaned_data = data.clean_text(expand_abbreviations=True)

    # Main progress bar for temperatures
    temp_pbar = tqdm(total=len(temperatures), desc="Temperature Progress", position=0, leave=True,
                     dynamic_ncols=True, colour='red')

    def preselect_data(phenotype, cleaned_data, easy_df, hard_df):
        easy_ids = set(easy_df[(easy_df['both_models_correct'] == True) &
                               (easy_df['phenotype'] == phenotype)]['hadm_id'].unique())
        hard_ids = set(hard_df[(hard_df['both_models_misclassified'] == True) &
                               (hard_df['phenotype'] == phenotype)]['hadm_id'].unique())

        # Take first 10 from each if available (adjust as needed)
        easy_ids_limited = list(easy_ids)[:10]
        hard_ids_limited = list(hard_ids)[:10]
        combined_ids = easy_ids_limited + hard_ids_limited

        # Filter data
        if combined_ids:
            filtered_data = cleaned_data[cleaned_data['hadm_id'].isin(combined_ids)]
        else:
            filtered_data = cleaned_data

        # Select sample rows (adjust sample size as needed)
        preselected = filtered_data.sample(min(20, len(filtered_data)),
                                           random_state=42) if not filtered_data.empty else filtered_data

        return preselected, easy_ids_limited, hard_ids_limited

    for temperature in temperatures:
        # Original progress bars and logic
        for phenotype in phenotypes:
            phenotype_pbar = tqdm(total=len(models), desc=f"Phenotype: {phenotype}", position=1, leave=True,
                                  dynamic_ncols=True, colour='cyan')

            # Preselect data for this phenotype
            preselected, easy_ids, hard_ids = preselect_data(phenotype, cleaned_data, easy_df, hard_df)
            if preselected.empty:
                logging.warning(f"No data selected for phenotype {phenotype}")
                continue

            for model in models:
                temp_pbar.set_postfix_str(f"Temp: {temperature}, Processing {phenotype} with {model}", refresh=True)
                model_pbar = tqdm(total=len(chunking_methods) * len(prompt_strategies) * len(aggregation_methods),
                                  desc=f"Model: {model}", position=2, leave=False, dynamic_ncols=True, colour='yellow')

                for chunking_method in chunking_methods:
                    for prompt_strategy in prompt_strategies:
                        for aggregation_method in aggregation_methods:
                            # Skip invalid combinations
                            if (prompt_strategy == 'etp' and aggregation_method != 'etp') or \
                                    (prompt_strategy != 'etp' and aggregation_method == 'etp'):
                                model_pbar.update(1)
                                continue

                            config_description = f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}"
                            model_pbar.set_description(config_description)

                            try:
                                # Initialize parser with current temperature
                                parser = LLMParser(model_name=model,
                                                   prompt_engineering=PromptEngineering(prompt_type=prompt_strategy),
                                                   api_url=OPENROUTER_URL,
                                                   api_key=OPENROUTER_API_KEY,
                                                   temperature=temperature)

                                # Original logic for non-chunked data
                                if chunking_method == 'none':
                                    if prompt_strategy == 'etp' and aggregation_method == 'etp':
                                        for _, row in preselected.iterrows():
                                            entropy, predictions = parser.evaluate_consistency_for_etp_none(
                                                row['text'], phenotype, fixed_etp_threshold=0.4, k=k)

                                            detailed_results.append({
                                                'hadm_id': row['hadm_id'],
                                                'phenotype': phenotype,
                                                'model': model,
                                                'temperature': temperature,
                                                'chunking_method': chunking_method,
                                                'prompt_strategy': prompt_strategy,
                                                'aggregation_method': aggregation_method,
                                                'entropy': entropy,
                                                'predictions': str(predictions),
                                                'k': k
                                            })
                                    else:
                                        for _, row in preselected.iterrows():
                                            entropy, predictions = parser.evaluate_consistency(row['text'], phenotype,
                                                                                               k=k)

                                            detailed_results.append({
                                                'hadm_id': row['hadm_id'],
                                                'phenotype': phenotype,
                                                'model': model,
                                                'temperature': temperature,
                                                'chunking_method': chunking_method,
                                                'prompt_strategy': prompt_strategy,
                                                'aggregation_method': aggregation_method,
                                                'entropy': entropy,
                                                'predictions': str(predictions),
                                                'k': k
                                            })
                                else:
                                    # Original logic for chunked data
                                    predictions_by_hadm = parser.evaluate_consistency_for_chunked_data(
                                        preselected, phenotype, model, chunking_method,
                                        prompt_strategy, aggregation_method, k)

                                    for hadm_id, preds in predictions_by_hadm.items():
                                        entropy = parser._calculate_entropy(preds)
                                        detailed_results.append({
                                            'hadm_id': hadm_id,
                                            'phenotype': phenotype,
                                            'model': model,
                                            'temperature': temperature,
                                            'chunking_method': chunking_method,
                                            'prompt_strategy': prompt_strategy,
                                            'aggregation_method': aggregation_method,
                                            'entropy': entropy,
                                            'predictions': str(preds),
                                            'k': k
                                        })

                                # Calculate mean entropy for this configuration
                                mean_entropy = np.mean([r['entropy'] for r in detailed_results
                                                        if r['phenotype'] == phenotype
                                                        and r['model'] == model
                                                        and r['temperature'] == temperature
                                                        and r['chunking_method'] == chunking_method
                                                        and r['prompt_strategy'] == prompt_strategy
                                                        and r['aggregation_method'] == aggregation_method])

                                semi_aggregated_results.append({
                                    'phenotype': phenotype,
                                    'model': model,
                                    'temperature': temperature,
                                    'chunking_method': chunking_method,
                                    'prompt_strategy': prompt_strategy,
                                    'aggregation_method': aggregation_method,
                                    'mean_entropy': mean_entropy,
                                    'k': k
                                })

                            except Exception as e:
                                logging.error(f"Error evaluating config {config_description}: {e}")
                            finally:
                                model_pbar.update(1)

                phenotype_pbar.update(1)
                model_pbar.close()

            phenotype_pbar.close()
        temp_pbar.update(1)

    temp_pbar.close()
    save_entropy_results(detailed_results, semi_aggregated_results)


async def llm_parsing_huggingface_async(phenotype, model_name, configurations, cleaned_data,
                                 show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    A version of llm_parsing_async that uses the Hugging Face API for OpenBioLLM models.
    Compatible with OpenBioLLM-8B and OpenBioLLM-70B via Hugging Face inference provider API.
    """
    phenotype_parsed = {}
    phenotype_preparsed = {}

    config_pbar = tqdm(
        total=len(configurations),
        desc=f"Configs for {phenotype} with {model_name}",
        unit="config", colour="green", position=1, leave=True, dynamic_ncols=True, miniters=1
    ) if show_config_bar else None

    try:
        for chunking_method, prompt_strategy, aggregation_method in configurations:
            if config_pbar:
                config_pbar.set_description(f"Config: {chunking_method}-{prompt_strategy}-{aggregation_method}")
                config_pbar.refresh()

            key = (chunking_method, prompt_strategy, phenotype, model_name)
            post_aggregation_key = (chunking_method, prompt_strategy, aggregation_method, phenotype, model_name)

            if key not in phenotype_preparsed:
                try:
                    try:
                        chunked_data = cleaned_data.chunk_text(method=chunking_method)
                        if chunked_data is None or chunked_data.empty:
                            logging.error(
                                f"No data was returned from chunk_text for method '{chunking_method}'. "
                                f"Skipping this config for {phenotype} with {model_name}."
                            )
                            continue
                    except (TypeError, KeyError) as e:
                        logging.error(f"Error while chunking data: {e}")
                        continue

                    prompt_engineering = PromptEngineering(prompt_type=prompt_strategy)
                    llm_parser = LLMParser(
                        model_name=model_name,
                        prompt_engineering=prompt_engineering,
                        api_url=HUGGINGFACE_URL,  # Use Hugging Face URL
                        api_key=HUGGINGFACE_API_KEY  # Use Hugging Face API key
                    )

                    parsing_bar = tqdm(
                        total=len(chunked_data),
                        desc="Parsing chunks",
                        unit="chunk",
                        leave=True,
                        colour="blue",
                        position=2
                    ) if show_chunk_bar else None

                    try:
                        if reasoning:
                            # Not implementing async for reasoning yet
                            pre_aggregation_results, errors_df = llm_parser.extract_then_phenotype_reasoning(
                                chunked_data, phenotype, parsing_bar
                            )
                        elif uncertainty:
                            pre_aggregation_results, errors_df = await llm_parser.parse_uncertainty_async(
                                chunked_data, phenotype, parsing_bar, batch_size
                            )
                        else:
                            if prompt_strategy != 'etp':
                                pre_aggregation_results, errors_df = await llm_parser.parse_async(
                                    chunked_data, phenotype, parsing_bar, batch_size
                                )
                            else:
                                pre_aggregation_results, errors_df = await llm_parser.extract_then_phenotype_async(
                                    chunked_data, phenotype, parsing_bar, batch_size
                                )

                        # Save results
                        if reasoning:
                            pre_agg_path = generate_parse_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_reasoning_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )
                        else:
                            pre_agg_path = generate_parse_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}", True
                            )
                            errors_path = generate_errors_path(
                                phenotype, f"{chunking_method}_{prompt_strategy}_{model_name}_errors"
                            )

                        phenotype_preparsed[key] = pre_aggregation_results
                        save_csv(pre_aggregation_results, pre_agg_path)
                        if errors_df is not None:
                            save_csv(errors_df, errors_path)

                    except Exception as e:
                        logging.error(f"Error during parsing: {e}")
                        traceback.print_exc()
                    finally:
                        if parsing_bar:
                            parsing_bar.close()

                except Exception as e:
                    logging.error(f"Error during processing: {e}")
                    traceback.print_exc()
                    continue

            if post_aggregation_key not in phenotype_parsed:
                try:
                    result_aggregator = ResultAggregator(aggregation_method=aggregation_method)
                    if aggregation_method == 'etp':
                        result_aggregator.set_phenotype(phenotype)
                        result_aggregator.set_etp_rule_type('fixed')
                        result_aggregator.set_etp_threshold(0.4)

                    final_results = result_aggregator.aggregate(phenotype_preparsed[key])
                    post_agg_path = generate_parse_path(
                        phenotype,
                        f"{chunking_method}_{aggregation_method}_{prompt_strategy}_{model_name}_post",
                        False
                    )
                    save_csv(final_results, post_agg_path)
                    phenotype_parsed[post_aggregation_key] = final_results

                except Exception as e:
                    logging.error(f"Error during aggregation: {e}")
                    traceback.print_exc()

            if config_pbar:
                config_pbar.update(1)

    except Exception as e:
        logging.error(f"An error occurred while processing {phenotype} with {model_name}: {e}")
        traceback.print_exc()

    finally:
        if config_pbar:
            config_pbar.close()

    return phenotype_parsed


async def llm_phenotyping_huggingface_async(run_parsing=True, run_evaluation=True, run_json_parsing_error_analysis=True,
                                    show_overall_bar=True, show_config_bar=True, show_chunk_bar=True,
                                    phenotypes=PHENOTYPES, models=MODELS_OPENBIO, chunking_methods=CHUNKING_METHODS,
                                    prompt_strategies=PROMPT_STRATEGIES, aggregation_methods=AGGREGATION_METHODS,
                                    batch_size=100, uncertainty=False):
    """
    A version of llm_phenotyping_async specifically for Hugging Face models like OpenBioLLM.
    Uses the Hugging Face inference API instead of OpenRouter.
    
    Parameters:
        models: Should be from MODELS_OPENBIO like 'openbio_8b' or 'openbio_70b'
        Other parameters are the same as llm_phenotyping_async
    """
    fs = setup_gcs_access()

    chunking_methods = chunking_methods
    prompt_strategies = prompt_strategies
    aggregation_methods = aggregation_methods

    all_combinations = product(chunking_methods, prompt_strategies, aggregation_methods)

    configurations = [
        (chunking, prompt, aggregation)
        for chunking, prompt, aggregation in all_combinations
        if (prompt == 'etp' and aggregation == 'etp') or
           (chunking == 'none' and prompt != 'etp' and aggregation == 'any') or
           (chunking == 'none' and prompt == 'etp' and aggregation == 'etp') or
           (chunking != 'none' and prompt != 'etp' and aggregation != 'etp')
    ]

    phenotypes = phenotypes
    models = models

    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    cleaned_data = data.clean_text(expand_abbreviations=True)
    total_discharge_summary = len(data)

    evaluator = Evaluator(GOLDEN_LABELS_PATH)

    # Create dictionaries to collect results for all phenotype-model combinations
    all_metrics = {}
    misclassifications_concat = {}
    properclassifications_concat = {}

    # For uncertainty mode, create additional dictionaries to collect results by confidence level
    if uncertainty:
        confidence_levels = ['all', 'high', 'medium', 'low']
        # Use a nested defaultdict to store metrics by confidence level, then by phenotype-model key
        all_metrics_by_confidence = {level: {} for level in confidence_levels}
        misclassifications_by_confidence = {level: {} for level in confidence_levels}
        properclassifications_by_confidence = {level: {} for level in confidence_levels}

    overall_iterator = list(product(phenotypes, models))
    overall_pbar = tqdm(total=len(overall_iterator), desc="Overall Progress", unit="task", colour="red",
                        position=0, leave=True, dynamic_ncols=True, miniters=1) if show_overall_bar else None

    for phenotype, model_name in overall_iterator:
        if show_overall_bar:
            overall_pbar.set_postfix_str(f"Processing {phenotype} with {model_name}", refresh=True)

        key = (phenotype, model_name)
        parsed_pickle_path = f"{PICKLE_PATH}{phenotype}_{model_name}_parsed.pkl"

        # Parsing all notes and predicting phenotypes according to each configuration
        if run_parsing:
            # Use the Hugging Face version of load_phenotype_predictions_async
            phenotype_predictions = await load_phenotype_predictions_huggingface_async(
                phenotype, model_name, configurations, cleaned_data,
                parsed_pickle_path, show_config_bar, show_chunk_bar,
                batch_size=batch_size, uncertainty=uncertainty
            )

            phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)
        else:
            with open(parsed_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
                phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)

        # Evaluating all predictions and saving metrics
        if run_evaluation:
            if uncertainty:
                # Process data separately for each confidence level
                for confidence_level in confidence_levels:
                    # Filter phenotype_predictions by confidence level
                    filtered_predictions = {}
                    
                    for config_key, predictions_df in phenotype_predictions.items():
                        if 'confidence' in predictions_df.columns:
                            if confidence_level == 'all':
                                # Use all predictions
                                filtered_predictions[config_key] = predictions_df
                            else:
                                # Filter by specific confidence level
                                filtered_df = predictions_df[predictions_df['confidence'] == confidence_level].copy()
                                if not filtered_df.empty:
                                    filtered_predictions[config_key] = filtered_df
                        else:
                            # If confidence column is missing, only include in 'all' category
                            if confidence_level == 'all':
                                filtered_predictions[config_key] = predictions_df
                            logging.warning(f"Confidence column missing in predictions for {config_key}")
                    
                    # Skip evaluation if no predictions for this confidence level
                    if not filtered_predictions:
                        logging.info(f"No predictions found for {phenotype}, {model_name} with confidence level: {confidence_level}")
                        continue
                    
                    # Run evaluation for this confidence level
                    metrics_df, misclass_df, properclass_df = evaluation(
                        phenotype, model_name, configurations, evaluator, filtered_predictions
                    )
                    
                    # Add confidence level column to results
                    for k, df in metrics_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    for k, df in misclass_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    for k, df in properclass_df.items():
                        if isinstance(df, pd.DataFrame):
                            df['confidence_level'] = confidence_level
                    
                    # Store results for this confidence level in the global dictionaries
                    all_metrics_by_confidence[confidence_level][key] = metrics_df
                    misclassifications_by_confidence[confidence_level][key] = misclass_df
                    properclassifications_by_confidence[confidence_level][key] = properclass_df
            else:
                # Standard evaluation without confidence levels
                clean_phenotype_metrics_df, misclassifications_df, properclassifications_df = evaluation(
                    phenotype, model_name, configurations, evaluator, phenotype_predictions
                )

                all_metrics[key] = clean_phenotype_metrics_df
                misclassifications_concat[key] = misclassifications_df
                properclassifications_concat[key] = properclassifications_df

        if show_overall_bar:
            overall_pbar.update(1)

    if show_overall_bar:
        overall_pbar.close()

    # After processing all phenotype-model combinations, process and save the final results
    if run_evaluation:
        if uncertainty:
            # Process and combine results from all confidence levels
            all_metrics_list = []
            misclassifications_list = []
            properclassifications_list = []
            
            # Combine results from all confidence levels
            for confidence_level in confidence_levels:
                for key, metrics_df in all_metrics_by_confidence[confidence_level].items():
                    all_metrics_list.append(metrics_df)
                    
                for key, misclass_df in misclassifications_by_confidence[confidence_level].items():
                    misclassifications_list.append(misclass_df)
                    
                for key, properclass_df in properclassifications_by_confidence[confidence_level].items():
                    properclassifications_list.append(properclass_df)
            
            try:
                if all_metrics_list:
                    # Process metrics with confidence level
                    all_metrics_dfs = []
                    
                    for metric_dict in all_metrics_list:
                        for column in metric_dict.values():
                            if isinstance(column, pd.DataFrame):
                                # Remove duplicate columns
                                column = column.loc[:, ~column.columns.duplicated()]
                                
                                # Reset index for clean concatenation
                                df_reset = column.reset_index(drop=True)
                                
                                # Add to list
                                all_metrics_dfs.append(df_reset)
                            else:
                                logging.warning(f"Skipping non-DataFrame object: {type(column)}")
                    
                    if all_metrics_dfs:
                        # Concatenate with clean indices
                        clean_all_metrics_df = pd.concat(all_metrics_dfs, axis=0, ignore_index=True)
                        
                        # Remove any duplicate columns
                        clean_all_metrics_df = clean_all_metrics_df.loc[:, ~clean_all_metrics_df.columns.duplicated()]
                        
                        # Save the consolidated results for all confidence levels
                        overall_summary_path = generate_metrics_summary_path(level='overall_confidence_hf')
                        save_csv(clean_all_metrics_df, overall_summary_path)
                        
                        # Also save individual confidence level files
                        for confidence_level in confidence_levels:
                            level_df = clean_all_metrics_df[clean_all_metrics_df['confidence_level'] == confidence_level]
                            if not level_df.empty:
                                level_path = generate_metrics_summary_path(level=f'confidence_{confidence_level}_hf')
                                save_csv(level_df, level_path)
                        
                        # Analyzing the data - with confidence grouping
                        aggregate_and_save_dataframes(clean_all_metrics_df, 
                                                     output_dir='output/classic/metrics/analysis_uncertainty_hf')
                        
                        logging.info(f"All uncertainty-based evaluation metrics saved to {overall_summary_path}")
                        logging.info(f"Final DataFrame shape: {clean_all_metrics_df.shape}")
                    else:
                        logging.warning("No uncertainty metric DataFrames to concatenate.")
                else:
                    logging.warning("No uncertainty metrics to concatenate.")
            except Exception as e:
                logging.error(f"Could not concatenate uncertainty metrics: {e}")
                logging.error(f"Error type: {type(e).__name__}")
                if 'all_metrics_dfs' in locals():
                    for i, df in enumerate(all_metrics_dfs):
                        try:
                            unique_cols = list(dict.fromkeys(df.columns))
                            logging.error(f"DataFrame {i} unique columns: {unique_cols}")
                            logging.error(f"DataFrame {i} shape: {df.shape}")
                        except Exception as inner_e:
                            logging.error(f"Error inspecting DataFrame {i}: {inner_e}")
            
            try:
                if misclassifications_list:
                    misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in
                                              misclass_dict.values()]
                    properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in
                                                 properclass_dict.values()]
                    
                    if misclassifications_dfs:
                        misclassifications_df = pd.concat(misclassifications_dfs, ignore_index=True)
                        misclassification_path = generate_misclassification_path(status='confidence_all_hf')
                        save_csv(misclassifications_df, misclassification_path)
                        
                        properclassifications_df = pd.concat(properclassifications_dfs, ignore_index=True)
                        properclassifications_path = generate_properclassification_path(status='confidence_all_hf')
                        save_csv(properclassifications_df, properclassifications_path)
                        
                        # Save individual confidence level files
                        for confidence_level in confidence_levels:
                            level_misclass = misclassifications_df[misclassifications_df['confidence_level'] == confidence_level]
                            if not level_misclass.empty:
                                level_path = generate_misclassification_path(status=f'confidence_{confidence_level}_hf')
                                save_csv(level_misclass, level_path)
                            
                            level_proper = properclassifications_df[properclassifications_df['confidence_level'] == confidence_level]
                            if not level_proper.empty:
                                level_path = generate_properclassification_path(status=f'confidence_{confidence_level}_hf')
                                save_csv(level_proper, level_path)
                        
                        # Analyze misclassifications with uncertainty info
                        misclassification_report_path = generate_misclassification_path(status='confidence_report_hf')
                        results = analyze_misclassification_errors(misclassification_path, 2)
                        export_results(results, misclassification_report_path)
                        
                        logging.info("All uncertainty-based misclassifications analyzed and saved.")
                    else:
                        logging.warning("No uncertainty misclassification DataFrames to process.")
                else:
                    logging.warning("No uncertainty misclassifications to process.")
            except Exception as e:
                logging.error(f"Error processing uncertainty misclassifications: {e}")
        else:
            # Standard evaluation processing (no confidence levels)
            all_metrics_list = []
            misclassifications_list = []
            properclassifications_list = []
            for phenotype in phenotypes:
                for model_name in models:
                    key = (phenotype, model_name)
                    if key in all_metrics:
                        all_metrics_list.append(all_metrics[key])
                    if key in misclassifications_concat:
                        misclassifications_list.append(misclassifications_concat[key])
                        properclassifications_list.append(properclassifications_concat[key])

            try:
                if all_metrics_list:
                    # Initialize list for cleaned DataFrames
                    all_metrics_dfs = []

                    for i, metric_dict in enumerate(all_metrics_list):
                        for column in metric_dict.values():
                            if isinstance(column, pd.DataFrame):
                                # Remove duplicate columns
                                column = column.loc[:, ~column.columns.duplicated()]

                                # Reset index to ensure clean concatenation
                                df_reset = column.reset_index(drop=True)

                                # Add to list
                                all_metrics_dfs.append(df_reset)
                            else:
                                logging.warning(f"Skipping non-DataFrame object: {type(column)}")

                    if all_metrics_dfs:
                        # Concatenate with clean indices
                        clean_all_metrics_df = pd.concat(all_metrics_dfs, axis=0, ignore_index=True)

                        # Remove any duplicate columns that might have been created during concatenation
                        clean_all_metrics_df = clean_all_metrics_df.loc[:, ~clean_all_metrics_df.columns.duplicated()]

                        # Save the concatenated DataFrame
                        overall_summary_path = generate_metrics_summary_path(level='overall')
                        save_csv(clean_all_metrics_df, overall_summary_path)

                        # Analyzing the data
                        aggregate_and_save_dataframes(clean_all_metrics_df, output_dir='output/classic/metrics/analysis')
                        logging.info(f"All evaluation metrics concatenated and saved to {overall_summary_path}")
                        logging.info(f"Final DataFrame shape: {clean_all_metrics_df.shape}")
                    else:
                        logging.warning("No metric DataFrames to concatenate.")
                else:
                    logging.warning("No metrics to concatenate.")
            except Exception as e:
                logging.error(f"Could not concatenate all metrics: {e}")
                logging.error(f"Error type: {type(e).__name__}")
                if 'all_metrics_dfs' in locals():
                    for i, df in enumerate(all_metrics_dfs):
                        try:
                            # Log column names without duplicates
                            unique_cols = list(dict.fromkeys(df.columns))
                            logging.error(f"DataFrame {i} unique columns: {unique_cols}")
                            logging.error(f"DataFrame {i} shape: {df.shape}")
                        except Exception as inner_e:
                            logging.error(f"Error inspecting DataFrame {i}: {inner_e}")

            try:
                if misclassifications_list:
                    misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in
                                              misclass_dict.values()]
                    properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in
                                                 properclass_dict.values()]
                    if misclassifications_dfs:
                        misclassifications_df = pd.concat(misclassifications_dfs, ignore_index=True)
                        misclassification_path = generate_misclassification_path(status='raw_hf')
                        save_csv(misclassifications_df, misclassification_path)

                        properclassifications_df = pd.concat(properclassifications_dfs, ignore_index=True)
                        properclassifications_path = generate_properclassification_path(status='raw_hf')
                        save_csv(properclassifications_df, properclassifications_path)

                        misclassification_report_path = generate_misclassification_path(status='reports_hf')
                        results = analyze_misclassification_errors(misclassification_path, 2)
                        export_results(results, misclassification_report_path)
                        logging.info("All misclassifications analyzed and saved.")
                    else:
                        logging.warning("No misclassification DataFrames to process.")
                else:
                    logging.warning("No misclassifications to process.")
            except Exception as e:
                logging.error(f"Error processing misclassifications: {e}")

    if run_json_parsing_error_analysis:
        if uncertainty:
            # Handle uncertainty error analysis if needed
            errors_df = aggregate_error_csvs(phenotypes)
            if not errors_df.empty:
                try:
                    errors_per_model, errors_per_model_and_phenotype, errors_per_config = generate_error_report(errors_df)
                    
                    # Save with uncertainty suffix
                    errors_per_model_path = generate_errors_report_path("errors_per_model_uncertainty_hf")
                    errors_per_model_phenotype_path = generate_errors_report_path("errors_per_model_and_phenotype_uncertainty_hf")
                    errors_per_config_path = generate_errors_report_path("errors_per_config_uncertainty_hf")

                    save_csv(errors_per_model, errors_per_model_path)
                    save_csv(errors_per_model_and_phenotype, errors_per_model_phenotype_path)
                    save_csv(errors_per_config, errors_per_config_path)
                    logging.info("Uncertainty JSON parsing error report successfully saved")
                except Exception as e:
                    logging.error(f"Could not generate an uncertainty error report: {e}")
        else:
            errors_df = aggregate_error_csvs(phenotypes)
            if not errors_df.empty:
                try:
                    errors_per_model, errors_per_model_and_phenotype, errors_per_config = generate_error_report(errors_df)
                    errors_per_model_path = generate_errors_report_path("errors_per_model_hf")
                    errors_per_model_phenotype_path = generate_errors_report_path("errors_per_model_and_phenotype_hf")
                    errors_per_config_path = generate_errors_report_path("errors_per_config_hf")

                    save_csv(errors_per_model, errors_per_model_path)
                    save_csv(errors_per_model_and_phenotype, errors_per_model_phenotype_path)
                    save_csv(errors_per_config, errors_per_config_path)
                    logging.info("JSON parsing error report successfully saved")

                except Exception as e:
                    logging.error(f"Could not generate an error report: {e}")


async def load_phenotype_predictions_huggingface_async(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                                            show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    Hugging Face version of load_phenotype_predictions_async that uses the Hugging Face API.
    """
    if os.path.exists(metrics_pickle_path):
        try:
            with open(metrics_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
            logging.debug(f"Loaded phenotype metrics from {metrics_pickle_path}")
            return phenotype_predictions
        except Exception as e:
            logging.error(f"Failed to load phenotype metrics from {metrics_pickle_path}: {e}")

    logging.debug(f"Phenotype metrics not found at {metrics_pickle_path}. Running llm_parsing_huggingface_async...")
    return await generate_predictions_huggingface_async(
        phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
        show_config_bar, show_chunk_bar, reasoning, uncertainty, batch_size
    )


async def generate_predictions_huggingface_async(phenotype, model_name, configurations, cleaned_data, metrics_pickle_path,
                                     show_config_bar=True, show_chunk_bar=True, reasoning=False, uncertainty=False, batch_size=10):
    """
    Hugging Face version of generate_predictions_async that uses the Hugging Face API.
    """
    phenotype_predictions = await llm_parsing_huggingface_async(
        phenotype, model_name, configurations, cleaned_data,
        show_config_bar=show_config_bar, show_chunk_bar=show_chunk_bar,
        reasoning=reasoning, uncertainty=uncertainty, batch_size=batch_size
    )

    directory = os.path.dirname(metrics_pickle_path)
    os.makedirs(directory, exist_ok=True)
    try:
        with open(metrics_pickle_path, 'wb') as file:
            pickle.dump(phenotype_predictions, file)
        logging.info(f"Saved phenotype metrics to {metrics_pickle_path}")
    except Exception as e:
        logging.error(f"Failed to save phenotype metrics to {metrics_pickle_path}: {e}")

    return phenotype_predictions


if __name__ == "__main__":
    print(f'Using the following API key {OPENROUTER_API_KEY}')


    # General config

    async def main():
        '''
        await llm_phenotyping_async(
            run_parsing=True,
            run_evaluation=False,
            batch_size=200,
            models=MODELS_OPEN,
            phenotypes=PHENOTYPES_FIX,
            chunking_methods=CHUNKING_NONE,
            prompt_strategies=PROMPT_STRATEGIES_NONE,
            aggregation_methods=AGGREGATION_METHODS_NONE,
            uncertainty=False,
        )
        '''


        '''
        await llm_phenotyping_huggingface_async(
                    run_parsing=False,
                    run_evaluation=True,
                    batch_size=200,
                    models=MODELS_FINE_TUNED,
                    phenotypes=PHENOTYPES,
                    chunking_methods=CHUNKING_NONE,
                    prompt_strategies=PROMPT_STRATEGIES_NONE,
                    aggregation_methods=AGGREGATION_METHODS_NONE,
                    uncertainty=False,
                )
        '''

        print("starting entropy...")



        await entropy_evaluation_async(
            phenotypes=PHENOTYPES, 
            models=MODELS,
            chunking_methods=CHUNKING_METHODS,
            prompt_strategies=PROMPT_STRATEGIES,
            aggregation_methods=AGGREGATION_METHODS,
            batch_size=20,  # Lower number of concurrent API calls
        )


        '''
        entropy_evaluation(
            phenotypes=PHENOTYPES,
            models=MODELS_OPEN_SMALL,
            chunking_methods=CHUNKING_METHODS,
            prompt_strategies=PROMPT_STRATEGIES,
            aggregation_methods=AGGREGATION_METHODS,
        ) 
        '''

    # Run the async function
    asyncio.run(main())
