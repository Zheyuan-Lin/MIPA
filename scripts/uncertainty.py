async def llm_phenotyping_async(run_parsing=True, run_evaluation=True, run_json_parsing_error_analysis=True,
         show_overall_bar=True, show_config_bar=True, show_chunk_bar=True,
         phenotypes=PHENOTYPES, models=MODELS_OPEN, chunking_methods=CHUNKING_METHODS,
         prompt_strategies=PROMPT_STRATEGIES, aggregation_methods=AGGREGATION_METHODS,
         batch_size=100):
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

    all_metrics = {}
    misclassifications_concat = {}
    properclassifications_concat = {}

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
                batch_size=batch_size
            )

            phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)


        else:
            with open(parsed_pickle_path, 'rb') as file:
                phenotype_predictions = pickle.load(file)
                phenotype_predictions = filter_incomplete_parsing(phenotype_predictions, phenotype, model_name)

        # Evaluating all predictions and saving metrics
        if run_evaluation:
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
                misclassifications_dfs = [df for misclass_dict in misclassifications_list for df in misclass_dict.values()]
                properclassifications_dfs = [df for properclass_dict in properclassifications_list for df in properclass_dict.values()]
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