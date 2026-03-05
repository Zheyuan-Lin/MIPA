import asyncio
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Set, Tuple
import aiohttp
from scripts.llm.llm_module import *
from scripts.utils import *
from scripts.llm.reasoning_metrics import reasoning_metrics


async def entropy_evaluation(phenotypes, models, chunking_methods, prompt_strategies,
                             aggregation_methods, k=3, easy_path=EASY_PATH, hard_path=CHALLENGING_PATH):
    logging.info("Starting consistency evaluation with async processing")

    # Initialize lists to store results
    detailed_results = []
    semi_aggregated_results = []

    # Load data
    easy_df = pd.read_csv(easy_path)
    hard_df = pd.read_csv(hard_path)

    # Temperatures to evaluate in parallel
    temperatures = [0.0, 0.5, 1.0]

    # Load labeled data
    data_loader = DataLoader(phenotype='manual', filepath=GOLDEN_NOTES_PATH)
    data = data_loader.data()
    cleaned_data = data.clean_text(expand_abbreviations=True)

    # Create a session manager for API calls
    async with aiohttp.ClientSession() as session:
        # Process all temperatures in parallel
        temperature_tasks = []
        for temperature in temperatures:
            temperature_tasks.append(
                process_temperature(
                    temperature, phenotypes, models, chunking_methods,
                    prompt_strategies, aggregation_methods, cleaned_data,
                    easy_df, hard_df, k, session
                )
            )

        # Wait for all temperature tasks to complete and collect results
        temp_results = await asyncio.gather(*temperature_tasks)

        # Combine results from all temperatures
        for temp_detailed, temp_semi_agg in temp_results:
            detailed_results.extend(temp_detailed)
            semi_aggregated_results.extend(temp_semi_agg)

    # Save all results
    save_entropy_results(detailed_results, semi_aggregated_results)
    logging.info("Completed consistency evaluation")


async def process_temperature(temperature, phenotypes, models, chunking_methods,
                              prompt_strategies, aggregation_methods, cleaned_data,
                              easy_df, hard_df, k, session):
    """Process a single temperature value with parallel phenotype processing"""
    logging.info(f"Processing temperature {temperature}")

    detailed_results = []
    semi_aggregated_results = []

    # Process all phenotypes in parallel for this temperature
    phenotype_tasks = []
    for phenotype in phenotypes:
        phenotype_tasks.append(
            process_phenotype(
                phenotype, temperature, models, chunking_methods,
                prompt_strategies, aggregation_methods, cleaned_data,
                easy_df, hard_df, k, session
            )
        )

    # Wait for all phenotype tasks to complete
    phenotype_results = await asyncio.gather(*phenotype_tasks)

    # Combine results from all phenotypes
    for phenotype_detailed, phenotype_semi_agg in phenotype_results:
        detailed_results.extend(phenotype_detailed)
        semi_aggregated_results.extend(phenotype_semi_agg)

    return detailed_results, semi_aggregated_results


async def process_phenotype(phenotype, temperature, models, chunking_methods,
                            prompt_strategies, aggregation_methods, cleaned_data,
                            easy_df, hard_df, k, session):
    """Process a single phenotype with async model configuration processing"""
    logging.info(f"Processing phenotype {phenotype} at temperature {temperature}")

    detailed_results = []
    semi_aggregated_results = []

    # Filter dataset for this phenotype
    hard_hadm_ids = set(hard_df[
                            (hard_df['both_models_misclassified'] == True) &
                            (hard_df['phenotype'] == phenotype)
                            ]['hadm_id'].unique())
    hard_hadm_ids = list(hard_hadm_ids)[:10]

    easy_hadm_ids = set(easy_df[
                            (easy_df['both_models_correct'] == True) &
                            (easy_df['phenotype'] == phenotype)
                            ]['hadm_id'].unique())
    easy_hadm_ids = list(easy_hadm_ids)[:10]

    # Filter using the set of hadm_ids
    filtered_data = cleaned_data[cleaned_data['hadm_id'].isin(easy_hadm_ids + hard_hadm_ids)]
    preselected_rows = filtered_data.sample(min(10, len(filtered_data)), random_state=42)

    # Create a semaphore to limit concurrent API calls (adjust based on API rate limits)
    semaphore = asyncio.Semaphore(5)  # Allow 5 concurrent API calls

    # Process all model configurations
    config_tasks = []

    for model in models:
        for chunking_method in chunking_methods:
            for prompt_strategy in prompt_strategies:
                for aggregation_method in aggregation_methods:
                    # Skip invalid combinations
                    if (prompt_strategy == 'etp' and aggregation_method != 'etp') or \
                            (prompt_strategy != 'etp' and aggregation_method == 'etp'):
                        continue

                    config_tasks.append(
                        process_configuration(
                            phenotype, model, temperature, chunking_method,
                            prompt_strategy, aggregation_method, preselected_rows,
                            easy_hadm_ids, hard_hadm_ids, k, semaphore, session
                        )
                    )

    # Use asyncio.gather instead of async_tqdm.as_completed
    config_results = await asyncio.gather(*config_tasks)

    # Process results
    for config_detailed, config_semi_agg in config_results:
        if config_detailed:
            detailed_results.extend(config_detailed)
        if config_semi_agg:
            semi_aggregated_results.extend(config_semi_agg)

    return detailed_results, semi_aggregated_results


async def process_configuration(phenotype, model, temperature, chunking_method,
                                prompt_strategy, aggregation_method, preselected_rows,
                                easy_hadm_ids, hard_hadm_ids, k, semaphore, session):
    """Process a single configuration asynchronously with rate limiting"""

    config_description = f"{phenotype}-{model}-{temperature}-{chunking_method}-{prompt_strategy}-{aggregation_method}"
    logging.info(f"Processing configuration: {config_description}")

    detailed_results = []

    try:
        # Use semaphore to limit concurrent API calls
        async with semaphore:
            # Initialize async-compatible parser
            parser = AsyncLLMParser(
                model_name=model,
                prompt_engineering=PromptEngineering(prompt_type=prompt_strategy),
                api_url=OPENROUTER_URL,
                api_key=OPENROUTER_API_KEY,
                temperature=temperature,
                session=session
            )

            if chunking_method == 'none':
                if prompt_strategy == 'etp' and aggregation_method == 'etp':
                    # Process each row with async API calls
                    row_tasks = []
                    for _, row in preselected_rows.iterrows():
                        row_tasks.append(
                            parser.evaluate_consistency_for_etp_none_async(
                                row['text'], phenotype, fixed_etp_threshold=0.4, k=k
                            )
                        )

                    # Process rows in parallel
                    row_results = await asyncio.gather(*row_tasks)

                    # Collect results
                    for idx, (entropy, predictions) in enumerate(row_results):
                        row = preselected_rows.iloc[idx]
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
                            'k': k,
                            'difficulty': 'easy' if row['hadm_id'] in easy_hadm_ids else 'hard'
                        })
                else:
                    # Process each row with async API calls for regular consistency
                    row_tasks = []
                    for _, row in preselected_rows.iterrows():
                        row_tasks.append(
                            parser.evaluate_consistency_async(row['text'], phenotype, k=k)
                        )

                    # Process rows in parallel
                    row_results = await asyncio.gather(*row_tasks)

                    # Collect results
                    for idx, (entropy, predictions) in enumerate(row_results):
                        row = preselected_rows.iloc[idx]
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
                            'k': k,
                            'difficulty': 'easy' if row['hadm_id'] in easy_hadm_ids else 'hard'
                        })
            else:
                # Process chunked data asynchronously
                predictions_by_hadm = await parser.evaluate_consistency_for_chunked_data_async(
                    preselected_rows, phenotype, model, chunking_method,
                    prompt_strategy, aggregation_method, k
                )

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
                        'k': k,
                        'difficulty': 'easy' if hadm_id in easy_hadm_ids else 'hard'
                    })

        # Calculate mean entropy for this configuration
        if detailed_results:
            mean_entropy = np.mean([r['entropy'] for r in detailed_results])
            semi_aggregated_result = {
                'phenotype': phenotype,
                'model': model,
                'temperature': temperature,
                'chunking_method': chunking_method,
                'prompt_strategy': prompt_strategy,
                'aggregation_method': aggregation_method,
                'mean_entropy': mean_entropy,
                'k': k
            }
            return detailed_results, [semi_aggregated_result]

    except Exception as e:
        logging.error(f"Error in configuration {config_description}: {str(e)}")
        return [], []

    return detailed_results, []


# AsyncLLMParser class implementation
class AsyncLLMParser:
    """Async version of LLMParser for making concurrent API calls"""

    def __init__(self, model_name, prompt_engineering, api_url, api_key, temperature, session):
        self.model_name = model_name
        self.prompt_engineering = prompt_engineering
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.session = session

    async def evaluate_consistency_async(self, text, phenotype, k=3):
        """Async version of evaluate_consistency"""
        tasks = []
        for _ in range(k):
            tasks.append(self._get_llm_prediction_async(text, phenotype))

        predictions = await asyncio.gather(*tasks)
        entropy = self._calculate_entropy(predictions)
        return entropy, predictions

    async def evaluate_consistency_for_etp_none_async(self, text, phenotype, fixed_etp_threshold=0.4, k=3):
        """Async version of evaluate_consistency_for_etp_none"""
        # This is a placeholder - you'll need to adapt this to match your original implementation
        # For example, if your original method makes multiple API calls for ETP:
        predictions = []
        for _ in range(k):
            # Make ETP-specific API call
            etp_prediction = await self._get_etp_prediction_async(text, phenotype, fixed_etp_threshold)
            predictions.append(etp_prediction)

        entropy = self._calculate_entropy(predictions)
        return entropy, predictions

    async def _get_etp_prediction_async(self, text, phenotype, threshold):
        """Make an ETP-specific API call"""
        # This is just an example - adapt based on your actual implementation
        prompt = self.prompt_engineering.get_prompt(phenotype, text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        async with self.session.post(self.api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return self._parse_response(result)

    async def evaluate_consistency_for_chunked_data_async(self, data, phenotype, model,
                                                          chunking_method, prompt_strategy,
                                                          aggregation_method, k):
        """Async version of evaluate_consistency_for_chunked_data"""
        # This is a placeholder - you'll need to adapt this to match your original implementation
        # Basic implementation to demonstrate the approach:
        predictions_by_hadm = {}

        # Process each hospital admission record
        for _, row in data.iterrows():
            hadm_id = row['hadm_id']
            text = row['text']

            # Break text into chunks based on chunking_method
            chunks = self._chunk_text(text, method=chunking_method)

            chunk_predictions = []
            # Process each chunk with k predictions
            for chunk in chunks:
                chunk_tasks = []
                for _ in range(k):
                    chunk_tasks.append(self._get_llm_prediction_async(chunk, phenotype))

                # Get predictions for this chunk
                chunk_results = await asyncio.gather(*chunk_tasks)
                chunk_predictions.extend(chunk_results)

            # Aggregate predictions based on the desired method
            if aggregation_method == 'majority':
                predictions_by_hadm[hadm_id] = self._aggregate_by_majority(chunk_predictions)
            else:
                # Add other aggregation methods as needed
                predictions_by_hadm[hadm_id] = chunk_predictions

        return predictions_by_hadm

    def _chunk_text(self, text, method):
        """Split text into chunks based on specified method"""
        # Implement your chunking logic here
        if method == 'overlap':
            # Example: Create overlapping chunks of 500 chars with 100 char overlap
            chunks = []
            chunk_size = 500
            overlap = 100

            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])

            return chunks
        else:
            # Default implementation or other methods
            return [text]

    def _aggregate_by_majority(self, predictions):
        """Aggregate predictions by majority vote"""
        if not predictions:
            return []

        # Count occurrences of each prediction
        prediction_counts = {}
        for pred in predictions:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
            else:
                prediction_counts[pred] = 1

        # Find the majority prediction
        majority_pred = max(prediction_counts.items(), key=lambda x: x[1])[0]
        return [majority_pred] * len(predictions)

    async def _get_llm_prediction_async(self, text, phenotype):
        """Make an async API call to the LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        prompt = self.prompt_engineering.get_prompt(phenotype, text)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        try:
            async with self.session.post(self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return self._parse_response(result)
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            raise

    def _calculate_entropy(self, predictions):
        """Calculate entropy from a list of predictions"""
        if not predictions:
            return 0

        # Convert predictions to counts
        prediction_counts = {}
        for pred in predictions:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
            else:
                prediction_counts[pred] = 1

        # Convert counts to probabilities
        total = len(predictions)
        probabilities = [count / total for count in prediction_counts.values()]

        # Calculate entropy
        entropy = -sum([p * np.log2(p) for p in probabilities])
        return entropy

    def _parse_response(self, response_json):
        """Parse the response from the LLM API"""
        # Adapt this to match your API's response format
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")


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


# Main execution
if __name__ == "__main__":
    # Set up standard progress tracking
    logging.basicConfig(level=logging.INFO)
    print(f'Using the following API key {OPENROUTER_API_KEY}')

    print("Starting entropy evaluation with async processing...")

    # Run the async evaluation
    asyncio.run(entropy_evaluation(
        phenotypes=PHENOTYPES,
        models=MODELS_TEST,
        chunking_methods=CHUNKING_METHODS,
        prompt_strategies=PROMPT_STRATEGIES,
        aggregation_methods=AGGREGATION_METHODS,
        k=3))

    print("Entropy evaluation complete!")
