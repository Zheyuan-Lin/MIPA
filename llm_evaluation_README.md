# LLM Phenotyping and Consistency Evaluation

This document provides a guide to using the `llm_evaluation.py` script, a powerful tool for evaluating Large Language Models (LLMs) on clinical phenotyping tasks using the MIMIC-IV dataset.

The script has two core capabilities:
1.  **LLM Phenotyping**: Performs a comprehensive evaluation of one or more LLMs across a variety of configurations (e.g., different chunking methods, prompting strategies) to measure their performance on phenotyping tasks.
2.  **Consistency Evaluation**: Assesses the reliability and stability of LLM predictions by running the same task multiple times and measuring the consistency of the outputs.

## Quickstart

### 1. Environment Setup

Before running the script, ensure your environment is correctly configured.

1.  **Python Version**: This project requires **Python 3.11**.
2.  **Virtual Environment**: Activate the project's virtual environment:
    ```sh
    source venv/bin/activate
    ```
3.  **Dependencies**: Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### 4. API Keys

The script uses various APIs to access the different LLMs. You will need to configure the appropriate API keys as environment variables in a `.env` file at the project root.

*   **OpenRouter**: For most models, you will need an OpenRouter API key.
    ```
    OPENROUTER_API_KEY="your_open_router_api_key"
    ```
*   **MedBio Models**: To use the `openbio_8b` and `openbio_70b` models, you will also need to provide keys for DeepInfra and Lambda Labs.
    ```
    DEEPINFRA_API_KEY="your_deepinfra_api_key"
    LAMBDA_API_KEY="your_lambda_api_key"
    ```

Create the `.env` file in the project's root directory and add the keys for the models you intend to use.

### 2. Run a Default Evaluation

To get started quickly, you can run both the phenotyping and consistency evaluations using the default settings. The script will cycle through a predefined list of phenotypes, models, and methods.

```sh
python llm_evaluation.py --run_llm_phenotyping --run_entropy
```

## Configuring an Evaluation Run

You can customize evaluation runs using command-line arguments to tailor the scope, processing methods, and output.

```sh
python llm_evaluation.py [OPTIONS]
```

### Main Function Flags

*   `--run_llm_phenotyping`: Run the main phenotyping evaluation.
*   `--run_entropy`: Run the consistency evaluation.

### Evaluation Scope

Define the scope of your evaluation with these parameters:

*   `--phenotypes`: A list of one or more phenotypes to evaluate (e.g., `c_diff_past`, `dementia`).
*   `--models`: A list of one or more LLMs to use (e.g., `llama_3_8b`, `gpt_4o_mini`).

### Processing and Prompting Strategies

Control how the clinical notes are processed and presented to the LLM:

*   `--chunking_methods`: How to split long documents into smaller pieces.
    *   `none`: No chunking.
    *   `document`: Splits by sections.
    *   `naive`: Splits by a fixed number of tokens.
    *   `semantic`: Uses semantic understanding to create coherent chunks.
*   `--prompt_strategies`: The prompting technique to use.
    *   `zero_shot`: Direct question without examples.
    *   `few_shot`: Provide a few examples in the prompt.
    *   `COT` (Chain of Thought): Asks the model to "think step-by-step."
    *   `etp` (Extract, Transform, Predict): A specialized strategy for structured data extraction.
*   `--aggregation_methods`: How to combine results from multiple chunks.
    *   `any`: A single positive prediction makes the final result positive.
    *   `majority`: Majority vote determines the final result.
    *   `etp`: A specialized aggregation for the `etp` prompt strategy.

### Function-Specific Arguments

*   `--k <int>`: For consistency evaluation (`--run_entropy`), specifies the number of times to run the evaluation. Default is `3`.
*   `--etp_cutoff <int>`: For the `etp` strategy, sets the threshold for a positive classification. Default is `2`.

### Example of a Custom Run

This command runs a phenotyping evaluation on the `sle` phenotype using the `gemma_7b_it` model. It processes notes using both `document` and `naive` chunking, applies the `etp` prompt strategy, and aggregates the results using the `etp` method.

```sh
python llm_evaluation.py --run_llm_phenotyping \
--phenotypes sle \
--models gemma_7b_it \
--chunking_methods document naive \
--prompt_strategies etp \
--aggregation_methods etp
```

## Understanding the Output

All evaluation results are saved in the `output/` directory, which is organized to facilitate detailed analysis.

### Output Directory Structure

The `output/` directory is organized into three main subdirectories, each corresponding to a different type of analysis. Within these, results are further organized by **phenotype**.

```
output/
├── classic/
│   ├── parse/
│   │   ├── aggregated/
│   │   │   └── ... (one folder per phenotype)
│   │   ├── detailed/
│   │   │   └── ... (one folder per phenotype)
│   │   └── pickle/
│   │       └── ... (one folder per phenotype)
│   ├── metrics/
│   │   ├── analysis/
│   │   ├── phenotype_metrics/
│   │   │   └── ... (one folder per phenotype)
│   │   └── summary/
│   └── errors/
│       ├── json_errors/
│       │   └── ... (one folder per phenotype)
│       ├── json_errors_report/
│       ├── misclassifications_errors/
│       │   └── ... (one folder per phenotype)
│       ├── misclassifications_errors_report/
│       └── proper_classifications/
│           
│        
│           
└── entropy_report/
    ├── detailed_results.csv
    ├── config_level_results.csv
    ├── model_level_results.csv
    ├── detailed_consistency_results.csv
    ├── prompt_level_results.csv
    ├── chunking_level_results.csv
    └── grouped_reports/
```

* `output/classic/`: Contains the primary results from the phenotyping runs, including detailed parsing outputs, performance metrics, and error analyses.

  *   detailed/ contains chunk level prediction
  *   `aggregated/` contains discharge summary level prediction
  *   `metrics/`
      *   `phenotype_metrics/` contains performance metrics per phenotype
      *   `summary/` contains contains performance metrics at the config level (*prompt-aggregation-model*)
      *   `analysis/` contains performance metrics at the model level

  *   `errors/`
      *   `json_errors/`
          *   chunk level error of json parsing of the llm output
      *   `json_errors_report/`
          *   analysis of the total type and amount of json parsing errors
      *   `misclassifications_errors/`
          *   discharge_summary level misclassified observations
      *   `misclassifications_errors_report/`
          *   aggregated analysis by model and configuration of misclassified observations
      *   `proper_classifications/`
          *   discharge_summary level data of properly classified observations

* `output/entropy_report/`: Contains the results from the consistency or entropu analysis (`--run_entropy`) evaluations, as well as high-level summary reports from the phenotyping runs.

  *   Entropy`detailed_results.csv`: The most granular output, providing prediction details for every individual note.

  *   `model_level_results.csv`: An aggregated summary of performance metrics (accuracy, precision, recall, F1-score) for each model across all tested configurations.

  *   `config_level_results.csv`: A detailed breakdown of performance metrics for each specific combination of model, chunking method, and prompt strategy.

  *   `detailed_consistency_results.csv`: The raw output from the consistency evaluation, showing prediction stability across multiple runs.

