# MIMIC-IV Phenotyping Atlas

This repository provides a comprehensive suite of scripts and tools for building a phenotyping atlas from the MIMIC-IV dataset. The primary goal is to process the raw MIMIC-IV data, generate structured and analyzable datasets for various clinical phenotypes, and provide a framework for training and evaluating phenotyping algorithms using both traditional machine learning and large language models (LLMs).

The project is divided into two main components:

1.  **MIMIC Parser (`mimic_parser.py`)**: A powerful script that processes the raw MIMIC-IV and MIMIC-IV-Note datasets, creating structured and tabular data for a predefined set of clinical phenotypes.
2.  **Supervised ML** (**`supervised_ML.py`**): Script for training and evaluating various ML models for EHR phenotyping
3.  **LLM Evaluation (`llm_evaluation.py`)**: A flexible tool for evaluating the performance of various LLMs in phenotyping tasks based on clinical notes. For detailed documentation on this component, please see [`llm_evaluation_README.md`](./llm_evaluation_README.md).

## Getting Started

### 1. Obtain MIMIC-IV Dataset

Before you can use the scripts in this repository, you must first obtain access to the MIMIC-IV dataset through PhysioNet.

1. **Request Access**: Apply for credentialed access on the [PhysioNet website](https://physionet.org).

2.  **Download the Data**: Once you have access, download the following versions of the dataset:
    *   **MIMIC-IV v2.2**:
        ```sh
        wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciv/2.2/
        ```
    *   **MIMIC-IV-Note v2.2**:
        ```sh
        wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-iv-note/2.2/
        ```
    
    - **Store MIMIC-IV database in your GCS environment**
    
      ```
      gs://mipa-bucket/raw/mimic-iv-note'
      gs://mipa-bucket/raw/mimic-iv'
      ```
    
      

### 2. Set Up the Environment

This project requires **Python 3.11**. We strongly recommend using a virtual environment to manage dependencies and ensure a consistent setup.

1.  **Clone the Repository**:
    ```sh
    git clone https://github.com/your-repo/MIMIC-IV-Phenotype-Atlas.git
    cd MIMIC-IV-Phenotype-Atlas
    ```

2.  **Create and Activate the Virtual Environment**:
    *   Create the virtual environment:
        ```sh
        python3.11 -m venv venv
        ```
    *   Activate it:
        ```sh
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    Install all the required Python packages using the `requirements.txt` file:
    
    ```sh
    pip install -r requirements.txt
    ```

### 3. Set Up MedCat

The `mimic_parser.py` script uses the MedCat library for Named Entity Recognition (NER) on clinical notes. To use this functionality, you need to download the MedCat models.

1.  **Download MedCat Models**:
    You can download the necessary models from the [MedCat repository](https://github.com/CogStack/MedCAT). Follow their instructions to download and place the models in the `medcat/` directory within this project.

2.  **Configure Paths**:
    Ensure that the paths to the MedCat models are correctly configured in `scripts/constants.py`. By default, the script expects the models to be in the `medcat/medmen` directory.
    - We expect `/medcat/medmen/cdb.dat` and  `/medcat/medmen/vocab.dat`


### 4. Configure Google Cloud Storage (GCS)

The `mimic_parser.py` script exports processed data to Google Cloud Storage by default. To configure GCS:

1.  **Set up GCS credentials**: Place your service account JSON file in the `secrets/` directory as `service-account-file.json`.
2.  **Configure GCS paths**: The GCS bucket paths are configured in `scripts/constants.py`. By default, the script exports to `gs://mipa-bucket/`.



## `mimic_parser.py` - Creating the Phenotyping Atlas

The `mimic_parser.py` script is the core component for processing the raw MIMIC-IV data and generating the phenotyping atlas. It creates structured datasets for both training and testing/validation for a list of predefined phenotypes.

### Data Processing and Output

The script processes data from `data/raw/mimiciv` and `data/raw/mimic-iv-note` and exports the processed files to Google Cloud Storage (GCS) by default, with local paths available as fallback.

#### Training and Test/Validation Sets

A key feature of the `mimic_parser.py` script is its ability to generate distinct datasets for training and for testing/validation:

*   **Training Set**: When generating the training set (`--train True`), the script uses "silver labels" by default. These are noisy labels generated based on a set of predefined rules and heuristics, which allows for the creation of a large, automatically labeled dataset for training machine learning models.
*   **Test/Validation Set**: For the test and validation sets (`--train False`), the script removes the validation cohort and exports the data without the noisy labels. This ensures that the evaluation is performed on a clean, unbiased dataset.

### Output Structure

The processed data is organized as follows:

#### GCS Export Structure (Default)
```
gs://mipa-bucket/
├── processed/
│   ├── cohort/
│   │   ├── train/
│   │   │   ├── c_diff_complication/
│   │   │   ├── c_diff_past/
│   │   │   ├── dementia/
│   │   │   └── ... (other phenotypes)
│   │   ├── test/
│   │   │   ├── c_diff_complication/
│   │   │   ├── c_diff_past/
│   │   │   └── ... (other phenotypes)
│   │   ├── val/
│   │   │   ├── c_diff_complication/
│   │   │   ├── c_diff_past/
│   │   │   └── ... (other phenotypes)
│   │   └── shared_test_validation/
│   │       ├── long/
│   │       ├── pivot/
│   │       ├── summary/
│   │       └── dict/
│   └── indexes/
└── models/
```

### Usage

The `mimic_parser.py` script can be run from the command line with a variety of options to customize the data processing pipeline.

```sh
python mimic_parser.py [OPTIONS]
```

### Parameters

#### General Options

-   `--train`: Specifies the type of dataset to generate (`train` or `test`). Default is `train`.
    *   If `True`: `train`  , the atlas is exported for each phenotype to `gs://mipa-bucket/processed/cohort/train/phenotype`.
    *   If  `False`: `test`, the atlas is exported to `gs://mipa-bucket/processed/cohort/val/phenotype` and `gs://mipa-bucket/processed/cohort/test/phenotype`.
-   `--noisy_label`: If `True`, uses noisy "silver labels" to generate the training dataset. Default is `True`.
-   `--time`: If `True`, includes time-related features. This is necessary for count-based feature aggregation. Default is `True`.
-   `--notes`: If `True`, includes clinical notes (discharge summaries and radiology notes). Default is `True`.
-   `--NLP`: If `True`, applies the MedCat parser for UMLS-based NER on all notes. This is computationally intensive. Default is `True`.
-   `--pivot_type`: The type of pivot to perform (`simple` or `timeseries`). Default is `simple`.
    -   Simple pivot count-aggregates events per admission
    -   Timeseries pivot count-aggregates events per pre-defined aggregation window

-   `--agg_window`: The aggregation window for time-series data (e.g., `D` for daily, `H` for hourly). Default is `D`.

#### Diagnosis ICD Code Grouping

1.  **`--root_icd9`**: Root ICD-9 codes (default: `True`)
    - If `True`, matches ICD-9 to ICD-10 by using the first three ICD-9 digits for better matching.

2.  **`--group_icd10`**: Group ICD-10 codes (default: `True`)
    - If `True`, after converting ICD-9 to ICD-10, groups ICD codes according to the first three digits.

3.  **`--options`**: Additional options (default: `ICD10`)
    - `ICD10_grouped`: Returns ICD-10 codes grouped according to the first three digits.
    - `original`: Returns original ICD-9 and ICD-10 codes from the raw database.
    - Note: `ICD10_grouped` must be used if `group_icd10` is `True`, otherwise non-grouped ICD-10 codes will be returned. All non-converted ICD-9 codes are dropped if not matched.

#### Medication NDC Code Grouping

1.  **`--med_grouping`**: Medication grouping (default: `True`)
    - Groups NDC codes based on their proprietary name.

#### Feature Selection

-   `--lower_labs`: Lower limit percentile for lab results (default: `0.25`)
    - Removes the bottom 25th percentile of unique elements in the lab events table (`itemid`).
-   `--lower_chart`: Lower limit percentile for chart events (default: `0.25`)
    - Removes the bottom 25th percentile of unique elements in the chart events table (`itemid`).
-   `--lower_icd`: Lower limit percentile for ICD codes (default: `0.25`)
    - Removes the bottom 25th percentile of unique elements in the diagnosis and procedure ICD table (`icd_code`).
-   `--lower_meds`: Lower limit percentile for medications (default: `0.25`)
    - Removes the bottom 25th percentile of unique elements in the diagnosis and medication table (`ndc`).
-   `--lower_nlp`: Lower limit percentile for NLP CUI codes (default: `0.10`)
    - Removes the bottom 10th percentile of unique elements in the processed NLP codes from the radiology notes and discharge summary notes (`cui_code`).

#### Alternative Modes

-   `--run_notes_nlp_parsing`: Run notes NLP parsing only via MedCat instead of the full pipeline (default: `False`)
-   `--debug_type`: Specifies data table to debug (choices: `DIAG`, `PROC`, `MEDS`, `CHART`, `LABS`, `RADNOTES`, `DSNOTES`)
-   `--debug_phenotype`: Specifies the phenotype to debug (choices: `c_diff_complication`, `c_diff_past`, `dementia`, `depression`, `diabetes_type_1`, `diabetes_type_2`, `hfref`, `hfpef`, `hypertension`, `sle`, `metastatic_cancer`, `obesity`, `rheumatoid_arthritis`, `alcohol_abuse`, `vte_complication`, `vte_past`)

### Examples

1.  **Generate the Training Set with Default Parameters**:
    ```sh
    python mimic_parser.py --train True
    ```

2.  **Generate the Test/Validation Set with Custom Parameters**:
    ```sh
    python mimic_parser.py --train False --noisy_label False --pivot_type timeseries --agg_window H
    ```

3.  **Run Notes NLP Parsing**:
    ```sh
    python mimic_parser.py --run_notes_nlp_parsing --lower_nlp 0.10
    ```

4.  **Debug a Specific Data Table and Phenotype**:
    ```sh
    python mimic_parser.py --debug_type PROC --debug_phenotype diabetes_type_1
    ```



## `llm_evaluation.py` - LLM Phenotyping Evaluation

The `llm_evaluation.py` script provides comprehensive evaluation of Large Language Models for medical phenotyping tasks. For detailed documentation, see [`llm_evaluation_README.md`](./llm_evaluation_README.md).

### Prerequisites

- Discharge summaries must be manually added to the data/filters directory
  - `data/filters/golden_notes.csv` containing all discharge summaries of the MIPA dataset can be obtained by running the SQL query `docs/golden_notes.SQL` after obtaining the MIMIC-IV licesence
  - `data/filters/golden_labels.csv` containing all discharge summaries labels after consensus review (without the notes) is present in the repo

### Output Structure

The LLM evaluation outputs are organized as follows:

```
output/
├── models/
│   ├── parse/
│   │   ├── aggregated/
│   │   ├── detailed/
│   │   └── pickle/
│   ├── metrics/
│   │   ├── analysis/
│   │   ├── phenotype_metrics/
│   │   └── summary/
│   └── errors/
│       ├── json_errors/
│       ├── json_errors_report/
│       ├── misclassifications_errors/
│       ├── misclassifications_errors_report/
│       ├── proper_classifications/
│       └── parsing_errors_report/
│
└── entropy_report/
    ├── detailed_results.csv
    └── grouped_reports/
      ├── config_level_results.csv
      ├── model_level_results.csv
      ├── detailed_consistency_results.csv
      ├── prompt_level_results.csv
      ├── chunking_level_results.csv
```



## `supervised_ML.py` - Traditional Machine Learning Evaluation

The `supervised_ML.py` script is used to train and evaluate traditional EHR phenotyping algorithms using supervised machine learning approaches.

### Pre-requisites

- The training data and validation data for every phenotype must be available in `gs://mipa-bucket/processed/ ` as per the execution of the mimic_parser.py script
- GCS configuration as for the MIMIC parser



### Supported Models

The script supports the following machine learning models:

1.  **Logistic Regression**: Linear model for binary classification
2.  **Random Forest**: Ensemble method using multiple decision trees
3.  **Gradient Boosting Machine (GBM)**: Sequential ensemble method
4.  **Naive Bayes**: Probabilistic classifier based on Bayes' theorem



### Usage

```sh
python supervised_ML.py
```

The scripts trains and evaluates the four aforementioned models across all the phenotypes of the MIPA dataset.

### Output Structure

```
gs://mipa-bucket/models/
├── phenotype/
    ├── lr/
    ├── naive_bayes/
    ├── random_forest/
    └── gradient_boosting/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.







