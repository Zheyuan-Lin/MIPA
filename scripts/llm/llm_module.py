import logging
import json
import re
import time
import asyncio
import aiohttp
import traceback

import requests
import torch
import numpy as np
import pandas as pd
import traceback
from collections import defaultdict, Counter

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from requests.exceptions import HTTPError, ConnectionError

from scripts.constants import *
from scripts.constants_reasoning import *
from scripts.utils import *
from scripts.constants_etp import *

class TextChunker:
    def __init__(self, method='none', embedding_model_name='bert-base-uncased', naive_chunk_token_size=256,
                 min_tokens_per_cluster=256 ,max_tokens_per_cluster=512, group_size=1,
                 similarity_threshold=0.7):
        self.method = method
        if self.method == 'naive':
            self.naive_chunk_token_size = naive_chunk_token_size
        if self.method == 'semantic':
            self.model_name = embedding_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"Loaded model: {self.model_name} successfully")
            self.threshold = similarity_threshold  # similarity threshold for semantic chunking
            self.min_tokens_per_cluster = min_tokens_per_cluster
            self.max_tokens_per_cluster = max_tokens_per_cluster
            self.group_size = group_size


    def chunk_text(self, text: str) -> List[str]:
        if self.method == 'none':
            return [text]
        elif self.method == 'document':
            return self.document_chunking(text)
        elif self.method == 'semantic':
            return self.semantic_chunking(text)
        elif self.method == 'naive':
            return self.naive_chunking(text)
        else:
            raise ValueError(f"Unsupported chunking method: {self.method}")

    def document_chunking(self, text: str) -> List[str]:
        # Define common section headers in EHR discharge summaries
        section_headers = [
            "Allergies:", "Chief Complaint:", "Major Surgical or Invasive Procedure:", "History of Present Illness:", "Past Medical History:",
            "Social History:", "Family History:", "Physical Exam:", "Pertinent Results:", "Brief Hospital Course:", "Medications on Admission:",
            "Discharge Medications:", "Discharge Disposition:", "Discharge Diagnosis:", "Discharge Condition:", "Discharge Instructions:",
            "Followup Instructions:"
        ]

        # Create a regex pattern to match section headers
        pattern = re.compile(r'(' + '|'.join(re.escape(header) for header in section_headers) + ')', re.IGNORECASE)

        # Split the text based on the section headers
        sections = pattern.split(text)
        chunks = []

        # Combine headers with their corresponding sections
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            if len(content) > 150:
                chunks.append(f"{header}\n{content}")
            else:
                continue

        return chunks

    def naive_chunking(self, text: str) -> List[str]:
        # Tokenize the text
        tokens = text.split()
        chunks = []
        start_index = 0
        while start_index < len(tokens):
            # Determine the end index for the chunk considering max_tokens
            end_index = start_index + self.naive_chunk_token_size
            if end_index < len(tokens):
                # Convert the token range back to a string to search for sentence endings
                chunk_text = ' '.join(tokens[start_index:end_index + 20])
                match = re.search(r'([.!?])\s', chunk_text)
                if match:
                    last_punct_position = 0
                    # Find the last occurrence of sentence-ending punctuation to determine the chunk boundary

                    for m in re.finditer(r'([.!?])\s', chunk_text):
                        last_punct_position = m.end()
                    # Adjust end_index based on the position of the last punctuation found

                    end_index = start_index + len(chunk_text[:last_punct_position].split())
                else:
                    # If no punctuation is found, use the max_tokens limit

                    end_index = start_index + self.naive_chunk_token_size
            else:
                end_index = len(tokens)
            chunk = ' '.join(tokens[start_index:end_index]).strip()
            chunks.append(chunk)
            start_index = end_index
        return chunks

    def semantic_chunking(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        embeddings = {sentence: self._get_embedding(sentence) for sentence in sentences if sentence.strip()}

        similarity_matrix = cosine_similarity(list(embeddings.values()))
        Z = linkage(similarity_matrix, 'ward')
        labels = fcluster(Z, t=self.threshold, criterion='distance')

        clusters = {}
        for label, sentence in zip(labels, embeddings.keys()):
            if label in clusters:
                clusters[label].append(sentence)
            else:
                clusters[label] = [sentence]

        final_clusters = self._merge_clusters(clusters, embeddings)

        return [' '.join(cluster) for cluster in final_clusters.values()]

    def _get_embedding(self, text):
        with torch.no_grad():
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            model_output = self.model(**encoded_input)
            embedding = model_output[0][:, 0]
            return torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze().numpy()

    def _merge_clusters(self, clusters, embeddings):
        cluster_keys = list(clusters.keys())
        merged_clusters = {}
        cluster_embeddings = {}

        # Calculate a representative embedding for each cluster
        for key in cluster_keys:
            cluster_embeddings[key] = np.mean([embeddings[sent] for sent in clusters[key]], axis=0)

        for key in cluster_keys:
            if key not in merged_clusters:
                merged_clusters[key] = clusters[key]
                current_embedding = cluster_embeddings[key]
                token_count = len(' '.join(clusters[key]).split())

                if token_count < self.min_tokens_per_cluster:
                    self._attempt_merge(key, merged_clusters, cluster_embeddings, current_embedding)

        # Ensure no cluster exceeds the maximum token limit
        self._ensure_max_token_limit(merged_clusters, cluster_embeddings)

        # Final cleaning to remove very small clusters
        merged_clusters = {k: v for k, v in merged_clusters.items() if len(' '.join(v).split()) >= 40}

        return merged_clusters

    def _attempt_merge(self, key, merged_clusters, cluster_embeddings, current_embedding):
        max_similarity = 0
        merge_candidate = None
        for other_key, other_cluster in merged_clusters.items():
            if other_key != key:
                other_embedding = cluster_embeddings[other_key]
                similarity = cosine_similarity([current_embedding], [other_embedding])[0][0]
                combined_token_count = len(' '.join(other_cluster + merged_clusters[key]).split())
                if similarity > max_similarity and combined_token_count <= self.max_tokens_per_cluster:
                    max_similarity = similarity
                    merge_candidate = other_key

        if merge_candidate and max_similarity > self.threshold:
            merged_clusters[merge_candidate].extend(merged_clusters.pop(key))

    def _ensure_max_token_limit(self, merged_clusters, cluster_embeddings):
        for key in list(merged_clusters.keys()):
            if len(' '.join(merged_clusters[key]).split()) > self.max_tokens_per_cluster:
                # Split or redistribute the cluster if it's too large
                self._split_large_cluster(key, merged_clusters, cluster_embeddings)

    def _split_large_cluster(self, key, merged_clusters, cluster_embeddings):
        cluster = merged_clusters.pop(key)
        sentences = cluster
        new_clusters = []
        current_cluster = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_token_count + sentence_tokens <= self.max_tokens_per_cluster:
                current_cluster.append(sentence)
                current_token_count += sentence_tokens
            else:
                new_clusters.append(current_cluster)
                current_cluster = [sentence]
                current_token_count = sentence_tokens

        if current_cluster:
            new_clusters.append(current_cluster)

        for i, new_cluster in enumerate(new_clusters):
            new_key = f"{key}_{i}"
            merged_clusters[new_key] = new_cluster
            cluster_embeddings[new_key] = np.mean([embeddings[sent] for sent in new_cluster], axis=0)

    def count_tokens(self, text: str) -> int:
        return len(text.split())
class TextCleaner:
    def __init__(self):
        self.abbreviations = self.load_abbreviations_from_json(ABV_PATH)

    def load_abbreviations_from_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def expand_abbreviations(self, text):
        for abbreviation, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbreviation) + r'\b', expansion, text)
        return text

    def clean_text(self, text: str, expand_abbreviations=True) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,\':-]', '', text)
        if expand_abbreviations:
            text = self.expand_abbreviations(text)
        return text
class TextData(pd.DataFrame):
    @property
    def _constructor(self):
        return TextData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cleaner = TextCleaner()

    def clean_text(self, expand_abbreviations=True):
        self.loc[:, 'text'] = self['text'].apply(lambda x: self.text_cleaner.clean_text(x, expand_abbreviations))
        return self

    def chunk_text(self, method='none', **kwargs):
        chunker = TextChunker(method=method, **kwargs)
        chunks = []
        for index, row in self.iterrows():
            hadm_id = row['hadm_id']
            text = row['text']
            text_chunks = chunker.chunk_text(text)
            for chunk in text_chunks:
                chunks.append({'hadm_id': hadm_id, 'chunk': chunk})
        return TextData(chunks)

    def context_compression(self, method: str = 'naive'):
        chunker = TextChunker(method=method)
        compression_stats = []
        for index, row in self.iterrows():
            hadm_id = row['hadm_id']
            original_text = row['text']
            original_tokens = chunker.count_tokens(original_text)
            chunked_text = ' '.join(chunker.chunk_text(original_text))
            chunked_tokens = chunker.count_tokens(chunked_text)
            tokens_saved = original_tokens - chunked_tokens
            compression_stats.append({'hadm_id': hadm_id, 'tokens_saved': tokens_saved})
        return pd.DataFrame(compression_stats)

    def count_chunks(self):
        if 'hadm_id' in self.columns and 'chunk' in self.columns:
            return self.groupby('hadm_id').size().reset_index(name='num_chunks')
        else:
            raise ValueError("DataFrame does not contain the required 'hadm_id' and 'chunk' columns")

    def get_clusters(self):
        return self.clusters


class DataLoader:
    #TODO modify naming as this loads all gold labels within single file

    DSNOTES = 'DSNOTES'
    TRAIN = True
    LONG = True

    def __init__(self, phenotype: str, filepath: str = None):
        # If a filepath is provided, use it to load the data
        if filepath:
            self.filepath = filepath
        else:
            # Existing logic to generate the file path based on the phenotype
            if phenotype == 'demo':
                self.filepath = 'gs://mipa-bucket/raw/demo_ds/data.csv'
            elif phenotype == 'test':
                self.filepath = 'gs://mipa-bucket/raw/sample_ds/discharge_sample.csv'
            else:
                self.filepath = generate_file_path(phenotype, self.DSNOTES, train=self.TRAIN, long=self.LONG)

    def data(self) -> TextData:
        try:
            df = load_csv(self.filepath)
            return TextData(df)
        except FileNotFoundError:
            logging.error(f"File not found: {self.filepath}")
            raise
        except pd.errors.EmptyDataError:
            logging.error(f"No data: {self.filepath}")
            raise
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {e}")
            raise

class PromptEngineering:
    """
    A class for generating prompts for language models based on different prompting strategies.

    Attributes:
        prompt_type (str): The type of prompting strategy to use. Supported types include 'zero_shot', 'few_shot', and 'COT'.
        templates (dict): A dictionary mapping prompt types to their respective template strings.

    Methods:
        get_prompt(phenotype: str, discharge_summary: str) -> str:
            Generates a prompt string based on the specified phenotype and discharge summary.
    """

    def __init__(self, prompt_type: str = 'zero_shot'):
        """
        Initializes the PromptEngineering class with a specified prompting strategy.

        Parameters:
            prompt_type (str): The prompting strategy to use. Default is 'zero_shot'.
                Supported values:
                - 'zero_shot': Uses a template that requires no prior examples.
                - 'few_shot': Uses a template that includes a few examples before asking the question.
                - 'COT': Uses a template that encourages a step-by-step reasoning process (Chain of Thought).

        Raises:
            ValueError: If an unsupported prompt_type is provided.
        """
        if prompt_type not in ['zero_shot', 'few_shot', 'COT', 'etp', 'COT_reasoning', 'COT_confidence', 'SCOT', 'TOT', 'GOT']:
            raise ValueError("Unsupported prompt_type. Choose from 'zero_shot', 'few_shot', 'etp' or 'COT'.")

        self.prompt_type = prompt_type
        self.templates = {
            'zero_shot': PROMPT_ZERO_SHOT,
            'few_shot': PROMPT_FEW_SHOT,
            'COT': PROMPT_COT,
            'COT_reasoning': PROMPT_COT_REASONING,
            'COT_confidence': PROMPT_COT_CONFIDENCE,
            'SCOT': PROMPT_SCOT_REASONING,
            'TOT': PROMPT_TOT_REASONING,
            'GOT': PROMPT_GOT_REASONING_STEP_1,
            'etp': 'etp'
        }

        self.templates_reasoning_output = {
            'COT_reasoning': PROMPT_COT_OUTPUT,
            'SCOT': PROMPT_SCOT_OUTPUT,
            'TOT': PROMPT_TOT_OUTPUT,
            'GOT': PROMPT_GOT_OUTPUT,
        }

    def get_prompt(self, phenotype: str, discharge_summary: str) -> str:
        """
        Generates a prompt string using the specified phenotype and discharge summary based on the current prompting strategy.

        Parameters:
            phenotype (str): The medical condition or phenotype to be assessed.
            discharge_summary (str): The text of the discharge summary or a relevant segment of it.

        Returns:
            str: A formatted prompt string ready to be used with a language model.
        """
        # Dictionary of phenotype key and phenotype string
        phenotypes_string = {
            'diabetes_type_1': "Type 1 Diabetes",
            'diabetes_type_2': "Type 2 Diabetes",
            'c_diff_complication': "Clostridium Difficile Infection During Current Hospital Stay",
            'c_diff_past': "Clostridium Difficile Before During Current Hospital Stay",
            'dementia': "Dementia",
            'depression': "Depression",
            'hypertension': "Hypertension",
            'hfpef': "Heart Failure with Preserved Ejection Fraction",
            'hfref': "Heart Failure with Reduced Ejection Fraction",
            'sle': "Systemic Lupus Erythematosus",
            'rheumatoid_arthritis': "Rheumatoid Arthritis (seropositive or seronegative)",
            'metastatic_cancer': "Solid Tumor with Metastasis (Stage IV)",
            'obesity': "Obesity",
            'alcohol_abuse': "Alcohol Abuse",
            'vte_past': "Past Venous Thromboembolism (including DVT, PE, Splanchnic Venous Thrombosis, Cerebral Venous Sinus Thrombosis)",
            'vte_complication': "Venous Thromboembolis (occuring during current hospital stay including DVT, PE, Splanchnic Venous Thrombosis, Cerebral Venous Sinus Thrombosis)",
        }

        etp_templates = {
                'diabetes_type_1'     : PROMPT_ETP_DMI,
                'diabetes_type_2'     : PROMPT_ETP_DMII,
                'c_diff_complication' : PROMPT_ETP_CDIFF_COMPLICATION,
                'c_diff_past'         : PROMPT_ETP_CDIFF_PAST,
                'dementia'            : PROMPT_ETP_DEMENTIA,
                'depression'          : PROMPT_ETP_DEPRESSION,
                'hypertension'        : PROMPT_ETP_HTN,
                'hfpef'               : PROMPT_ETP_HFpEF,
                'hfref'               : PROMPT_ETP_HFrEF,
                'sle'                 : PROMPT_ETP_LUPUS,
                'rheumatoid_arthritis': PROMPT_ETP_RHEUMATOID_ARTHRITIS,
                'metastatic_cancer'   : PROMPT_ETP_METASTATIC_CANCER,
                'obesity'             : PROMPT_ETP_OBESITY,
                'alcohol_abuse'       : PROMPT_ETP_ALCOHOL_ABUSE,
                'vte_past'            : PROMPT_ETP_VENOUS_THROMBOEMBOLISM_PAST,
                'vte_complication'    : PROMPT_ETP_VENOUS_THROMBOEMBOLISM_COMPLICATION,
        }
        phenotype_desc = phenotypes_string.get(phenotype)

        if self.prompt_type == 'etp':
            template = etp_templates.get(phenotype)
            if template is None:
                raise KeyError(f"Phenotype '{phenotype}' not found in etp_templates.")
            else:
                return template.format(phenotype=phenotype, discharge_summary=discharge_summary)
        else:
            template = self.templates.get(self.prompt_type)
            if template is None:
                raise KeyError(f"Prompt type '{self.prompt_type}' not found in templates.")
            else:
                return template.format(phenotype=phenotype_desc, discharge_summary=discharge_summary)

    def get_prompt_reasoning_output(self, phenotype: str, reasoning_analysis: str) -> str:
        """
        Generates a prompt string using the specified phenotype and discharge summary based on the current prompting strategy.

        Parameters:
            phenotype (str): The medical condition or phenotype to be assessed.
            reasoning_analysis (str): The reasoning analysis from the first prompt.

        Returns:
            str: A formatted prompt string ready to be used with a language model.
        """

        template = self.templates_reasoning_output.get(self.prompt_type)
        if template is None:
            raise KeyError(f"Prompt type '{self.prompt_type}' not found in templates.")

        return template.format(phenotype=phenotype, reasoning_analysis=reasoning_analysis)

class LLMParser:
    def __init__(self, model_name: str, prompt_engineering: PromptEngineering, api_url: str, api_key: str, max_retries = 2, temperature = 0.5):
        # Dictionary mapping simplified model names to their API paths
        self.models = {
            #Mistral AI
            'mistral_7b_free': 'mistralai/mistral-7b-instruct:free',
            'mistral_7b': 'mistralai/mistral-7b-instruct-v0.3',
            'mixtral_8x7b': 'mistralai/mixtral-8x7b-instruct',
            'mixtral_8x22b': 'mistralai/mixtral-8x22b-instruct',

            # Qwen
            'qwen_2_7b': 'qwen/qwen-2-7b-instruct',

            # OpenAI
            'gpt_3.5_turbo': 'openai/gpt-3.5-turbo',
            'gpt_4o': 'openai/gpt-4o-2024-11-20',
            'gpt_4_turbo': 'openai/gpt-4-turbo',
            'gpt_4': 'openai/gpt-4',
            'gpt_4o_mini': 'openai/gpt-4o-mini-2024-07-18',

            # Anthropic
            'claude_3.5_sonnet': 'anthropic/claude-3.5-sonnet',
            'claude_3_haiku': 'anthropic/claude-3-haiku',
            # Meta
            'llama_1b': 'meta-llama/llama-3.2-1b-instruct',
            'llama_3b': 'meta-llama/llama-3.2-3b-instruct',
            'llama_8b' : 'meta-llama/llama-3.1-8b-instruct',
            'llama_70b': 'meta-llama/llama-3.3-70b-instruct',
            'llama_405b': 'meta-llama/llama-3.1-405b-instruct',

            # Google
            'gemini_pro_1_5': 'google/gemini-pro-1.5',
            'gemini_flash_1_5': 'google/gemini-flash-1.5',
            'gemini_flash_2': 'google/gemini-2.0-flash-001',

            # Reasoning
            'o1_mini': 'openai/o1-mini',
            'o1_preview': 'openai/o1-preview',

            # Deepseek
            'deepseek_r1_1.5b': 'deepseek/deepseek-r1-distill-qwen-1.5b',
            'deepseek_r1_14b': 'deepseek/deepseek-r1-distill-qwen-14b',
            'deepseek_r1_32b': 'deepseek/deepseek-r1-distill-qwen-32b',
            'deepseek_r1_full': 'deepseek/deepseek-r1',
            
            # OpenBioLLM Models via HuggingFace
            'openbio_8b': 'aaditya/Llama3-OpenBioLLM-8B',
            'openbio_70b': 'aaditya/Llama3-OpenBioLLM-70B',
        }
        self.model_name = model_name
        self.model = self.models.get(model_name, 'mistralai/mistral-7b-instruct:free')  # Default to
        # mistral-7b-instruct [free model]
        self.prompt_engineering = prompt_engineering
        self.api_url = api_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature
        # self.semaphore = asyncio.Semaphore(50)  # Limit concurrent requests


    def supports_logprobs(self, model_name):
        # List of models known to support logprobs
        logprob_supported_models = [
            'llama_1b',
            'llama_3b',
            'llama_70b',
            "gpt_4o_mini"

            # Add other models that support logprobs here
        ]

        return model_name in logprob_supported_models

    def get_key(self, dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None

    def _push_to_llm(self, prompt: str, json_extractor: Callable, phenotype=None, json_output=True) -> Union[Tuple[Any, float, Any, Any, str], Tuple[Any, float, Any, Any, str, float]]:
            """
            Push the prompt to the LLM and return the parsed response.

            :param prompt: The prompt to send to the LLM
            :param json_extractor: Function to extract JSON from the LLM response
            :return: Tuple of (extracted_json, processing_time, input_tokens, output_tokens, response_text)
            """
            if json_output:
                data_dict = {
                    "model": self.model,
                    "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": self.temperature
                }
            else:
                data_dict = {
                    "model": self.model,
                    "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "text"},
                    "temperature": self.temperature
                }

            # Add logprobs if supported
            if self.supports_logprobs(self.get_key(self.models, self.model)):
                data_dict.update({
                    "logprobs": True,
                    "top_logprobs": 5
                })

            data_dict["provider"] = {
                "order": ["DeepInfra", "Lambda", "Novita", "Hyperbolic", "Lepton", "Together", "Fireworks", "OpenAI"],
                "allow_fallbacks": True
            }

            # Convert to JSON string after all updates are done
            data = json.dumps(data_dict)

            backoff_factor = 1

            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    response = requests.post(
                        self.api_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=data
                    )
                    response.raise_for_status()
                    end_time = time.time()
                    processing_time = end_time - start_time

                    parsed_response = response.json()
                    response_text = parsed_response['choices'][0]['message']['content']

                    if not isinstance(response_text, str):
                        logging.error(f"Response text is not a string. Type: {type(response_text)}")
                        continue

                    # Get logprob
                    if ('choices' in parsed_response and
                            parsed_response['choices'] and
                            'logprobs' in parsed_response['choices'][0] and
                            parsed_response['choices'][0]['logprobs'] is not None):
                        logprobs = [token['logprob'] for token in parsed_response['choices'][0]['logprobs']['content']]
                        if logprobs:
                            avg_logprob = sum(logprobs) / len(logprobs)
                    else:
                        avg_logprob = None

                    # Classic
                    if phenotype is None:
                        extracted_json = json_extractor(response_text)
                        return_tuple = (extracted_json, processing_time, parsed_response['usage']['prompt_tokens'],
                                        parsed_response['usage']['completion_tokens'], response_text, avg_logprob)


                    # ETP
                    else:
                        extracted_json, validation_percentage = json_extractor(response_text, phenotype)
                        return_tuple = (extracted_json, processing_time, parsed_response['usage']['prompt_tokens'],
                                        parsed_response['usage']['completion_tokens'], response_text,
                                        validation_percentage, avg_logprob)

                    return return_tuple

                except HTTPError as e:
                    if response.status_code == 429:
                        logging.warning(f"Rate limit exceeded. Retrying in {backoff_factor} seconds...")
                        time.sleep(backoff_factor)
                        backoff_factor *= 2  # Exponential backoff
                    else:
                        logging.error(f"HTTP error occurred: {e}")
                        break  # Exit the retry loop for other HTTP errors

                except ConnectionError as e:
                    logging.error(f"Network connection error occurred: {e}")
                    break  # Exit the retry loop for connection errors

                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    logging.warning(f"Retrying in {backoff_factor} seconds...")
                    time.sleep(backoff_factor)
                    backoff_factor *= 2  # Exponential backoff

            logging.error("Max retries exceeded. Failed to get a valid response.")
            return None, 0.0, 0, 0, None

    def parse(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Parse the text chunks using the specified language model and return the results.
        """
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})

        for _, row in chunk_df.iterrows():
            hadm_id = row['hadm_id']
            prompt = self.prompt_engineering.get_prompt(phenotype, row['chunk'])
            parse_counts[hadm_id]['total'] += 1

            try:
                result = self._push_to_llm(prompt, self._extract_json_from_output_classic, phenotype=None)

                # Unpack with default value for avg_logprob
                if len(result) == 5:  # Old format without avg_logprob
                    extracted_json, processing_time, input_tokens, output_tokens, original_response = result
                    avg_logprob = None
                else:  # New format with avg_logprob
                    extracted_json, processing_time, input_tokens, output_tokens, original_response, avg_logprob = result

                error_occurred = False
                error_type = ""

                if extracted_json is None:
                    error_occurred = True
                    error_type = "Failed to extract JSON from response"
                    logging.debug(error_type)
                else:
                    phenotype_status = self._extract_phenotype(extracted_json)
                    if phenotype_status is None:
                        error_occurred = True
                        error_type = "Failed to extract phenotype status from JSON"
                        logging.debug(error_type)

                if error_occurred:
                    errors.append({
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'raw_output': original_response,
                        'error_type': error_type,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    })
                else:
                    results.append({
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'llm_response': extracted_json,
                        'phenotype_status': phenotype_status,
                        'justification': self._extract_justification(extracted_json),
                        'chunk_tokens': self._return_ntokens(row['chunk']),
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'processing_time': processing_time,
                        'avg_logprob': avg_logprob
                    })
                    parse_counts[hadm_id]['success'] += 1

            except Exception as e:
                logging.error(f"Error processing row: {e}")
                traceback.print_exc()
                errors.append({
                    'hadm_id': hadm_id,
                    'chunk': row['chunk'],
                    'error_type': f"Other error : {e})"
                })

            if pbar:
                pbar.update(1)

        result_df = pd.DataFrame(results)
        error_df = pd.DataFrame(errors) if errors else None

        # Calculate and add parsed ratios
        parsed_ratios = {
            hadm_id: counts['success'] / counts['total']
            for hadm_id, counts in parse_counts.items()
        }
        result_df['parsed_ratio'] = result_df['hadm_id'].map(parsed_ratios)

        return result_df, error_df

    def extract_then_phenotype(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None) -> Tuple[
        pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Parse the text chunks using the specified language model and return the results.
        """
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})

        for _, row in chunk_df.iterrows():
            hadm_id = row['hadm_id']
            prompt = self.prompt_engineering.get_prompt(phenotype, row['chunk'])
            parse_counts[hadm_id]['total'] += 1

            try:
                result = self._push_to_llm(
                    prompt, self._extract_json_from_output_etp, phenotype)

                # Handle different return value lengths
                if len(result) == 6:  # Old format without avg_logprob
                    extracted_json, processing_time, input_tokens, output_tokens, original_response, etp_json_validation_percentage = result
                    avg_logprob = None
                    error_occurred = False
                    error_type = ""

                elif len(result) == 7:  # New format with avg_logprob
                    (extracted_json, processing_time, input_tokens, output_tokens,
                     original_response, etp_json_validation_percentage, avg_logprob) = result
                    error_occurred = False
                    error_type = ""
                else:
                    raise ValueError(f"Unexpected number of return values: {len(result)}")

                if extracted_json is None:
                    error_occurred = True
                    error_type = "Failed to extract JSON ETP from response"
                    logging.debug(error_type)

                if error_occurred:
                    errors.append({
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'output': original_response,
                        'error_type': error_type,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    })
                else:
                    results.append({
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'llm_response': extracted_json,
                        'etp_validation_percentage': etp_json_validation_percentage,
                        'chunk_tokens': self._return_ntokens(row['chunk']),
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'processing_time': processing_time,
                        'avg_logprob': avg_logprob
                    })
                    parse_counts[hadm_id]['success'] += 1

            except Exception as e:
                logging.error(f"Error processing row: {e}")
                traceback.print_exc()
                errors.append({
                    'hadm_id': hadm_id,
                    'chunk': row['chunk'],
                    'error_type': f"Other error : {e})"
                })

            if pbar:
                pbar.update(1)

        # Create result_df using json_normalize for the llm_response column
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            # Normalize the JSON response with default values for optional fields
            def normalize_response(response):
                base_fields = {
                    'phenotype': None,
                    'status': None,
                    'justification': None,
                    'confidence': None,
                    'selected_path': None,
                    'key_decision_points': None
                }
                if isinstance(response, dict):
                    # Update base fields with available values
                    base_fields.update(response)
                return base_fields

            # Apply normalization to each response
            normalized_responses = [normalize_response(resp) for resp in result_df['llm_response'].tolist()]
            response_df = pd.DataFrame(normalized_responses)

            # Ensure key_decision_points is a list or None
            response_df['key_decision_points'] = response_df['key_decision_points'].apply(
                lambda x: x if isinstance(x, list) else None
            )

            # Concatenate with original DataFrame, excluding the original llm_response
            result_df = pd.concat([result_df.drop('llm_response', axis=1), response_df], axis=1)

            # Calculate parsed ratios
            parsed_ratios = {
                hadm_id: counts['success'] / counts['total']
                for hadm_id, counts in parse_counts.items()
            }

            # Add parsed ratio to result_df
            result_df['parsed_ratio'] = result_df['hadm_id'].map(parsed_ratios)

            # Rename 'status' column to 'phenotype_status' if it exists
            if 'status' in result_df.columns:
                result_df.rename(columns={'status': 'phenotype_status'}, inplace=True)

            # Ensure all columns are present with correct types
            result_df = result_df.astype({
                'phenotype': 'object',
                'phenotype_status': 'object',  # Keep as object to handle potential null values
                'justification': 'object',
                'confidence': 'object',
                'selected_path': 'object',
                'key_decision_points': 'object',
                'parsed_ratio': 'float64'
            })

        error_df = pd.DataFrame(errors) if errors else None

        if result_df.empty and error_df is None:
            logging.info("No valid responses or errors recorded.")

        return result_df, error_df

    def _calculate_weighted_avg_logprob(self, metrics):
        total_logprobs = 0
        total_weights = 0

        # Process each numbered logprob with its corresponding output tokens
        for i in range(1, 3):  # For avg_logprob_1 and avg_logprob_2
            logprob_key = f'avg_logprob_{i}'
            output_tokens_key = f'output_tokens{i}'

            logprob = metrics.get(logprob_key)
            output_tokens = metrics.get(output_tokens_key)

            if (logprob is not None and
                    isinstance(logprob, (float, int)) and
                    output_tokens is not None and
                    output_tokens > 0):
                total_logprobs += logprob * output_tokens
                total_weights += output_tokens

        # Add 'got' metrics if they exist
        if (metrics.get('got_avg_logprob') is not None and
                isinstance(metrics['got_avg_logprob'], (float, int)) and
                metrics.get('got_output_tokens') is not None and
                metrics['got_output_tokens'] > 0):
            total_logprobs += metrics['got_avg_logprob'] * metrics['got_output_tokens']
            total_weights += metrics['got_output_tokens']

        if total_logprobs > 0:
            return total_logprobs / total_weights

        return None

    def extract_then_phenotype_reasoning(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None) -> Tuple[
        pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Parse the text chunks using a two/three-step reasoning process with the language model.
        """
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})

        for _, row in chunk_df.iterrows():
            hadm_id = row['hadm_id']
            parse_counts[hadm_id]['total'] += 1

            # Initialize metrics and responses
            metrics = {
                'processing_time1': 0,
                'input_tokens1': 0,
                'output_tokens1': 0,
                'avg_logprob_1': 0,

                'processing_time2': 0,
                'input_tokens2': 0,
                'output_tokens2': 0,
                'avg_logprob_2': 0,

                'got_processing_time': 0,
                'got_input_tokens': 0,
                'got_output_tokens': 0,
                'got_avg_logprob': 0,

            }
            responses = {
                'step1_response': None,
                'got_response': None,
                'final_response': None
            }

            try:
                # Step 1: Get initial reasoning analysis (as string)
                prompt_reasoning = self.prompt_engineering.get_prompt(phenotype, row['chunk'])
                return_tuple = self._push_to_llm(
                    prompt_reasoning, lambda x: x, phenotype=None, json_output=False # Pass through the response as is
                )
                _, metrics['processing_time1'], metrics['input_tokens1'], metrics['output_tokens1'], responses[
                    'step1_response'], avg_logprob = return_tuple

                # For GOT strategy, perform intermediate step
                if self.prompt_engineering.prompt_type == 'GOT':
                    prompt_got = PROMPT_GOT_REASONING_STEP_2.format(
                        phenotype=phenotype,
                        reasoning_analysis=responses['step1_response']
                    )
                    _, metrics['got_processing_time'], metrics['got_input_tokens'], metrics['got_output_tokens'], responses['got_response'], avg_logprob = self._push_to_llm(
                        prompt_got, lambda x: x, phenotype=None, json_output=False
                    )
                    reasoning_analysis = responses['got_response']
                else:
                    reasoning_analysis = responses['step1_response']

                # Step 2/3: Get final output based on reasoning
                prompt_output = self.prompt_engineering.get_prompt_reasoning_output(phenotype, reasoning_analysis)
                extracted_json_final, metrics['processing_time2'], metrics['input_tokens2'], metrics['output_tokens2'], responses['final_response'], avg_logprob = self._push_to_llm(
                    prompt_output, self._extract_json_from_output_reasoning,     phenotype=None, json_output=True
                )

                # Calculate total metrics
                total_processing_time = sum(v for k, v in metrics.items() if 'processing_time' in k)
                total_input_tokens = sum(v for k, v in metrics.items() if 'input_tokens' in k)
                total_output_tokens = sum(v for k, v in metrics.items() if 'output_tokens' in k)
                weighted_avg_logprob = self._calculate_weighted_avg_logprob(metrics)

                if extracted_json_final is None:
                    error_entry = {
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'error_type': "Failed to extract JSON from final response",
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_input_tokens + total_output_tokens,
                        'processing_time': total_processing_time,
                        'avg_logprob': weighted_avg_logprob,
                        'step1_response': responses['step1_response'],
                        'got_response': responses['got_response'],
                        'final_response': responses['final_response'],
                    }
                    errors.append(error_entry)
                else:
                    result_entry = {
                        'hadm_id': hadm_id,
                        'chunk': row['chunk'],
                        'llm_response': extracted_json_final,
                        'chunk_tokens': self._return_ntokens(row['chunk']),
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_input_tokens + total_output_tokens,
                        'processing_time': total_processing_time,
                        'avg_logprob': weighted_avg_logprob,
                        'step1_response': responses['step1_response'],
                        'got_response': responses['got_response'],
                        'final_response': responses['final_response']
                    }
                    results.append(result_entry)
                    parse_counts[hadm_id]['success'] += 1

            except Exception as e:
                logging.error(f"Error processing row: {e}")
                traceback.print_exc()
                error_entry = {
                    'hadm_id': hadm_id,
                    'chunk': row['chunk'],
                    'error_type': f"Other error: {str(e)}",
                    'input_tokens': sum(v for k, v in metrics.items() if 'input_tokens' in k),
                    'output_tokens': sum(v for k, v in metrics.items() if 'output_tokens' in k),
                    'total_tokens': sum(v for k, v in metrics.items() if 'tokens' in k),
                    'processing_time': sum(v for k, v in metrics.items() if 'processing_time' in k),
                    'step1_response': responses['step1_response'],
                    'got_response': responses['got_response'],
                    'final_response': responses['final_response']
                }
                errors.append(error_entry)

            if pbar:
                pbar.update(1)

        # Create result_df using json_normalize for the llm_response column
        result_df = pd.DataFrame(results)

        if result_df.empty:
            # Create empty DataFrame with required columns
            result_df = pd.DataFrame(columns=[
                'hadm_id', 'chunk', 'phenotype', 'phenotype_status',
                'justification', 'confidence', 'selected_path',
                'key_decision_points', 'chunk_tokens', 'input_tokens',
                'output_tokens', 'total_tokens', 'processing_time', 'avg_logprob',
                'step1_response', 'got_response', 'final_response',
            ])
        else:
            # Normalize the JSON response with default values for optional fields
            def normalize_response(response):
                if not isinstance(response, dict):
                    logging.error(f"Invalid response type: {type(response)}, value: {response}")
                    return {
                        'phenotype': None,
                        'status': None,
                        'justification': None,
                        'confidence': None,
                        'selected_path': None,
                        'key_decision_points': None
                    }

                return {
                    'phenotype': response.get('phenotype'),
                    'status': response.get('status'),
                    'justification': response.get('justification'),
                    'confidence': response.get('confidence'),
                    'selected_path': response.get('selected_path'),
                    'key_decision_points': response.get('key_decision_points', [])
                }

            # Apply normalization to each response
            normalized_responses = [normalize_response(resp[0]) for resp in result_df['llm_response'].tolist()]
            response_df = pd.DataFrame(normalized_responses)

            # Ensure key_decision_points is a list or None
            response_df['key_decision_points'] = response_df['key_decision_points'].apply(
                lambda x: x if isinstance(x, list) else None
            )

            # Concatenate with original DataFrame, excluding the original llm_response
            result_df = pd.concat([result_df.drop('llm_response', axis=1), response_df], axis=1)

            # Rename 'status' column to 'phenotype_status' if it exists
            if 'status' in result_df.columns:
                result_df.rename(columns={'status': 'phenotype_status'}, inplace=True)

            # Ensure all columns are present with correct types
            result_df = result_df.astype({
                'phenotype': 'object',
                'phenotype_status': 'object',
                'justification': 'object',
                'confidence': 'object',
                'selected_path': 'object',
                'key_decision_points': 'object'
            })

        # Ensure error_df has consistent columns
        if errors:
            error_df = pd.DataFrame(errors)
            error_columns = [
                'hadm_id', 'chunk', 'error_type',
                'input_tokens', 'output_tokens', 'total_tokens',
                'processing_time', 'step1_response', 'got_response', 'final_response'
            ]

            # Ensure all columns exist with appropriate default values
            for col in error_columns:
                if col not in error_df.columns:
                    error_df[col] = None

            # Reorder columns
            error_df = error_df[error_columns]
        else:
            error_df = None

        if result_df.empty and error_df is None:
            logging.info("No valid responses or errors recorded.")

        return result_df, error_df


    def _return_ntokens(self, text):
        return len(text.split())

    def _extract_json_from_output_classic(self, output):
        """
        Extract and clean JSON from LLM output, handling various edge cases and formats.
        Designed to be generalizable for any JSON structure.

        Args:
            output (str): Raw output string from LLM

        Returns:
            dict: Parsed JSON object or None if parsing fails
        """
        if not output:
            logging.error("Empty output provided")
            return None

        try:
            # 0. If output is a single string of digit, directly return


            # 1. Extract potential JSON structure using improved check_double_bracket
            json_str = check_double_bracket(output)
            if not json_str:
                logging.error("No JSON structure found in output")
                return None

            # 2. Clean and normalize using improved clean_json_classic
            cleaned_json_str = clean_json_classic(json_str)
            if not cleaned_json_str:
                logging.error("Failed to clean JSON string")
                return None

            # 3. Parse JSON
            try:
                json_output = json.loads(cleaned_json_str)

                # Ensure we have a dictionary
                if not isinstance(json_output, dict):
                    logging.error("Parsed JSON is not a dictionary")
                    return None

                return json_output

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}\nCleaned string: {cleaned_json_str}")
                return None

        except Exception as e:
            logging.error(f"Unexpected error processing JSON: {e}\nOriginal output: {output}")
            return None

    def _extract_json_from_output_etp(self, output, phenotype):
        first_clean = check_double_bracket(output)
        cleaned_json_str = clean_json_etp(first_clean)
        if cleaned_json_str is None:
            logging.error("Cleaned JSON string is None, cannot parse.")
            return None
        try:
            json_output = json.loads(cleaned_json_str)
            # Apply etp_grammar function
            json_output, validation_percentage = etp_grammar(phenotype, json_output)
            return json_output, validation_percentage
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode JSON from matched cleaned string: {cleaned_json_str}. Error: {e}, Original String {output}")
            return None

    def _extract_phenotype(self, extracted_json):
        if not isinstance(extracted_json, dict):
            logging.warning(f"extracted_json is not a dictionary: {extracted_json}")
            return None
        status = extracted_json.get("status")
        if status is None:
            logging.warning("'status' key not found in extracted_json")
            return None
        match = re.search(r'\d+', str(status))
        return match.group(0) if match else None

    def _extract_confidence(self, extracted_json):
        if not isinstance(extracted_json, dict):
            logging.warning(f"extracted_json is not a dictionary: {extracted_json}")
            return None
        confidence = extracted_json.get("confidence")
        if confidence is None:
            logging.warning("'confidence' key not found in extracted_json")
            return None
        return confidence.lower()

    def _extract_json_from_output_reasoning(self, output):
        """
        Extract and clean JSON specifically for reasoning prompt outputs.
        Handles both simple and detailed reasoning JSON formats.

        Args:
            output (str): Raw output string from LLM containing reasoning JSON

        Returns:
            tuple: (json_output, None) to match ETP format but without validation percentage
        """
        if not output:
            logging.error("Empty output provided")
            return None

        try:
            # 1. Basic cleanup
            # Remove any text before the first { and after the last }
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
            if not json_match:
                logging.error("No JSON structure found in output")
                return None

            text = json_match.group()

            # 2. Normalize quotes and basic cleanup
            text = re.sub(r'["""]', '"', text)
            text = text.replace('`', '"').replace('\\_', '_')

            # 3. Handle specific reasoning format issues
            # Fix selected_path format
            text = re.sub(r'"selected_path"\s*:\s*"([^"]*(?:Branch\s*\d+)[^"]*)"',
                          lambda m: f'"selected_path": "{m.group(1).replace(""", "").replace(""", "")}"', text)

            # Fix key_decision_points array
            text = re.sub(r'("key_decision_points"\s*:\s*\[)([^\]]+)(\])',
                          lambda m: m.group(1) + re.sub(r'(?<!^)\s*"([^"]+)"', r', "\1"', m.group(2).strip()) + m.group(
                              3),
                          text, flags=re.DOTALL)

            # 4. Clean field values
            # Handle status field
            text = re.sub(r'"status"\s*:\s*"?(\d+)"?', r'"status": "\1"', text)

            # Handle confidence field
            text = re.sub(r'"confidence"\s*:\s*"?(HIGH|MEDIUM|LOW)"?(?:\s+if[^,}]+)?', r'"confidence": "\1"', text,
                          flags=re.IGNORECASE)

            # 5. Final cleanup
            # Remove any trailing commas before closing brackets
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            # Ensure proper spacing
            text = re.sub(r'"\s*:\s*"', '": "', text)
            text = re.sub(r'"\s*:\s*\[', '": [', text)

            try:
                json_output = json.loads(text)

                # Validate and normalize the structure
                required_fields = ['phenotype', 'status', 'justification', 'confidence']
                if not all(field in json_output for field in required_fields):
                    logging.error(f"Missing required fields in JSON output: {json_output}")
                    return None

                # Normalize values
                json_output['confidence'] = json_output['confidence'].upper()
                json_output['status'] = str(json_output['status'])

                # Handle optional fields
                if 'key_decision_points' in json_output:
                    if not isinstance(json_output['key_decision_points'], list):
                        json_output['key_decision_points'] = [json_output['key_decision_points']]

                # Return tuple format to match ETP interface
                return (json_output, None)

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}\nCleaned string: {text}")
                return None

        except Exception as e:
            logging.error(f"Unexpected error processing JSON: {e}\nOriginal output: {output}")
            return None


    def _extract_justification(self, extracted_json):
        if extracted_json and "justification" in extracted_json:
            return extracted_json.get("justification")
        return None

    def evaluate_consistency(self, chunk, phenotype, k=10):

        predictions = []
        label_counts = {}
        max_retries = 2
        backoff_factor = 3

        if self.prompt_engineering.prompt_type != 'etp':
            for _ in range(k):
                prompt = self.prompt_engineering.get_prompt(phenotype, chunk)
                data = json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                })

                for attempt in range(max_retries):
                    try:
                        response = requests.post(self.api_url, headers={"Authorization": f"Bearer {self.api_key}"}, data=data)
                        response.raise_for_status()
                        response_text = response.json()['choices'][0]['message']['content']
                        extracted_json = self._extract_json_from_output_classic(response_text)
                        predicted_phenotype = self._extract_phenotype(extracted_json)
                        predictions.append(predicted_phenotype)
                        break

                    except HTTPError as e:
                        if response.status_code == 429:
                            logging.warning(f"Rate limit exceeded. Retrying in {backoff_factor} seconds...")
                            time.sleep(backoff_factor)
                            backoff_factor *= 2
                        else:
                            logging.error(f"HTTP error occurred: {e}")
                            break

                    except ConnectionError as e:
                        logging.error(f"Network connection error occurred: {e}")
                        break

                    except Exception as e:
                        logging.error(f"An unexpected error occurred: {e}")
                        break

            return self._calculate_entropy(predictions), predictions

        # if etp
        else:
            for _ in range(k):
                prompt = self.prompt_engineering.get_prompt(phenotype, chunk)
                data = json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                })

                for attempt in range(max_retries):
                    try:
                        response = requests.post(self.api_url, headers={"Authorization": f"Bearer {self.api_key}"}, data=data)
                        response.raise_for_status()
                        response_text = response.json()['choices'][0]['message']['content']
                        extracted_json, _ = self._extract_json_from_output_etp(response_text, phenotype)
                        predictions.append(extracted_json)

                        for label, value in extracted_json.items():
                            if label not in label_counts:
                                label_counts[label] = []
                            label_counts[label].append(value)

                        break

                    except HTTPError as e:
                        if response.status_code == 429:
                            logging.warning(f"Rate limit exceeded. Retrying in {backoff_factor} seconds...")
                            time.sleep(backoff_factor)
                            backoff_factor *= 2
                        else:
                            logging.error(f"HTTP error occurred: {e}")
                            break

                    except ConnectionError as e:
                        logging.error(f"Network connection error occurred: {e}")
                        break

                    except Exception as e:
                        logging.error(f"An unexpected error occurred: {e}")
                        break

            entropy = self.calculate_etp_entropy(label_counts)
            return entropy, label_counts


    def evaluate_consistency_for_chunked_data(self, data, phenotype, model_name, chunking_method, prompt_strategy,
                                              aggregation_method, k=3):
        """
        Evaluate prediction consistency for chunked data by running the full pipeline k times.
        """
        predictions = defaultdict(list)  # Store predictions for each hadm_id

        for _ in range(k):
            try:
                # Create parser and chunker
                prompt_engineering = PromptEngineering(prompt_type=prompt_strategy)
                llm_parser = LLMParser(model_name=model_name, prompt_engineering=prompt_engineering,
                                       api_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)

                # Chunk the data
                chunked_data = data.chunk_text(method=chunking_method)

                # Get predictions using the existing parse method
                if prompt_strategy != 'etp':
                    pre_aggregation_results, _ = llm_parser.parse(chunked_data, phenotype)
                else:
                    pre_aggregation_results, _ = llm_parser.extract_then_phenotype(chunked_data, phenotype)

                # Aggregate results
                result_aggregator = ResultAggregator(aggregation_method=aggregation_method)
                if aggregation_method == 'etp':
                    result_aggregator.set_phenotype(phenotype)
                    result_aggregator.set_etp_rule_type('fixed')
                    result_aggregator.set_etp_threshold(0.4)

                final_results = result_aggregator.aggregate(pre_aggregation_results)


                # Store predictions for each hadm_id
                for _, row in final_results.iterrows():
                    predictions[row['hadm_id']].append(row['phenotype_status'])

            except Exception as e:
                logging.error(f"Error in iteration: {e}")
                continue

        return predictions

    def evaluate_consistency_for_etp(self, chunk, phenotype, model_name, chunking_method, prompt_strategy,
                                              aggregation_method, k=3):
        """
        Evaluate prediction consistency for chunked data by running the full pipeline k times.
        """
        predictions = defaultdict(list)  # Store predictions for each hadm_id

        for _ in range(k):
            try:
                # Create parser and chunker
                prompt_engineering = PromptEngineering(prompt_type=prompt_strategy)
                llm_parser = LLMParser(model_name=model_name, prompt_engineering=prompt_engineering,
                                       api_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


                # Get predictions using the existing parse method
                pre_aggregation_results, _ = llm_parser.extract_then_phenotype(chunk, phenotype)

                # Aggregate results
                result_aggregator = ResultAggregator(aggregation_method=aggregation_method)
                if aggregation_method == 'etp':
                    result_aggregator.set_phenotype(phenotype)
                    result_aggregator.set_etp_rule_type('fixed')
                    result_aggregator.set_etp_threshold(0.4)

                final_results = result_aggregator.aggregate(pre_aggregation_results)

                # Store predictions for each hadm_id
                for _, row in final_results.iterrows():
                    predictions[row['hadm_id']].append(row['phenotype_status'])

            except Exception as e:
                logging.error(f"Error in iteration: {e}")
                continue

        return predictions

    def evaluate_consistency_for_etp_none(self, chunk, phenotype, fixed_etp_threshold=0.4, k=10):
        predictions = []
        label_counts = {}
        max_retries = 2
        backoff_factor = 3

        for _ in range(k):
            prompt = self.prompt_engineering.get_prompt(phenotype, chunk)
            data = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            })

            for attempt in range(max_retries):
                try:
                    response = requests.post(self.api_url, headers={"Authorization": f"Bearer {self.api_key}"},
                                             data=data)
                    response.raise_for_status()
                    response_text = response.json()['choices'][0]['message']['content']
                    extracted_json, _ = self._extract_json_from_output_etp(response_text, phenotype)

                    # USING FIXED ETP RULE
                    phenotype_weights = PHENOTYPES_ETP_SCORES.get(phenotype, {})
                    maximal_score = sum(phenotype_weights.values())
                    phenotype_keys = list(phenotype_weights.keys())

                    def calculate_status(row, etp_rule_type='fixed'):
                        # Calculate the score based on the current weights
                        score = sum(row[key] * phenotype_weights.get(key, 0) for key in phenotype_keys)
                        if etp_rule_type == 'fixed':
                            min_score = int(
                                maximal_score * fixed_etp_threshold)  # Using cutoff to determine the fixed threshold
                            return 1 if score >= min_score else 0
                        else:
                            raise ValueError("Invalid rule_type. Choose either 'fixed' or 'dynamic'.")

                    extracted_json_to_df = pd.DataFrame([extracted_json])  # Wrapping the dictionary in a list
                    extracted_json_to_df['phenotype_status'] = extracted_json_to_df.apply(calculate_status, axis=1)
                    predictions.append(extracted_json_to_df['phenotype_status'].iloc[0])

                    break  # Exit the retry loop on success

                except HTTPError as e:
                    if response.status_code == 429:
                        logging.warning(f"Rate limit exceeded. Retrying in {backoff_factor} seconds...")
                        time.sleep(backoff_factor)
                        backoff_factor *= 2
                    else:
                        logging.error(f"HTTP error occurred: {e}")
                        break

                except ConnectionError as e:
                    logging.error(f"Network connection error occurred: {e}")
                    break

                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    break

        return self._calculate_entropy(predictions), predictions

    def _calculate_entropy(self, labels):
        filtered_labels = [label for label in labels if label is not None]
        if not filtered_labels:
            return 0
        value, counts = np.unique(filtered_labels, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_etp_entropy(self, label_counts):
        entropies = []
        for labels in label_counts.values():
            if not labels:
                entropies.append(0)
            else:
                values, counts = np.unique(labels, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities))
                entropies.append(entropy)
        return np.mean(entropies)

    async def _push_to_llm_async(self, prompt: str, json_extractor: Callable, session: aiohttp.ClientSession,
                                 phenotype=None, json_output=True) -> Union[
        Tuple[Any, float, Any, Any, str], Tuple[Any, float, Any, Any, str, float]]:
        """
        Async version of _push_to_llm that uses aiohttp for HTTP requests.
        Works with both OpenRouter and Nebius/HuggingFace APIs.

        Args:
            session: aiohttp.ClientSession to be reused for all API calls
        """
        if json_output:
            data_dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": self.temperature
            }
        else:
            data_dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "text"},
                "temperature": self.temperature
            }

        # Add logprobs if supported
        if self.supports_logprobs(self.get_key(self.models, self.model)):
            data_dict.update({
                "logprobs": True,
                "top_logprobs": 5
            })

        # Determine which API we're using based on the API URL
        is_openrouter_api = 'openrouter.ai' in self.api_url.lower()
        is_nebius_api = 'nebius.com' in self.api_url.lower() or 'huggingface.co' in self.api_url.lower()

        # Add OpenRouter-specific parameters
        if is_openrouter_api:
            data_dict["include_reasoning"] = True
            data_dict["provider"] = {
                "order": ["DeepInfra", "Lambda", "Novita", "Hyperbolic", "Lepton", "Together", "Fireworks", "OpenAI",
                          "Google AI Studio"],
                "allow_fallbacks": False
            }

        # If using fine-tuned models, remove the OpenRouter-specific parameters
        if self.model_name in MODELS_FINE_TUNED or is_nebius_api:
            data_dict.pop("provider", None)
            data_dict.pop("include_reasoning", None)

        # Pre-serialize data for OpenRouter
        data = json.dumps(data_dict)

        backoff_factor = 1

        # Do reverse lookup to get the model name
        if self.model_name in MODELS_REASONING:
            reasoning = True
        else:
            reasoning = False

        return_tuple = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Different request formats based on API type
                if is_openrouter_api:
                    # OpenRouter: send pre-serialized JSON as data
                    async with session.post(
                            self.api_url,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            data=data
                    ) as response:
                        response.raise_for_status()
                        parsed_response = await response.json()
                else:
                    # Nebius/HuggingFace: send Python dict directly as json
                    async with session.post(
                            self.api_url,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json=data_dict
                    ) as response:
                        response.raise_for_status()
                        parsed_response = await response.json()

                end_time = time.time()
                processing_time = end_time - start_time

                # Add before response_text access
                if 'choices' not in parsed_response or not parsed_response['choices']:
                    raise ValueError("Invalid API response structure")

                choices = parsed_response['choices'][0]

                if 'message' not in choices or 'content' not in choices['message']:
                    raise RuntimeError("Invalid message structure in API response")

                response_text = choices['message']['content']

                # OpenRouter-specific field handling
                reasoning_string = ""
                reasoning_tokens = 0
                if is_openrouter_api:
                    reasoning_string = choices.get('reasoning') or choices.get('message', {}).get('reasoning', "") or ""
                    reasoning_tokens = parsed_response['usage'].get('reasoning_tokens', 0) or len(
                        reasoning_string.split())

                if not isinstance(response_text, str):
                    logging.error(f"Response text is not a string. Type: {type(response_text)}")
                    continue

                # Get logprob
                logprobs = choices.get('logprobs')
                avg_logprob = None

                if logprobs:  # Only calculate if we have actual logprobs
                    if isinstance(logprobs, dict) and 'content' in logprobs:
                        logprobs_content = logprobs['content']
                        try:
                            avg_logprob = sum(t.get('logprob', 0) for t in logprobs_content) / len(logprobs_content)
                        except ZeroDivisionError:
                            avg_logprob = None
                    else:
                        logging.warning(
                            f"Unexpected logprobs format: {type(logprobs)}, {logprobs}.  Skipping avg_logprob calculation.")
                        avg_logprob = None

                # Inner loop for JSON parsing retries
                if json_output:
                    for parsing_attempt in range(self.max_retries):
                        try:
                            # Try to extract JSON
                            if phenotype is None:  # Classic method
                                extracted_json = json_extractor(response_text)

                                # Check if extraction succeeded
                                if extracted_json is not None:
                                    phenotype_status = self._extract_phenotype(extracted_json)
                                    if phenotype_status is not None:
                                        if reasoning:
                                            return_tuple = (extracted_json, processing_time,
                                                            parsed_response['usage']['prompt_tokens'],
                                                            parsed_response['usage']['completion_tokens'],
                                                            response_text, reasoning_string, reasoning_tokens)

                                            return return_tuple

                                        else:
                                            # Successful extraction
                                            return_tuple = (extracted_json, processing_time,
                                                            parsed_response['usage']['prompt_tokens'],
                                                            parsed_response['usage']['completion_tokens'],
                                                            response_text, avg_logprob)
                                            return return_tuple



                            else:  # ETP method
                                extracted_json, validation_percentage = json_extractor(response_text, phenotype)

                                # Check if extraction succeeded
                                if extracted_json is not None:
                                    # Successful extraction
                                    return_tuple = (extracted_json, processing_time,
                                                    parsed_response['usage']['prompt_tokens'],
                                                    parsed_response['usage']['completion_tokens'],
                                                    response_text, validation_percentage, avg_logprob)

                                    return return_tuple

                            # If we're here, extraction failed but didn't raise an exception
                            if parsing_attempt < self.max_retries:
                                logging.error('Uncaught error')
                                # Create a more explicit prompt asking for valid JSON
                                retry_message = (
                                    f"I need a valid JSON response following the correct format. "
                                    f"Your previous response had formatting issues: {response_text}\n\n"
                                )

                                # Update the data_dict with the new message
                                data_dict["messages"] = [{"role": "user", "content": retry_message}]
                                data = json.dumps(data_dict)

                                # Make a new API call
                                logging.warning(
                                    f"JSON extraction failed, retrying with explicit format request (attempt {parsing_attempt + 1}/{self.max_retries})")
                                break  # Break out of parsing retry loop to make a new API call
                            else:
                                logging.error("Max JSON parsing retries reached, returning None")
                                return None, processing_time, parsed_response['usage']['prompt_tokens'], \
                                    parsed_response['usage']['completion_tokens'], response_text

                        except Exception as parsing_error:
                            if parsing_attempt < self.max_retries:
                                # Create a more explicit prompt asking for valid JSON
                                retry_message = (
                                    f"I need a valid JSON response following the correct format. "
                                    f"Your previous response had parsing error: {str(parsing_error)}\n\n"
                                    f"Previous response: {response_text}\n\n"
                                    f"Please respond with properly formatted JSON for the following request:\n{prompt}"
                                )

                                # Update the data_dict with the new message
                                data_dict["messages"] = [{"role": "user", "content": retry_message}]
                                data = json.dumps(data_dict)

                                # Make a new API call
                                logging.warning(
                                    f"JSON parsing error: {parsing_error}, retrying with explicit format request (attempt {parsing_attempt + 1}/{self.max_retries})")
                                break  # Break out of parsing retry loop to make a new API call
                            else:
                                logging.error(f"Max JSON parsing retries reached: {parsing_error}")
                                return None, processing_time, parsed_response['usage']['prompt_tokens'], \
                                    parsed_response['usage']['completion_tokens'], response_text

            except aiohttp.ClientPayloadError as e:
                logging.error(f"Payload error: {e}")
                await asyncio.sleep(backoff_factor)
                backoff_factor *= 2

            except aiohttp.ClientConnectorError as e:
                logging.error(f"Connection failed: {e}, retrying in {backoff_factor}s")
                await asyncio.sleep(backoff_factor)
                backoff_factor *= 2

            except aiohttp.ClientResponseError as e:
                logging.error(f"Response error (status {e.status}): {e}")
                await asyncio.sleep(backoff_factor)
                backoff_factor *= 2

            except aiohttp.ClientError as e:
                if isinstance(e, aiohttp.ClientResponseError):
                    try:
                        error_body = await e.response.text()
                        logging.error(f"HTTP error {e.status}: {e.message}\nResponse body: {error_body}")
                    except Exception as read_error:
                        logging.error(f"Failed to read error body: {read_error}")
                    if e.status == 429:
                        logging.warning(f"Rate limit exceeded. Retrying in {backoff_factor}s...")
                        await asyncio.sleep(backoff_factor)
                        backoff_factor *= 2
                    else:
                        break
                else:
                    logging.error(f"Client error: {str(e)}")
                    break


            except Exception as e:
                logging.error(f"Unexpected error: {type(e).__name__} - {str(e)}")
                logging.error(traceback.format_exc())
                # Re-raise the exception if you want to see the full traceback
                raise

        if phenotype is None:
            return None, 0.0, 0, 0, None, 0.0
        else:
            return None, 0.0, 0, 0, None, 0, 0.0
    async def parse_async(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None, batch_size: int = 10) -> Tuple[
        pd.DataFrame, Optional[pd.DataFrame]]:
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})
        semaphore = asyncio.Semaphore(min(500, batch_size // 2))
        max_reprocess_retries = 1
        
        # Track processed tasks (hadm_id + chunk) to avoid reprocessing successful ones
        processed_tasks = set()
        
        # Helper function to create a unique task ID from hadm_id and chunk
        def create_task_id(hadm_id, chunk):
            # Use first 100 chars of chunk for memory efficiency
            chunk_signature = chunk[:100].strip()
            return f"{hadm_id}_{hash(chunk_signature)}"

        async def _push_to_llm_with_semaphore(prompt, json_extractor, session, hadm_id, task_id, phenotype=None):
            try:
                async with semaphore:
                    result = await self._push_to_llm_async(prompt, json_extractor, session, phenotype)
                    if result is None:
                        logging.error(f"Received None result from _push_to_llm_async for HADM {hadm_id}, task {task_id}")
                        return None
                    return result
            except Exception as e:
                logging.error(f"Error in _push_to_llm_with_semaphore for HADM {hadm_id}, task {task_id}: {str(e)}")
                return None

        async with aiohttp.ClientSession() as session:
            # Process chunks in batches
            for i in range(0, len(chunk_df), batch_size):
                batch = chunk_df.iloc[i:i + batch_size]
                prompts = batch.apply(
                    lambda row: self.prompt_engineering.get_prompt(phenotype, row['chunk']),
                    axis=1
                ).values

                # Create tasks with prompts included in the tuple
                tasks = []
                for hadm_id, chunk, prompt in zip(batch['hadm_id'].values, batch['chunk'].values, prompts):
                    task_id = create_task_id(hadm_id, chunk)
                    if task_id not in processed_tasks:  # Skip already processed tasks
                        tasks.append((
                            hadm_id,
                            chunk,
                            prompt,
                            task_id,
                            _push_to_llm_with_semaphore(
                                prompt,
                                self._extract_json_from_output_classic,
                                session,
                                hadm_id,
                                task_id,
                                phenotype=None
                            )
                        ))
                        # Update total counts
                        parse_counts[hadm_id]['total'] += 1

                # Process all API calls concurrently
                completed_tasks = await asyncio.gather(*(task[4] for task in tasks), return_exceptions=True)

                # Process initial results
                for (hadm_id, chunk, prompt, task_id, _), result in zip(tasks, completed_tasks):
                    try:
                        if isinstance(result, Exception):
                            raise result
                        if result is None:
                            raise ValueError(f"Received None result for HADM {hadm_id}, task {task_id}")

                        # Initialize variables to avoid UnboundLocalError
                        extracted_json = None
                        processing_time = None
                        input_tokens = None
                        output_tokens = None
                        original_response = None
                        reasoning_str = None
                        reasoning_tokens = None
                        avg_logprob = None

                        # Determine result structure based on length
                        if len(result) == 5:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response = result
                            avg_logprob = None
                        elif len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, avg_logprob = result
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens = result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "No JSON identified"
                        else:
                            phenotype_status = self._extract_phenotype(extracted_json)
                            if phenotype_status is None:
                                error_occurred = True
                                error_type = "Failed phenotype extraction"

                        if error_occurred:
                            errors.append({
                                'hadm_id': hadm_id,
                                'chunk': chunk,
                                'prompt': prompt,
                                'task_id': task_id,
                                'raw_output': original_response,
                                'error_type': error_type,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens if input_tokens and output_tokens else None
                            })
                        else:
                            # Successfully processed
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            if len(result) == 7:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'reasoning_text': reasoning_str,
                                    'phenotype_status': phenotype_status,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'reasoning_tokens': reasoning_tokens
                                })
                            else:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'phenotype_status': phenotype_status,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'avg_logprob': avg_logprob
                                })

                    except Exception as e:
                        logging.error(f"Processing error for HADM {hadm_id}, task {task_id}: {str(e)}")
                        errors.append({
                            'hadm_id': hadm_id,
                            'chunk': chunk,
                            'prompt': prompt,
                            'task_id': task_id,
                            'error_type': f"Processing Error: {type(e).__name__}"
                        })

                    if pbar:
                        pbar.update(1)

            # Reprocess errors up to max_reprocess_retries
            for retry_attempt in range(max_reprocess_retries):
                if not errors:
                    break

                logging.info(f"Reprocessing {len(errors)} specific chunk errors (Attempt {retry_attempt + 1}/{max_reprocess_retries})...")
                
                # Track which errors were successfully reprocessed
                successfully_reprocessed = []
                new_errors = []

                # Only reprocess errors for specific tasks that haven't been successfully processed
                reprocess_tasks = []
                for error in errors:
                    task_id = error.get('task_id')
                    if not task_id:
                        # For backward compatibility with errors that don't have task_id
                        task_id = create_task_id(error['hadm_id'], error['chunk'])
                        error['task_id'] = task_id
                    
                    if task_id not in processed_tasks:
                        reprocess_tasks.append((
                            error['hadm_id'],
                            error['chunk'],
                            error['prompt'],
                            task_id,
                            _push_to_llm_with_semaphore(
                                error['prompt'],
                                self._extract_json_from_output_classic,
                                session,
                                error['hadm_id'],
                                task_id,
                                phenotype=None
                            )
                        ))

                if not reprocess_tasks:
                    logging.info("No tasks to reprocess - all specific chunks have been successfully processed")
                    break

                reprocess_completed = await asyncio.gather(*(task[4] for task in reprocess_tasks),
                                                           return_exceptions=True)
                
                for idx, ((hadm_id, chunk, prompt, task_id, _), result) in enumerate(zip(reprocess_tasks, reprocess_completed)):
                    try:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is None:
                            continue
                            
                        if isinstance(result, Exception):
                            logging.error(f"Reprocessing error for task {task_id}: {str(result)}")
                            new_errors.append(errors[error_idx])
                            continue
                        if result is None:
                            raise ValueError(f"Received None result for task {task_id} during reprocessing")

                        # Process the result as before
                        if len(result) == 5:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response = result
                            avg_logprob = None
                        elif len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, avg_logprob = result
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens = result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "Reprocess: No JSON identified"
                        else:
                            phenotype_status = self._extract_phenotype(extracted_json)
                            if phenotype_status is None:
                                error_occurred = True
                                error_type = "Reprocess: Failed phenotype extraction"

                        if error_occurred:
                            new_errors.append(errors[error_idx])
                        else:
                            # Successfully reprocessed
                            successfully_reprocessed.append(task_id)
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            if len(result) == 7:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'reasoning_text': reasoning_str,
                                    'phenotype_status': phenotype_status,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'reasoning_tokens': reasoning_tokens
                                })
                            else:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'phenotype_status': phenotype_status,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'avg_logprob': avg_logprob
                                })

                    except Exception as e:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is not None:
                            logging.error(f"Error during reprocessing for task {task_id}: {str(e)}")
                            new_errors.append(errors[error_idx])

                # Update errors list with only the ones that failed reprocessing
                errors = [error for error in new_errors if error.get('task_id') not in processed_tasks]
                
                # Log reprocessing results
                logging.info(f"Reprocessing attempt {retry_attempt + 1} complete:")
                logging.info(f"- Successfully reprocessed: {len(successfully_reprocessed)} chunks")
                logging.info(f"- Remaining errors: {len(errors)} chunks")
                
                if not errors:
                    logging.info("All chunk errors successfully reprocessed!")
                    break

        # Create DataFrames
        result_df = pd.DataFrame(results)
        error_df = pd.DataFrame(errors) if errors else None
        if error_df is not None and 'prompt' in error_df.columns:
            error_df = error_df.drop(columns=['prompt'])
            if 'task_id' in error_df.columns:
                error_df = error_df.drop(columns=['task_id'])

        # Calculate parsed ratios
        if not result_df.empty:
            parsed_ratios = pd.Series({
                hadm_id: counts['success'] / counts['total']
                for hadm_id, counts in parse_counts.items()
            })
            result_df['parsed_ratio'] = result_df['hadm_id'].map(parsed_ratios)

        return result_df, error_df

    async def parse_uncertainty_async(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None, batch_size: int = 10) -> Tuple[
        pd.DataFrame, Optional[pd.DataFrame]]:
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})
        semaphore = asyncio.Semaphore(min(500, batch_size // 2))
        max_reprocess_retries = 1
        
        # Track processed tasks (hadm_id + chunk) to avoid reprocessing successful ones
        processed_tasks = set()
        
        # Helper function to create a unique task ID from hadm_id and chunk
        def create_task_id(hadm_id, chunk):
            # Use first 100 chars of chunk for memory efficiency
            chunk_signature = chunk[:100].strip()
            return f"{hadm_id}_{hash(chunk_signature)}"

        async def _push_to_llm_with_semaphore(prompt, json_extractor, session, hadm_id, task_id, phenotype=None):
            try:
                async with semaphore:
                    result = await self._push_to_llm_async(prompt, json_extractor, session, phenotype)
                    if result is None:
                        logging.error(f"Received None result from _push_to_llm_async for HADM {hadm_id}, task {task_id}")
                        return None
                    return result
            except Exception as e:
                logging.error(f"Error in _push_to_llm_with_semaphore for HADM {hadm_id}, task {task_id}: {str(e)}")
                return None

        async with aiohttp.ClientSession() as session:
            # Process chunks in batches
            for i in range(0, len(chunk_df), batch_size):
                batch = chunk_df.iloc[i:i + batch_size]
                prompts = batch.apply(
                    lambda row: self.prompt_engineering.get_prompt(phenotype, row['chunk']),
                    axis=1
                ).values

                # Create tasks with prompts included in the tuple
                tasks = []
                for hadm_id, chunk, prompt in zip(batch['hadm_id'].values, batch['chunk'].values, prompts):
                    task_id = create_task_id(hadm_id, chunk)
                    if task_id not in processed_tasks:  # Skip already processed tasks
                        tasks.append((
                            hadm_id,
                            chunk,
                            prompt,
                            task_id,
                            _push_to_llm_with_semaphore(
                                prompt,
                                self._extract_json_from_output_classic,
                                session,
                                hadm_id,
                                task_id,
                                phenotype=None
                            )
                        ))
                        # Update total counts
                        parse_counts[hadm_id]['total'] += 1

                # Process all API calls concurrently
                completed_tasks = await asyncio.gather(*(task[4] for task in tasks), return_exceptions=True)

                # Process initial results
                for (hadm_id, chunk, prompt, task_id, _), result in zip(tasks, completed_tasks):
                    try:
                        if isinstance(result, Exception):
                            raise result
                        if result is None:
                            raise ValueError(f"Received None result for HADM {hadm_id}, task {task_id}")

                        # Initialize variables to avoid UnboundLocalError
                        extracted_json = None
                        processing_time = None
                        input_tokens = None
                        output_tokens = None
                        original_response = None
                        reasoning_str = None
                        reasoning_tokens = None
                        avg_logprob = None

                        # Determine result structure based on length
                        if len(result) == 5:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response = result
                            avg_logprob = None
                        elif len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, avg_logprob = result
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens = result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "No JSON identified"
                        else:
                            phenotype_status = self._extract_phenotype(extracted_json)
                            confidence = self._extract_confidence(extracted_json)
                            justification = self._extract_justification(extracted_json)

                            if phenotype_status is None:
                                error_occurred = True
                                error_type = "Failed phenotype extraction"

                        if error_occurred:
                            errors.append({
                                'hadm_id': hadm_id,
                                'chunk': chunk,
                                'prompt': prompt,
                                'task_id': task_id,
                                'raw_output': original_response,
                                'error_type': error_type,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens if input_tokens and output_tokens else None
                            })
                        else:
                            # Successfully processed
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            if len(result) == 7:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'reasoning_text': reasoning_str,
                                    'phenotype_status': phenotype_status,
                                    'justification': justification,
                                    'confidence': confidence,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'reasoning_tokens': reasoning_tokens
                                })
                            else:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'phenotype_status': phenotype_status,
                                    'justification': justification,
                                    'confidence': confidence,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'avg_logprob': avg_logprob
                                })

                    except Exception as e:
                        logging.error(f"Processing error for HADM {hadm_id}, task {task_id}: {str(e)}")
                        errors.append({
                            'hadm_id': hadm_id,
                            'chunk': chunk,
                            'prompt': prompt,
                            'task_id': task_id,
                            'error_type': f"Processing Error: {type(e).__name__}"
                        })

                    if pbar:
                        pbar.update(1)

            # Reprocess errors up to max_reprocess_retries
            for retry_attempt in range(max_reprocess_retries):
                if not errors:
                    break

                logging.info(f"Reprocessing {len(errors)} specific chunk errors (Attempt {retry_attempt + 1}/{max_reprocess_retries})...")
                
                # Track which errors were successfully reprocessed
                successfully_reprocessed = []
                new_errors = []

                # Only reprocess errors for specific tasks that haven't been successfully processed
                reprocess_tasks = []
                for error in errors:
                    task_id = error.get('task_id')
                    if not task_id:
                        # For backward compatibility with errors that don't have task_id
                        task_id = create_task_id(error['hadm_id'], error['chunk'])
                        error['task_id'] = task_id
                    
                    if task_id not in processed_tasks:
                        reprocess_tasks.append((
                            error['hadm_id'],
                            error['chunk'],
                            error['prompt'],
                            task_id,
                            _push_to_llm_with_semaphore(
                                error['prompt'],
                                self._extract_json_from_output_classic,
                                session,
                                error['hadm_id'],
                                task_id,
                                phenotype=None
                            )
                        ))

                if not reprocess_tasks:
                    logging.info("No tasks to reprocess - all specific chunks have been successfully processed")
                    break

                reprocess_completed = await asyncio.gather(*(task[4] for task in reprocess_tasks),
                                                           return_exceptions=True)
                
                for idx, ((hadm_id, chunk, prompt, task_id, _), result) in enumerate(zip(reprocess_tasks, reprocess_completed)):
                    try:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is None:
                            continue
                            
                        if isinstance(result, Exception):
                            logging.error(f"Reprocessing error for task {task_id}: {str(result)}")
                            new_errors.append(errors[error_idx])
                            continue
                        if result is None:
                            raise ValueError(f"Received None result for task {task_id} during reprocessing")

                        # Process the result as before
                        if len(result) == 5:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response = result
                            avg_logprob = None
                        elif len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, avg_logprob = result
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens = result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "Reprocess: No JSON identified"
                        else:
                            phenotype_status = self._extract_phenotype(extracted_json)
                            confidence = self._extract_confidence(extracted_json)
                            justification = self._extract_justification(extracted_json)

                            if phenotype_status is None:
                                error_occurred = True
                                error_type = "Reprocess: Failed phenotype extraction"

                        if error_occurred:
                            new_errors.append(errors[error_idx])
                        else:
                            # Successfully reprocessed
                            successfully_reprocessed.append(task_id)
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            if len(result) == 7:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'reasoning_text': reasoning_str,
                                    'phenotype_status': phenotype_status,
                                    'justification': justification,
                                    'confidence': confidence,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'reasoning_tokens': reasoning_tokens
                                })
                            else:
                                results.append({
                                    'hadm_id': hadm_id,
                                    'chunk': chunk,
                                    'llm_response': extracted_json,
                                    'phenotype_status': phenotype_status,
                                    'justification': justification,
                                    'confidence': confidence,
                                    'processing_time': processing_time,
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'raw_output': original_response,
                                    'avg_logprob': avg_logprob
                                })

                    except Exception as e:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is not None:
                            logging.error(f"Error during reprocessing for task {task_id}: {str(e)}")
                            new_errors.append(errors[error_idx])

                # Update errors list with only the ones that failed reprocessing
                errors = [error for error in new_errors if error.get('task_id') not in processed_tasks]
                
                # Log reprocessing results
                logging.info(f"Reprocessing attempt {retry_attempt + 1} complete:")
                logging.info(f"- Successfully reprocessed: {len(successfully_reprocessed)} chunks")
                logging.error(f"- Remaining errors: {len(errors)} chunks")
                
                if not errors:
                    logging.error("All chunk errors successfully reprocessed!")
                    break

        # Create DataFrames
        result_df = pd.DataFrame(results)
        error_df = pd.DataFrame(errors) if errors else None
        if error_df is not None and 'prompt' in error_df.columns:
            error_df = error_df.drop(columns=['prompt'])
            if 'task_id' in error_df.columns:
                error_df = error_df.drop(columns=['task_id'])

        # Calculate parsed ratios
        if not result_df.empty:
            parsed_ratios = pd.Series({
                hadm_id: counts['success'] / counts['total']
                for hadm_id, counts in parse_counts.items()
            })
            result_df['parsed_ratio'] = result_df['hadm_id'].map(parsed_ratios)

        return result_df, error_df

    async def extract_then_phenotype_async(self, chunk_df: pd.DataFrame, phenotype: str, pbar=None,
                                           batch_size: int = 10) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Async version of extract_then_phenotype that processes chunks in batches with reprocessing.
        """
        results = []
        errors = []
        parse_counts = defaultdict(lambda: {'success': 0, 'total': 0})
        semaphore = asyncio.Semaphore(max(500, batch_size // 2))
        max_reprocess_retries = 1

        # Track processed tasks (hadm_id + chunk) to avoid reprocessing successful ones
        processed_tasks = set()

        # Helper function to create a unique task ID from hadm_id and chunk
        def create_task_id(hadm_id, chunk):
            # Use first 100 chars of chunk for memory efficiency
            chunk_signature = chunk[:100].strip()
            return f"{hadm_id}_{hash(chunk_signature)}"

        async def _push_to_llm_with_semaphore(prompt, json_extractor, session, hadm_id, task_id, phenotype=None):
            try:
                async with semaphore:
                    result = await self._push_to_llm_async(prompt, json_extractor, session, phenotype)
                    if result is None:
                        logging.error(f"Received None result from _push_to_llm_async for HADM {hadm_id}, task {task_id}")
                        return None
                    return result
            except Exception as e:
                logging.error(f"Error in _push_to_llm_with_semaphore for HADM {hadm_id}, task {task_id}: {str(e)}")
                return None

        async with aiohttp.ClientSession() as session:
            # Process chunks in batches
            for i in range(0, len(chunk_df), batch_size):
                batch = chunk_df.iloc[i:i + batch_size]

                prompts = batch.apply(
                    lambda row: self.prompt_engineering.get_prompt(phenotype, row['chunk']),
                    axis=1
                ).values

                # Create tasks with prompts included in the tuple
                tasks = []
                for hadm_id, chunk, prompt in zip(batch['hadm_id'].values, batch['chunk'].values, prompts):
                    task_id = create_task_id(hadm_id, chunk)
                    if task_id not in processed_tasks:  # Skip already processed tasks
                        tasks.append((
                            hadm_id,
                            chunk,
                            prompt,
                            task_id,
                            _push_to_llm_with_semaphore(
                                prompt,
                                self._extract_json_from_output_etp,
                                session,
                                hadm_id,
                                task_id,
                                phenotype
                            )
                        ))
                        # Update total counts
                        parse_counts[hadm_id]['total'] += 1

                # Process all API calls concurrently
                completed_tasks = await asyncio.gather(*(task[4] for task in tasks), return_exceptions=True)

                # Process initial results
                for (hadm_id, chunk, prompt, task_id, _), result in zip(tasks, completed_tasks):
                    try:
                        if isinstance(result, Exception):
                            raise result
                        if result is None:
                            raise ValueError(f"Received None result for HADM {hadm_id}, task {task_id}")

                        # Initialize variables to avoid UnboundLocalError
                        extracted_json = None
                        processing_time = None
                        input_tokens = None
                        output_tokens = None
                        original_response = None
                        reasoning_str = None
                        reasoning_tokens = None
                        avg_logprob = None
                        etp_json_validation_percentage = None

                        # Determine result structure based on length
                        if len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, etp_json_validation_percentage = result
                            avg_logprob = None
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, etp_json_validation_percentage, avg_logprob = result
                        elif len(result) == 8:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens, etp_json_validation_percentage= result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "No JSON identified"
                        else:
                            # For ETP, we calculate phenotype status during aggregation
                            # But store the extracted entities for now
                            phenotype_status = None
                            extracted_entities = extracted_json

                        if error_occurred:
                            errors.append({
                                'hadm_id': hadm_id,
                                'chunk': chunk,
                                'prompt': prompt,
                                'task_id': task_id,
                                'raw_output': original_response,
                                'error_type': error_type,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens if input_tokens and output_tokens else None
                            })
                        else:
                            # Successfully processed
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            results.append({
                                'hadm_id': hadm_id,
                                'chunk': chunk,
                                'llm_response': extracted_entities,
                                'etp_validation_percentage': etp_json_validation_percentage,
                                'processing_time': processing_time,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens,
                                'raw_output': original_response,
                                'avg_logprob': avg_logprob
                            })

                    except Exception as e:
                        logging.error(f"Processing error for HADM {hadm_id}, task {task_id}: {str(e)}")
                        errors.append({
                            'hadm_id': hadm_id,
                            'chunk': chunk,
                            'prompt': prompt,
                            'task_id': task_id,
                            'error_type': f"Processing Error: {type(e).__name__}"
                        })

                    if pbar:
                        pbar.update(1)

            # Reprocess errors up to max_reprocess_retries
            for retry_attempt in range(max_reprocess_retries):
                if not errors:
                    break

                logging.info(f"Reprocessing {len(errors)} specific chunk errors (Attempt {retry_attempt + 1}/{max_reprocess_retries})...")
                
                # Track which errors were successfully reprocessed
                successfully_reprocessed = []
                new_errors = []

                # Only reprocess errors for tasks that haven't been successfully processed
                reprocess_tasks = []
                for error in errors:
                    task_id = error.get('task_id')
                    if not task_id:
                        # For backward compatibility with errors that don't have task_id
                        task_id = create_task_id(error['hadm_id'], error['chunk'])
                        error['task_id'] = task_id
                    
                    if task_id not in processed_tasks:
                        reprocess_tasks.append((
                            error['hadm_id'],
                            error['chunk'],
                            error['prompt'],
                            task_id,
                            _push_to_llm_with_semaphore(
                                error['prompt'],
                                self._extract_json_from_output_etp,
                                session,
                                error['hadm_id'],
                                task_id,
                                phenotype
                            )
                        ))

                if not reprocess_tasks:
                    logging.info("No tasks to reprocess - all specific chunks have been successfully processed")
                    break

                reprocess_completed = await asyncio.gather(*(task[4] for task in reprocess_tasks),
                                                           return_exceptions=True)
                
                for idx, ((hadm_id, chunk, prompt, task_id, _), result) in enumerate(zip(reprocess_tasks, reprocess_completed)):
                    try:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is None:
                            continue
                            
                        if isinstance(result, Exception):
                            logging.error(f"Reprocessing error for task {task_id}: {str(result)}")
                            new_errors.append(errors[error_idx])
                            continue
                        if result is None:
                            raise ValueError(f"Received None result for task {task_id} during reprocessing")

                        # Process the result as before
                        if len(result) == 6:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, etp_json_validation_percentage = result
                            avg_logprob = None
                        elif len(result) == 7:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, etp_json_validation_percentage, avg_logprob = result
                        elif len(result) == 8:
                            extracted_json, processing_time, input_tokens, output_tokens, original_response, reasoning_str, reasoning_tokens, etp_json_validation_percentage= result
                        else:
                            raise ValueError(f"Unexpected return values: {len(result)}")

                        error_occurred = False
                        error_type = ""
                        
                        if extracted_json is None:
                            error_occurred = True
                            error_type = "Reprocess: No JSON identified"
                        else:
                            # For ETP, we calculate phenotype status during aggregation
                            # But store the extracted entities for now
                            phenotype_status = None
                            extracted_entities = extracted_json

                        if error_occurred:
                            new_errors.append(errors[error_idx])
                        else:
                            # Successfully reprocessed
                            successfully_reprocessed.append(task_id)
                            processed_tasks.add(task_id)
                            parse_counts[hadm_id]['success'] += 1
                            
                            results.append({
                                'hadm_id': hadm_id,
                                'chunk': chunk,
                                'llm_response': extracted_entities,
                                'etp_validation_percentage': etp_json_validation_percentage,
                                'processing_time': processing_time,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens,
                                'raw_output': original_response,
                                'avg_logprob': avg_logprob
                            })

                    except Exception as e:
                        error_idx = next((i for i, e in enumerate(errors) if e.get('task_id') == task_id), None)
                        if error_idx is not None:
                            logging.error(f"Error during reprocessing for task {task_id}: {str(e)}")
                            new_errors.append(errors[error_idx])

                # Update errors list with only the ones that failed reprocessing
                errors = [error for error in new_errors if error.get('task_id') not in processed_tasks]
                
                # Log reprocessing results
                logging.info(f"Reprocessing attempt {retry_attempt + 1} complete:")
                logging.info(f"- Successfully reprocessed: {len(successfully_reprocessed)} chunks")
                logging.info(f"- Remaining errors: {len(errors)} chunks")
                
                if not errors:
                    logging.info("All chunk errors successfully reprocessed!")
                    break

        # Create DataFrames
        result_df = pd.DataFrame(results)
        error_df = pd.DataFrame(errors) if errors else None
        if error_df is not None and 'prompt' in error_df.columns:
            error_df = error_df.drop(columns=['prompt'])

        # Apply ETP normalization to define the extracted fields
        if not result_df.empty:
            try:
                def normalize_response(response):
                    # Extract features from the response if available
                    if isinstance(response, dict):
                        return response
                    elif isinstance(response, str):
                        try:
                            return json.loads(response)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, return it as is
                            return {'raw': response}
                    # If it's neither a dict nor a string, return an empty dict
                    return {}

                # Normalize the extracted_entities column if it exists
                normalized_responses = pd.DataFrame([
                    normalize_response(resp) for resp in result_df['llm_response']
                ])

                # if 'llm_response' in result_df.columns:
                #    result_df['llm_response'] = result_df['llm_response'].apply(normalize_response)

                result_df = pd.concat([result_df.drop('llm_response', axis=1), normalized_responses], axis=1)

                # Calculate parsed ratios
                parsed_ratios = pd.Series({
                    hadm_id: counts['success'] / counts['total']
                    for hadm_id, counts in parse_counts.items()
                })
                result_df['parsed_ratio'] = result_df['hadm_id'].map(parsed_ratios)
            except Exception as e:
                logging.error(f"Error in normalization: {e}")

        return result_df, error_df

class ResultAggregator:
    '''
    A class for aggregating results from multiple text chunks into a single result.
    '''

    def __init__(self, aggregation_method: str = 'any'):
        self.aggregation_method = aggregation_method
        self.fixed_etp_threshold = 0.4
        self.phenotype = None
        self.etp_rule_type = 'fixed'
        self.ETP_LEARNED_PATH = None

    def set_etp_threshold(self, fixed_etp_threshold):
        self.fixed_etp_threshold = fixed_etp_threshold
    def set_phenotype(self, phenotype):
        self.phenotype = phenotype
    def set_etp_rule_type(self, etp_rule_type):
        self.etp_rule_type = etp_rule_type

    def set_etp_learned_path(self, etp_learned_path):
        self.ETP_LEARNED_PATH = etp_learned_path

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results from multiple text chunks into a single result.
        Works with both standard and reasoning outputs (which don't have parsed_ratio).
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for aggregation")
            return df

        # Calculate token metrics
        token_metrics = self._calculate_token_metrics(df)

        # Get phenotype status based on aggregation method
        if self.aggregation_method == 'any':
            phenotype_status_df = self._aggregate_any(df)
        elif self.aggregation_method == 'two':
            phenotype_status_df = self._aggregate_two(df)
        elif self.aggregation_method == 'majority':
            phenotype_status_df = self._aggregate_majority(df)
        elif self.aggregation_method == 'etp':
            phenotype_status_df = self._aggregate_etp(df)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Merge results
        result = phenotype_status_df.merge(token_metrics, on='hadm_id', how='left')

        # Add parsed_ratio if it exists in the input DataFrame
        if 'parsed_ratio' in df.columns:
            parsed_ratio = self._calculate_parsed_ratio(df)
            result = result.merge(parsed_ratio, on='hadm_id', how='left')

        if 'confidence' in df.columns:
            confidence = self._calculate_confidence(df)
            result = result.merge(confidence, on='hadm_id', how='left')
        return result

    def _calculate_parsed_ratio_old(self, df):
        '''
        Calculate the number of parsed chunks for each HADM_ID and the ratio of parsed chunks.
        '''
        df['parsed'] = df.apply(lambda row: all(cell != '' and pd.notna(cell) for cell in row), axis=1)
        return df.groupby('hadm_id')['parsed'].agg(
            lambda x: np.mean(x) if len(x) > 1 else (1.0 if x.iloc[0] else 0.0)
        ).reset_index(name='parsed_chunks_ratio')

    def _calculate_parsed_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the mean parsed ratio for each hadm_id.

        :param df: Input DataFrame containing 'hadm_id' and 'parsed_ratio' columns
        :return: DataFrame with 'hadm_id' and mean 'parsed_ratio' for each hadm_id
        """
        if 'hadm_id' not in df.columns or 'parsed_ratio' not in df.columns:
            raise ValueError("Input DataFrame must contain 'hadm_id' and 'parsed_ratio' columns")

        # Group by 'hadm_id' and calculate the mean of 'parsed_ratio'
        parsed_ratio_df = df.groupby('hadm_id')['parsed_ratio'].mean().reset_index()
        return parsed_ratio_df

    def _calculate_token_metrics(self, df):
        """Calculate token and processing metrics."""
        metrics_columns = [
            'chunk_tokens', 'input_tokens', 'output_tokens',
            'total_tokens', 'processing_time', 'avg_logprob', 'hadm_id'
        ]

        # Only include columns that exist in the DataFrame
        available_columns = [col for col in metrics_columns if col in df.columns]

        token_metrics = df.groupby('hadm_id').agg({
            col: ['mean', 'std'] if col != 'hadm_id' else 'count'
            for col in available_columns
        }).dropna(how='all').reset_index()

        # Flatten column names
        token_metrics.columns = ['hadm_id'] + [
            f'{col[0]}_{col[1]}' if col[1] != '' else col[0]
            for col in token_metrics.columns[1:]
        ]

        # Rename count column
        if 'hadm_id_count' in token_metrics.columns:
            token_metrics = token_metrics.rename(columns={'hadm_id_count': 'n_chunks'})

        return token_metrics

    def _calculate_confidence(self, df):
        """Return the most frequent confidence for each hadm_id.
        
        If there are no confidence values or an empty sequence is encountered,
        'low' is returned as the default value. Also handles special case when 
        chunking is 'none' and there's only a single confidence per HADM_ID.
        """
        # Check if confidence column exists in the dataframe
        if 'confidence' not in df.columns:
            # Create a default dataframe with 'low' confidence for all HADM_IDs
            return pd.DataFrame({
                'hadm_id': df['hadm_id'].unique(),
                'confidence': 'low'
            })
        
        # Define a safe aggregation function that handles empty sequences
        def safe_most_frequent(x):
            if x.empty or len(x) == 0:
                return 'low'  # Default value
            try:
                # Try to get the most frequent value
                counts = x.value_counts()
                if counts.empty:
                    return 'low'
                return counts.idxmax()
            except Exception:
                # In case of any error, return the default
                return 'low'
        
        # Special case: when chunking is 'none', there's only one confidence value per HADM_ID
        # In this case, we take the value directly without aggregation if possible
        if len(df) == len(df['hadm_id'].unique()):
            # No chunking, single confidence per HADM_ID
            # Still need to handle missing values
            return df[['hadm_id', 'confidence']].fillna('low')
        
        # Process normal case with multiple confidences per HADM_ID
        return df.groupby('hadm_id')['confidence'].agg(safe_most_frequent).reset_index(name='confidence')

    def _aggregate_any(self, df):
        df['phenotype_status'] = pd.to_numeric(df['phenotype_status'], errors='coerce')
        return df.groupby('hadm_id')['phenotype_status'].max().reset_index()

    def _aggregate_majority(self, df):
        df['phenotype_status'] = pd.to_numeric(df['phenotype_status'], errors='coerce')
        return df.groupby('hadm_id')['phenotype_status'].apply(
            lambda x: 1 if x.mean() > 0.5 else 0).reset_index()
    def _aggregate_two(self, df):
        df['phenotype_status'] = pd.to_numeric(df['phenotype_status'], errors='coerce')
        return df.groupby('hadm_id')['phenotype_status'].apply(
            lambda x: 1 if x.sum() >=  2 else 0).reset_index()
    def _aggregate_etp(self, df):
        # TODO: dynamic needs the ETP_LEARNED_PATH - learn thresholds not weights
        # Load phenotype information based on the rule type
        if self.etp_rule_type == 'dynamic':
            with open(self.ETP_LEARNED_PATH, 'r') as f:
                all_phenotype_info = json.load(f)
            phenotype_info = all_phenotype_info[self.phenotype]
            phenotype_weights = phenotype_info['weights']
            dynamic_threshold = phenotype_info['threshold']
            maximal_score = sum(phenotype_weights.values())

        else:
            # Use baseline weights for fixed rule type
            phenotype_weights = PHENOTYPES_ETP_SCORES.get(self.phenotype, {})
            maximal_score = sum(phenotype_weights.values())

        phenotype_keys = list(phenotype_weights.keys())

        # Check if data was chunked
        if df.groupby('hadm_id').size().max() > 1:
            aggregation_rules = {key: 'max' for key in phenotype_keys}
            print('debugging')
            aggregation_rules['etp_validation_percentage'] = 'mean' # represents % of parsed keywords for each chunk
            df = df.groupby('hadm_id').agg(aggregation_rules).reset_index()

        def calculate_etp_score(row):
            score = sum((row[key] or 0) * phenotype_weights.get(key, 0)
                        for key in phenotype_keys)
            return score

        def calculate_status(row):
            # Calculate the score based on the current weights
            score = calculate_etp_score(row)
            if self.etp_rule_type == 'fixed':
                min_score = int(maximal_score * self.fixed_etp_threshold)  # Using cutoff to determine the fixed threshold
                return 1 if score >= min_score else 0
            elif self.etp_rule_type == 'dynamic':
                # Use the dynamic threshold loaded from the JSON file
                # TODO : must implement
                return 1 if score >= dynamic_threshold else 0
            else:
                raise ValueError("Invalid rule_type. Choose either 'fixed' or 'dynamic'.")

        df['phenotype_status'] = df.apply(calculate_status, axis=1)
        df['etp_score'] = df.apply(calculate_etp_score, axis=1) # returning raw sum to debug results for different threshold
        df['maximal_score'] = maximal_score

        return df[['hadm_id', 'phenotype_status', 'etp_score', 'maximal_score','etp_validation_percentage']]

    def _aggregate_etp_training(self, df, phenotype):
        # Get the weighting rule and keys for the specified phenotype
        phenotype_weights = PHENOTYPES_ETP_SCORES.get(phenotype, {})
        phenotype_keys = PHENOTYPES_ETP_KEYS.get(phenotype, [])

        # Check if data was chunked (i.e., multiple rows per hadm_id)
        if df.groupby('hadm_id').size().max() > 1:
            # Aggregate the columns per hadm_id
            aggregation_rules = {key: 'max' for key in phenotype_keys}
            aggregation_rules['etp_validation_percentage'] = 'mean'  # Mean for validation percentage

            # Aggregate the dataframe based on the defined rules
            df = df.groupby('hadm_id').agg(aggregation_rules).reset_index()

        # Calculate the score for each hadm_id
        df['score'] = df.apply(lambda row: sum(row[key] * phenotype_weights.get(key, 0) for key in phenotype_keys), axis=1)

        # The DataFrame is now ready to be used for training a model
        return df[['hadm_id', 'score']]

class Evaluator:
    def __init__(self, labels_file: str):
        self.golden_labels_df = pd.read_csv(labels_file)

    def evaluate(self, results_df: pd.DataFrame, phenotype: str) -> Tuple[dict, pd.DataFrame]:
        # Merge results with golden labels on HADM_ID
        if results_df is None or results_df.empty:
            raise ValueError("Results DataFrame is empty or None")

        merged_df = pd.merge(self.golden_labels_df, results_df, on='hadm_id')

        # Extract the relevant columns for the phenotype
        true_labels = merged_df[phenotype].tolist()
        predictions = merged_df['phenotype_status'].tolist()

        # Filter out None values
        valid_indices = [i for i, (t, p) in enumerate(zip(true_labels, predictions))
                         if p is not None and not pd.isna(p) and not pd.isna(t)]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]

        total_n = len(predictions)
        valid_n = len(valid_predictions)

        # Errors recorded during the evaluation
        # Identify misclassified indices
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(valid_true_labels, valid_predictions)) if
                                 true != pred]
        misclassified_hadm_ids = [merged_df.iloc[i]['hadm_id'] for i in misclassified_indices]

        # Identify properly classified indices
        properly_classified_indices = [i for i in range(len(valid_predictions)) if i not in misclassified_indices]
        properly_classified_hadm_ids = [merged_df.iloc[i]['hadm_id'] for i in properly_classified_indices]

        # Create DataFrame for misclassified HADM_IDs
        errors_hadm_ids_df = pd.DataFrame(misclassified_hadm_ids, columns=['hadm_id'])
        proper_hadm_ids_df = pd.DataFrame(properly_classified_hadm_ids, columns=['hadm_id'])

        # Initialize metrics dictionary
        metrics = {}

        # Calculate metrics using valid data
        try:
            metrics['n'] = total_n
            metrics['valid_n'] = valid_n
            metrics['accuracy'] = accuracy_score(valid_true_labels, valid_predictions)
            metrics['precision'] = precision_score(valid_true_labels, valid_predictions, zero_division=0)
            metrics['recall'] = recall_score(valid_true_labels, valid_predictions, zero_division=0)
            metrics['f1_score'] = f1_score(valid_true_labels, valid_predictions, zero_division=0)

            # Compute confusion matrix and extract TP, TN, FP, FN
            cm = confusion_matrix(valid_true_labels, valid_predictions)
            if cm.shape == (2, 2):
                metrics['true_negatives'], metrics['false_positives'], metrics['false_negatives'], metrics['true_positives'] = cm.ravel()
            elif cm.shape == (1, 1):
                # Handle cases where only one class is present
                if valid_true_labels[0] == valid_predictions[0]:
                    if valid_predictions[0] == 1:
                        metrics['true_positives'] = cm[0, 0]
                        metrics['true_negatives'] = metrics['false_positives'] = metrics['false_negatives'] = 0
                    else:
                        metrics['true_negatives'] = cm[0, 0]
                        metrics['true_positives'] = metrics['false_positives'] = metrics['false_negatives'] = 0
                else:
                    metrics['false_positives'] = cm[0, 0] if valid_predictions[0] == 1 else 0
                    metrics['false_negatives'] = cm[0, 0] if valid_predictions[0] == 0 else 0
                    metrics['true_positives'] = metrics['true_negatives'] = 0


            # Calculate token metrics, ignore columns that are not in results_df
            def safe_mean(x):
                return pd.Series(x).dropna().mean()

            def safe_std(x):
                return pd.Series(x).dropna().std()

            token_metric_columns_old = {
                'n_chunks': safe_mean,
                'chunk_tokens_mean': safe_mean,
                'chunk_tokens_sd': safe_std,
                'input_tokens_mean': safe_mean,
                'input_tokens_sd': safe_std,
                'output_tokens_mean': safe_mean,
                'output_tokens_sd': safe_std,
                'processing_time_mean': safe_mean,
                #'avg_logprob': safe_mean,
                #'avg_logprob_sd': safe_std
            }

            token_metric_columns = {
                'n_chunks': 'mean',
                'chunk_tokens_mean': 'mean',
                'chunk_tokens_sd': 'mean',
                'input_tokens_mean': 'mean',
                'input_tokens_sd': 'mean',
                'output_tokens_mean': 'mean',
                'output_tokens_sd': 'mean',
                'processing_time_mean': 'mean',
                # 'avg_logprob': 'mean',
                # 'avg_logprob_sd': 'mean'
            }

            # Filter for columns that exist in the DataFrame
            available_metrics = {
                col: agg_func
                for col, agg_func in token_metric_columns.items()
                if col in results_df.columns
            }

            # Calculate metrics only for available columns
            if available_metrics:
                token_metrics = results_df.agg(available_metrics).to_dict()
            else:
                token_metrics = {}

            metrics.update(token_metrics)

            # Add logprob information
            if 'avg_logprob_mean' in results_df.columns:
                errors_hadm_ids_df = errors_hadm_ids_df.merge(
                    results_df[['hadm_id', 'avg_logprob_mean']],
                    on='hadm_id',
                    how='left'
                )
                proper_hadm_ids_df = proper_hadm_ids_df.merge(
                    results_df[['hadm_id', 'avg_logprob_mean']],
                    on='hadm_id',
                    how='left'
                )


        except Exception as e:
            logging.error(f"Error calculating token metrics: {e}")
            metrics.update({col: None for col in token_metric_columns})

        return metrics, errors_hadm_ids_df, proper_hadm_ids_df

class ExperimentConfig:
    def __init__(self, phenotype: str, chunking_method: str, aggregation_method: str, prompt_strategy: str, model_name: str):
        self.phenotype = phenotype
        self.chunking_method = chunking_method
        self.aggregation_method = aggregation_method
        self.prompt_strategy = prompt_strategy
        self.model_name = model_name

    def __str__(self):
        return f"{self.phenotype}-{self.model_name}-{self.prompt_strategy}-{self.chunking_method}-{self.aggregation_method}"


class EvaluationManager:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def run_experiment(self, results_df: pd.DataFrame, config: ExperimentConfig, phenotype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        metrics, errors, properclassifications = self.evaluator.evaluate(results_df, phenotype)
        metrics_df = pd.DataFrame([metrics])
        metrics_df['phenotype'] = phenotype
        metrics_df['config'] = str(config)
        # Splitting config into columns
        metrics_df[['phenotype', 'model', 'prompt_strategy', 'chunking_method', 'aggregation_method']] = metrics_df['config'].str.split('-', expand=True)

        column_order = ['phenotype', 'config', 'model', 'prompt_strategy', 'chunking_method', 'aggregation_method'] + [col for col in metrics_df.columns if
                                                  col not in ['Phenotype', 'Config']]
        metrics_df = metrics_df[column_order]
        return metrics_df, errors, properclassifications


def summarize_phenotype_performance(phenotype_metrics_df, phenotype, models):
    '''Returns a DataFrame with expanded configuration columns and all metrics for a given phenotype in a single csv file'''
    # Initialize an empty DataFrame to store expanded configuration components
    config_components = pd.DataFrame(index=phenotype_metrics_df.index)

    # Iterate over each configuration string
    for index, config in phenotype_metrics_df['Config'].items():
        # Split the configuration string by hyphens
        parts = config.split('-')

        # Identify the model by checking against the list of models
        model_name = next((part for part in parts if part in models), None)

        # If a model name is found, reconstruct the configuration components
        if model_name:
            model_index = parts.index(model_name)
            phenotype_name = parts[0]
            prompt_strategy = parts[model_index + 1]
            chunking_method = parts[model_index + 2]
            aggregation_method = parts[model_index + 3]

            # Assign the extracted components to the DataFrame
            config_components.loc[index, 'Phenotype'] = phenotype_name
            config_components.loc[index, 'ModelName'] = model_name
            config_components.loc[index, 'ChunkingMethod'] = chunking_method
            config_components.loc[index, 'PromptStrategy'] = prompt_strategy
            config_components.loc[index, 'AggregationMethod'] = aggregation_method

    # Concatenate the original DataFrame with the new configuration components
    expanded_df = pd.concat([config_components, phenotype_metrics_df.drop(columns=['Config', 'Phenotype'])], axis=1)

    phenotype_summary_path = generate_metrics_summary_path(level='phenotype', phenotype=phenotype)
    save_csv(expanded_df, phenotype_summary_path)

    return expanded_df

def aggregate_error_csvs(phenotypes, base_error_dir=ERRORS_PATH):
    def custom_split(filename):
        prompt_strategies = ['zero_shot', 'few_shot', 'COT', 'etp']
        # Split once at the first underscore
        first_split = filename.split('_', 1)

        if len(first_split) < 2:
            return first_split  # Return the list as is if there's no underscore to split on

        # Check if the second part starts with any of the prompt strategies
        for strategy in prompt_strategies:
            if first_split[1].startswith(strategy):
                # If it starts with a strategy, split the rest after the strategy
                second_part = first_split[1][len(strategy):]
                if second_part.startswith('_'):
                    second_part = second_part[1:]  # Remove leading underscore if present
                return [first_split[0], strategy, second_part]

        # If no strategy matches, perform a normal second split
        second_split = first_split[1].split('_', 1)
        return [first_split[0]] + second_split

    error_data = []
    try:
        for phenotype in phenotypes:
            phenotype_error_dir = os.path.join(base_error_dir, phenotype)
            for root, dirs, files in os.walk(phenotype_error_dir):
                for file in files:
                    if file.endswith('.csv'):
                        filepath = os.path.join(root, file)
                        try:
                            df = pd.read_csv(filepath)
                            filename_with_extension = os.path.basename(filepath)
                            filename_without_extension, _ = os.path.splitext(filename_with_extension)
                            filename_without_extension = filename_without_extension[:-7]

                            parts = custom_split(filename_without_extension)

                            model = parts[-1]
                            config = parts[0] + '/' + parts[1]

                            df['Model'] = model
                            df['Phenotype'] = phenotype
                            df['Configuration'] = config

                            error_data.append(df)
                        except Exception as e:
                            print(f"Failed to read {filepath}: {e}")

        if error_data:
            all_errors_df = pd.concat(error_data, ignore_index=True)
        else:
            all_errors_df = pd.DataFrame()

        return all_errors_df
    except Exception as e:
        print("Failed to aggregate errors CSVs.", e)
        return pd.DataFrame()


def analyze_misclassification_errors(file_path, total_cases_per_config):
    """
    Analyze misclassification errors with detailed grouping and error rates.

    Args:
        file_path: Path to the CSV file with error data
        total_cases_per_config: Number of total discharge summaries for a single configuration
    """
    # Load and prepare the data
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'Phenotype': 'phenotype',
        'ModelName': 'model',
        'ChunkingMethod': 'chunking_method',
        'PromptStrategy': 'prompt_strategy',
        'AggregationMethod': 'aggregation_method'
    })

    # Calculate total configurations for proper rate calculations
    configs = df[['phenotype', 'model', 'chunking_method', 'prompt_strategy', 'aggregation_method']].drop_duplicates()
    total_configs = len(configs)

    def calculate_group_metrics(group_col):
        """Helper function to calculate metrics for each unique value in a column"""
        metrics_list = []

        # Get unique values of the grouping column
        unique_values = df[group_col].unique()

        for value in unique_values:
            # Filter for this value
            subset = df[df[group_col] == value]

            # Count the unique configurations with this value
            value_configs = subset[['phenotype', 'model', 'chunking_method',
                                    'prompt_strategy', 'aggregation_method']].drop_duplicates()
            num_configs = len(value_configs)

            # Count errors for this value
            total_errors = len(subset)
            unique_cases = len(subset['hadm_id'].unique())

            # Calculate the actual denominator (total cases across all configs with this value)
            denominator = total_cases_per_config * num_configs

            metrics_list.append({
                group_col: value,
                'total_errors': total_errors,
                'unique_cases': unique_cases,
                'error_rate': total_errors / denominator if denominator > 0 else 0,
                'phenotypes': ', '.join(sorted(subset['phenotype'].unique())),
                'configs': num_configs
            })

        return pd.DataFrame(metrics_list)

    # Initialize results dictionary
    results = {}

    # Analyze by each dimension
    results['model_analysis'] = calculate_group_metrics('model')
    results['chunking_analysis'] = calculate_group_metrics('chunking_method')
    results['prompt_analysis'] = calculate_group_metrics('prompt_strategy')
    results['aggregation_analysis'] = calculate_group_metrics('aggregation_method')
    results['phenotype_analysis'] = calculate_group_metrics('phenotype')

    # 6. Group by HADM_ID with detailed error analysis
    hadm_analysis = df.groupby('hadm_id').agg({
        'phenotype': lambda x: ', '.join(sorted(set(x))),
        'model': lambda x: ', '.join(sorted(set(x))),
        'chunking_method': lambda x: ', '.join(sorted(set(x))),
        'prompt_strategy': lambda x: ', '.join(sorted(set(x))),
        'aggregation_method': lambda x: ', '.join(sorted(set(x)))
    })

    hadm_analysis['error_count'] = df.groupby('hadm_id').size()

    # For each HADM_ID, calculate how many of the total configs misclassified it
    hadm_analysis['error_percentage'] = hadm_analysis['error_count'] / total_configs
    hadm_analysis['total_configs'] = total_configs

    hadm_analysis = hadm_analysis.reset_index()
    results['hadm_analysis'] = hadm_analysis

    return results


def filter_incomplete_parsing(phenotype_predictions, phenotype, model_name, output_path=PARSING_REPORT_PATH):
    """
    Filters out rows with parsed_ratio < 1.0 and logs information about removed rows.

    Args:
        phenotype_predictions: Dictionary with configurations as keys and dataframes as values
        phenotype: Current phenotype being processed
        model_name: Current model being used
        output_path: Base path for saving log files

    Returns:
        Dictionary with filtered dataframes
    """
    # Create logs directory
    log_dir = f"{output_path}{phenotype}/{model_name}"
    os.makedirs(f"{log_dir}/removed_rows", exist_ok=True)
    os.makedirs(f"{log_dir}/hadm_ids", exist_ok=True)

    # First, deduplicate configurations by ignoring aggregation key
    # Group by chunk_type and prompt_type, keeping only one per group
    deduplicated_configs = {}
    config_map = {}  # Maps unique configs to original configs for final output

    for config, df in phenotype_predictions.items():
        if df is None or df.empty or 'parsed_ratio' not in df.columns:
            # Keep null/empty configs unchanged
            deduplicated_configs[config] = df
            continue

        # Extract the chunks and prompt strategy (first two elements of config tuple)
        unique_key = (config[0], config[1])  # Ignore aggregation method

        # Either add this as a new unique config or skip it if already seen
        if unique_key not in config_map:
            deduplicated_configs[config] = df
            config_map[unique_key] = config

    logging.info(f"Deduplicated {len(phenotype_predictions)} configs to {len(deduplicated_configs)} unique configs")

    # Track overall stats for model summary
    model_stats = {
        'model': model_name,
        'phenotype': phenotype,
        'total_chunks': 0,
        'removed_chunks': 0,
        'total_patients': set(),
        'affected_patients': set()
    }

    # Track chunks per config type
    chunk_type_stats = {}
    filtered_predictions = {}
    config_stats = []
    config_count = len(deduplicated_configs)  # Count unique configurations

    # Process each deduplicated configuration
    for config, df in deduplicated_configs.items():
        if df is None or df.empty or 'parsed_ratio' not in df.columns:
            # Keep null configs in the final output
            filtered_predictions[config] = df
            continue

        # Extract config components
        chunk_type = config[0]
        prompt_type = config[1]
        config_str = f"{chunk_type}_{prompt_type}"

        # Track chunk types
        if chunk_type not in chunk_type_stats:
            chunk_type_stats[chunk_type] = {
                'total_chunks': 0,
                'removed_chunks': 0,
                'configs': 0,
                'phenotypes': set([phenotype]),
                'patient_count': 0  # We'll track unique patients per chunk type
            }
        chunk_type_stats[chunk_type]['configs'] += 1

        # Get all patients in this config
        all_patients = set(df['hadm_id'].unique())
        patient_count = len(all_patients)

        # Update the patient count for this chunk type (use max to handle overlap)
        chunk_type_stats[chunk_type]['patient_count'] = max(
            chunk_type_stats[chunk_type]['patient_count'],
            patient_count
        )

        # Count chunks per patient to accurately calculate avg_chunks_per_patient
        chunks_per_patient = df.groupby('hadm_id')['n_chunks'].sum()
        avg_chunks_per_patient = chunks_per_patient.mean() if not chunks_per_patient.empty else 1.0

        # Get min and max chunks for logging
        min_chunks = chunks_per_patient.min() if not chunks_per_patient.empty else 0
        max_chunks = chunks_per_patient.max() if not chunks_per_patient.empty else 0

        # Log chunk distribution for debugging
        logging.info(f"Config {config_str} chunks per patient: mean={avg_chunks_per_patient:.2f}, " +
                     f"min={min_chunks}, max={max_chunks}, patients={patient_count}")

        # For chunking methods other than 'none', use chunk-specific metrics
        is_chunked = chunk_type not in ['none', 'None', None, '']

        # Find incomplete rows
        incomplete_rows = df[df['parsed_ratio'] < 1.0]
        affected_patients = set(incomplete_rows['hadm_id'].unique())

        # Get counts
        total_rows = df['n_chunks'].sum() if 'n_chunks' in df.columns else len(df)
        unparsed_per_row = (
                incomplete_rows['n_chunks'] / incomplete_rows['parsed_ratio'].replace(0, 1)  # Avoid division by zero
                - incomplete_rows['n_chunks']
        ).fillna(0)  # Handle NaN cases

        # Sum across all rows to get TOTAL unparsed chunks
        removed_rows = unparsed_per_row.sum().astype(int)

        # Update chunk type stats
        chunk_type_stats[chunk_type]['total_chunks'] += total_rows
        chunk_type_stats[chunk_type]['removed_chunks'] += removed_rows

        # Calculate percentages
        chunk_removal_pct = (removed_rows / total_rows * 100) if total_rows > 0 else 0
        patient_removal_pct = (len(affected_patients) / len(all_patients) * 100) if all_patients else 0

        # Update model stats
        model_stats['total_chunks'] += total_rows
        model_stats['removed_chunks'] += removed_rows
        model_stats['total_patients'].update(all_patients)
        model_stats['affected_patients'].update(affected_patients)

        # Save config stats
        config_stats.append({
            'config': config_str,
            'chunk_type': chunk_type,
            'prompt_type': prompt_type,
            'total_chunks': total_rows,
            'removed_chunks': removed_rows,
            'chunk_removal_pct': chunk_removal_pct,
            'avg_chunks_per_patient': avg_chunks_per_patient,
            'total_patients': patient_count,
            'affected_patients': len(affected_patients),
            'patient_removal_pct': patient_removal_pct,
            'min_chunks_per_patient': min_chunks,
            'max_chunks_per_patient': max_chunks,
            'is_chunked': is_chunked
        })

        if not incomplete_rows.empty:
            # Save removed rows
            save_csv(incomplete_rows, f"{log_dir}/removed_rows/{config_str}_removed_rows.csv")

            # Save affected hadm_ids to a separate file
            affected_ids = pd.DataFrame({'hadm_id': list(affected_patients)})
            save_csv(affected_ids, f"{log_dir}/hadm_ids/{config_str}_affected_hadm_ids.csv")

        # Filter the current config
        filtered_df = df[df['parsed_ratio'] == 1.0].copy()

        # Apply the filtering to all configs
        filtered_predictions[config] = filtered_df

        # Update any other configs with the same chunk/prompt type
        for full_config in phenotype_predictions.keys():
            if full_config != config and full_config[0] == chunk_type and full_config[1] == prompt_type:
                orig_df = phenotype_predictions[full_config]
                if orig_df is not None and not orig_df.empty and 'parsed_ratio' in orig_df.columns:
                    filtered_predictions[full_config] = orig_df[orig_df['parsed_ratio'] == 1.0].copy()

    # Save config summary
    if config_stats:
        config_df = pd.DataFrame(config_stats)

        # Enhance the config summary with additional metrics
        config_df['document_multiplier'] = config_df.apply(
            lambda x: x['total_chunks'] / x['total_patients'] if x['total_patients'] > 0 else 1.0,
            axis=1
        )

        # Log key config stats
        for _, row in config_df.iterrows():
            logging.info(
                f"Config {row['config']}: {row['total_chunks']} chunks "
                f"({row['avg_chunks_per_patient']:.2f} per patient, range {row['min_chunks_per_patient']}-{row['max_chunks_per_patient']}) "
                f"for {row['total_patients']} patients"
            )

        # Save config summary with detailed metrics
        save_csv(config_df, f"{log_dir}/config_summary.csv")

    # Update chunk_type_stats with avg_chunks_per_patient
    # This needs to be calculated from the configuration data
    for chunk_type, stats in chunk_type_stats.items():
        matching_configs = [row for row in config_stats if row['chunk_type'] == chunk_type]

        if matching_configs:
            # Calculate weighted average chunks per patient across configs
            total_chunks = sum(cfg['total_chunks'] for cfg in matching_configs)
            total_patients = stats['patient_count']  # Use the max patient count for this chunk type

            if total_patients > 0:
                # Direct calculation from total chunks / patients
                avg_chunks_per_patient = total_chunks / total_patients
            else:
                avg_chunks_per_patient = 1.0

            # Store in the chunk type stats
            stats['avg_chunks_per_patient'] = avg_chunks_per_patient

            # Log for debugging
            logging.info(f"Chunk type {chunk_type}: {total_chunks} chunks for {total_patients} patients "
                         f"= {avg_chunks_per_patient:.2f} chunks per patient")
        else:
            stats['avg_chunks_per_patient'] = 1.0

    # Calculate overall metrics
    overall_chunk_pct = (model_stats['removed_chunks'] / model_stats['total_chunks'] * 100) if model_stats[
                                                                                                   'total_chunks'] > 0 else 0
    overall_patient_pct = (len(model_stats['affected_patients']) / len(model_stats['total_patients']) * 100) if \
    model_stats['total_patients'] else 0

    # Calculate overall average chunks per patient
    # Weighted average based on chunk type stats
    weighted_avg_chunks = 0
    total_weight = 0

    for chunk_type, stats in chunk_type_stats.items():
        weighted_avg_chunks += stats['avg_chunks_per_patient'] * stats['total_chunks']
        total_weight += stats['total_chunks']

    overall_avg_chunks_per_patient = weighted_avg_chunks / total_weight if total_weight > 0 else 1.0

    # Create new entry for model-phenotype summary
    new_entry = pd.DataFrame([{
        'model': model_name,
        'phenotype': phenotype,
        'config_count': int(len(deduplicated_configs)),  # Only count unique configs
        'total_chunks': model_stats['total_chunks'],
        'removed_chunks': model_stats['removed_chunks'],
        'chunk_removal_pct': overall_chunk_pct,
        'avg_chunks_per_patient': overall_avg_chunks_per_patient,
        'total_patients': int(len(model_stats['total_patients'])),
        'affected_patients': int(len(model_stats['affected_patients'])),
        'patient_removal_pct': overall_patient_pct
    }])

    # Update model summary files
    update_summaries(model_name, phenotype, new_entry, output_path, chunk_type_stats,
                     model_stats['total_patients'], model_stats['affected_patients'])

    # Log overall summary
    logging.info(
        f"Overall {phenotype}/{model_name}: "
        f"Processed {len(deduplicated_configs)} unique configurations with {model_stats['total_chunks']} total chunks "
        f"({overall_avg_chunks_per_patient:.2f} per patient) and {len(model_stats['total_patients'])} unique patients"
    )

    return filtered_predictions


def update_summaries(model_name, phenotype, new_entry, output_path, chunk_stats, total_patients, affected_patients):
    """Update all summary files with consistent metrics"""
    # Ensure directory exists
    summary_dir = f"{output_path}overall"
    os.makedirs(summary_dir, exist_ok=True)

    # Convert numpy types to native Python types before JSON serialization
    chunk_stats_json = {}
    for chunk_type, stats in chunk_stats.items():
        chunk_stats_json[chunk_type] = {
            'total_chunks': int(stats['total_chunks']),  # Convert to Python int
            'removed_chunks': int(stats['removed_chunks']),
            'configs': int(stats['configs']),
            'phenotypes': list(stats['phenotypes']),
            'avg_chunks_per_patient': float(stats.get('avg_chunks_per_patient', 1.0)),
            'patient_count': int(stats.get('patient_count', 0))
        }
    # Add patient IDs to the entry for model-level aggregation
    patient_entry = new_entry.copy()
    patient_entry['patient_ids'] = ','.join(map(str, total_patients)) if total_patients else ''
    patient_entry['affected_ids'] = ','.join(map(str, affected_patients)) if affected_patients else ''
    patient_entry['chunk_stats'] = json.dumps(chunk_stats_json)

    # 1. Update model-phenotype summary
    model_phenotype_path = f"{summary_dir}/model_phenotype_summary.csv"

    if os.path.exists(model_phenotype_path):
        try:
            mp_summary = pd.read_csv(model_phenotype_path)
            # Remove existing entry for this model-phenotype combination
            mp_summary = mp_summary[~((mp_summary['model'] == model_name) &
                                      (mp_summary['phenotype'] == phenotype))]
            # Add new entry
            mp_summary = pd.concat([mp_summary, patient_entry], ignore_index=True)
        except:
            mp_summary = patient_entry
    else:
        mp_summary = patient_entry

    # Save model-phenotype summary
    save_csv(mp_summary, model_phenotype_path)

    # 2. Generate model-only summary by aggregating model-phenotype data
    try:
        model_groups = mp_summary.groupby('model')
        model_rows = []

        model_chunk_data = {}  # Store chunk stats per model

        for model, group in model_groups:
            # Collect patient IDs to correctly count unique patients
            all_patients = set()
            affected_patients = set()

            # Track chunk stats for this model
            model_chunk_stats = {}

            # Process all phenotypes for this model
            for _, row in group.iterrows():
                # Add patient IDs
                if pd.notna(row.get('patient_ids')) and row['patient_ids']:
                    all_patients.update(row['patient_ids'].split(','))
                if pd.notna(row.get('affected_ids')) and row['affected_ids']:
                    affected_patients.update(row['affected_ids'].split(','))

                # Process chunk stats
                if pd.notna(row.get('chunk_stats')) and row['chunk_stats']:
                    try:
                        chunk_data = json.loads(row['chunk_stats'])
                        for chunk_type, stats in chunk_data.items():
                            if chunk_type not in model_chunk_stats:
                                model_chunk_stats[chunk_type] = {
                                    'total_chunks': 0,
                                    'removed_chunks': 0,
                                    'configs': 0,
                                    'phenotypes': set(),
                                    'avg_chunks_per_patient': 0,
                                    'patient_count': 0
                                }

                            # Update stats
                            model_chunk_stats[chunk_type]['total_chunks'] += stats['total_chunks']
                            model_chunk_stats[chunk_type]['removed_chunks'] += stats['removed_chunks']
                            model_chunk_stats[chunk_type]['configs'] += stats['configs']
                            model_chunk_stats[chunk_type]['phenotypes'].update(stats['phenotypes'])

                            # Use max patient count since this represents unique patients for this chunk type
                            model_chunk_stats[chunk_type]['patient_count'] = max(
                                model_chunk_stats[chunk_type]['patient_count'],
                                stats.get('patient_count', 0)
                            )

                            # Proper weighted averaging of avg_chunks_per_patient
                            old_total = model_chunk_stats[chunk_type]['total_chunks'] - stats['total_chunks']
                            new_total = stats['total_chunks']

                            if old_total + new_total > 0:
                                old_avg = model_chunk_stats[chunk_type]['avg_chunks_per_patient']
                                new_avg = stats.get('avg_chunks_per_patient', 1.0)

                                if old_total == 0:
                                    # No existing data, just use new avg
                                    model_chunk_stats[chunk_type]['avg_chunks_per_patient'] = new_avg
                                else:
                                    # Weighted average calculation
                                    weighted_avg = ((old_avg * old_total) + (new_avg * new_total)) / (
                                                old_total + new_total)
                                    model_chunk_stats[chunk_type]['avg_chunks_per_patient'] = weighted_avg
                    except Exception as e:
                        logging.warning(f"Error processing chunk stats for {model}: {e}")

            # Store chunk stats for model-chunk summary
            model_chunk_data[model] = model_chunk_stats

            # Calculate aggregated metrics
            total_chunks = group['total_chunks'].sum()
            removed_chunks = group['removed_chunks'].sum()
            total_configs = group['config_count'].sum()
            phenotype_count = group['phenotype'].nunique()

            # Calculate averages
            avg_configs_per_phenotype = total_configs / phenotype_count if phenotype_count > 0 else 0
            avg_chunks_per_patient = total_chunks / len(all_patients) if all_patients else 0

            # Calculate percentages
            chunk_removal_pct = (removed_chunks / total_chunks * 100) if total_chunks > 0 else 0
            patient_removal_pct = (len(affected_patients) / len(all_patients) * 100) if all_patients else 0

            # Add model summary row
            model_rows.append({
                'model': model,
                'phenotype_count': phenotype_count,
                'total_configs': total_configs,
                'avg_configs_per_phenotype': avg_configs_per_phenotype,
                'total_chunks': total_chunks,
                'removed_chunks': removed_chunks,
                'chunk_removal_pct': chunk_removal_pct,
                'avg_chunks_per_patient': avg_chunks_per_patient,
                'avg_patients_number_removed_per_config': len(
                    affected_patients) / total_configs if total_configs > 0 else 0,
                'avg_patient_removal_rate': patient_removal_pct,
                'total_patients': len(all_patients),
                'affected_patients': len(affected_patients),
                'patient_removal_pct': patient_removal_pct
            })

        # Create and save model summary
        if model_rows:
            model_df = pd.DataFrame(model_rows)

            # Ensure consistent column order
            column_order = [
                'model', 'phenotype_count', 'total_configs', 'avg_configs_per_phenotype',
                'total_chunks', 'removed_chunks', 'chunk_removal_pct',
                'avg_chunks_per_patient', 'avg_patients_number_removed_per_config', 'avg_patient_removal_rate',
                'total_patients', 'affected_patients', 'patient_removal_pct'
            ]

            # Ensure columns exist
            for col in column_order:
                if col not in model_df.columns:
                    model_df[col] = None

            # Reorder columns
            model_df = model_df[column_order]

            # Save model summary
            save_csv(model_df, f"{summary_dir}/model_summary.csv")

            # 3. Generate model-chunk summary
            chunk_rows = []

            for model, chunk_data in model_chunk_data.items():
                model_row = model_df[model_df['model'] == model].iloc[0]

                for chunk_type, stats in chunk_data.items():
                    # Get the actual patient count for this chunk type
                    patient_count = stats['patient_count']

                    # Calculate metrics
                    pheno_count = len(stats['phenotypes'])
                    config_count = stats['configs']
                    total_chunks = stats['total_chunks']
                    removed_chunks = stats['removed_chunks']
                    avg_chunks_per_patient = stats['avg_chunks_per_patient']

                    # Estimated patients shouldn't be higher than actual patients
                    # For chunked docs, patients = total_patients
                    # For non-chunked, patients = chunks (since 1:1 relationship)
                    is_chunked = chunk_type not in ['none', 'None', None, '']

                    # Calculate derived metrics
                    avg_configs_per_phenotype = config_count / pheno_count if pheno_count > 0 else 0
                    chunk_removal_pct = (removed_chunks / total_chunks * 100) if total_chunks > 0 else 0
                    avg_chunks_per_config = total_chunks / config_count if config_count > 0 else 0

                    # Add chunk summary row
                    chunk_rows.append({
                        'model': model,
                        'chunk_type': chunk_type,
                        'phenotype_count': pheno_count,
                        'total_configs': config_count,
                        'avg_configs_per_phenotype': avg_configs_per_phenotype,
                        'total_chunks': total_chunks,
                        'removed_chunks': removed_chunks,
                        'chunk_removal_pct': chunk_removal_pct,
                        'avg_chunks_per_config': avg_chunks_per_config,
                        'avg_chunks_per_patient': avg_chunks_per_patient,
                        'total_patients': patient_count,  # Use direct patient count
                        'is_chunked': is_chunked
                    })

            # Create and save model-chunk summary
            if chunk_rows:
                chunk_df = pd.DataFrame(chunk_rows)

                # Ensure consistent column order
                chunk_columns = [
                    'model', 'chunk_type', 'phenotype_count', 'total_configs', 'avg_configs_per_phenotype',
                    'total_chunks', 'removed_chunks', 'chunk_removal_pct',
                    'avg_chunks_per_config', 'avg_chunks_per_patient', 'total_patients', 'is_chunked'
                ]

                # Ensure columns exist
                for col in chunk_columns:
                    if col not in chunk_df.columns:
                        chunk_df[col] = None

                # Reorder columns
                chunk_df = chunk_df[chunk_columns]

                # Save model-chunk summary
                save_csv(chunk_df, f"{summary_dir}/model_chunk_summary.csv")

    except Exception as e:
        logging.warning(f"Error generating model summaries: {e}")
        traceback.print_exc()


def export_results(results, output_prefix):
    """
    Export analysis results to CSV files.

    Args:
        results: Dictionary containing analysis results
        output_prefix: Path prefix for output files
    """
    analysis_types = [
        'model_analysis',
        'chunking_analysis',
        'prompt_analysis',
        'aggregation_analysis',
        'phenotype_analysis',
        'hadm_analysis'
    ]

    for analysis_type in analysis_types:
        if analysis_type in results:
            output_path = f"{output_prefix}{analysis_type}.csv"
            save_csv(results[analysis_type], output_path)


def select_challenging_cases(file_path, phenotypes, cases_per_phenotype=20, output_dir='challenging_cases',
                             file_path2=None):
    """
    Selects challenging cases (misclassified hadm_ids) for each phenotype to test a stronger reasoning model.
    Only includes cases that were misclassified by BOTH gpt_4o AND llama_405b models.

    Args:
        file_path: Path to the primary CSV file containing misclassification data
        phenotypes: List of phenotypes to analyze
        cases_per_phenotype: Number of cases to select per phenotype (default 20)
        output_dir: Directory to save CSV files
        file_path2: Optional path to a secondary CSV file to concatenate with the first one

    Returns:
        dict: Dictionary with phenotypes as keys and lists of selected hadm_ids as values
        pandas.DataFrame: Combined dataframe of all selected challenging cases
    """
    import os
    import numpy as np
    import pandas as pd

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data - handling carefully to avoid index issues
    data_frames = []

    # Load primary dataframe
    try:
        df1 = pd.read_csv(file_path)
        df1 = df1.rename(columns={
            'Phenotype': 'phenotype',
            'ModelName': 'model',
            'ChunkingMethod': 'chunking_method',
            'PromptStrategy': 'prompt_strategy',
            'AggregationMethod': 'aggregation_method'
        })
        data_frames.append(df1)
    except Exception as e:
        print(f"Error loading primary file: {e}")
        return {}, pd.DataFrame()

    # Load secondary dataframe if provided
    if file_path2 is not None:
        try:
            df2 = pd.read_csv(file_path2)
            df2 = df2.rename(columns={
                'Phenotype': 'phenotype',
                'ModelName': 'model',
                'ChunkingMethod': 'chunking_method',
                'PromptStrategy': 'prompt_strategy',
                'AggregationMethod': 'aggregation_method'
            })
            data_frames.append(df2)
        except Exception as e:
            print(f"Warning: Error loading secondary file: {e}")

    # Combine all data frames and reset index
    if len(data_frames) > 1:
        df = pd.concat(data_frames, ignore_index=True)
    else:
        df = data_frames[0]

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Initialize results
    selected_cases = {}
    all_selected_cases = []

    # Process each phenotype carefully to avoid reindexing issues
    for phenotype in phenotypes:
        try:
            # Instead of filtering with boolean masks, use iterating through rows to avoid reindexing
            # Find all records for this phenotype
            phenotype_rows = []
            for idx, row in df.iterrows():
                if row['phenotype'] == phenotype:
                    phenotype_rows.append(row.to_dict())

            if not phenotype_rows:
                print(f"No misclassifications found for phenotype: {phenotype}")
                continue

            # Create a fresh dataframe for this phenotype to avoid any indexing issues
            phenotype_df = pd.DataFrame(phenotype_rows)

            # Find HADM_IDs that were misclassified by llama_405b and by gpt_4o for this phenotype
            llama_405b_hadm_ids = set()
            gpt_4o_hadm_ids = set()

            for _, row in phenotype_df.iterrows():
                model = str(row['model']).lower()
                if 'llama_405b' in model:
                    llama_405b_hadm_ids.add(row['hadm_id'])
                if 'gpt_4o' in model or 'gpt-4o' in model:
                    gpt_4o_hadm_ids.add(row['hadm_id'])

            # Find cases misclassified by both models
            both_models_hadm_ids = llama_405b_hadm_ids.intersection(gpt_4o_hadm_ids)

            if not both_models_hadm_ids:
                print(f"No cases found for phenotype {phenotype} that were misclassified by both llama_405b and gpt_4o")
                continue

            print(f"Found {len(both_models_hadm_ids)} cases for phenotype {phenotype} misclassified by both models")

            # Filter to only include cases misclassified by both models
            filtered_rows = [row for row in phenotype_rows if row['hadm_id'] in both_models_hadm_ids]

            if not filtered_rows:
                print(f"No records remain after filtering for phenotype: {phenotype}")
                continue

            phenotype_df = pd.DataFrame(filtered_rows)

            # Count misclassifications per hadm_id
            hadm_counts = {}
            for row in filtered_rows:
                hadm_id = row['hadm_id']
                hadm_counts[hadm_id] = hadm_counts.get(hadm_id, 0) + 1

            # Get unique configurations count for normalization
            configs = set()
            for row in filtered_rows:
                config = (row['model'], row['chunking_method'], row['prompt_strategy'], row['aggregation_method'])
                configs.add(config)
            config_count = len(configs)

            # Calculate misclassification rates
            hadm_rates = {hadm_id: count / config_count for hadm_id, count in hadm_counts.items()}

            # Sort HADMs by misclassification count
            sorted_hadms = sorted(hadm_counts.items(), key=lambda x: x[1], reverse=True)

            # Select cases using our strategy
            if len(sorted_hadms) > cases_per_phenotype:
                # Take top half
                top_half_size = max(cases_per_phenotype, len(sorted_hadms) // 2)
                candidate_pool = sorted_hadms[:top_half_size]

                if len(candidate_pool) > cases_per_phenotype:
                    # Randomly sample
                    import random
                    random.seed(42)  # For reproducibility
                    selected_indices = random.sample(range(len(candidate_pool)), cases_per_phenotype)
                    selected_hadm_ids = [candidate_pool[i][0] for i in selected_indices]
                else:
                    selected_hadm_ids = [hadm_id for hadm_id, _ in candidate_pool]
            else:
                selected_hadm_ids = [hadm_id for hadm_id, _ in sorted_hadms]

            selected_cases[phenotype] = selected_hadm_ids

            # Get detailed information for selected cases
            selected_rows = []
            for idx, row in df.iterrows():
                if row['phenotype'] == phenotype and row['hadm_id'] in selected_hadm_ids:
                    selected_rows.append(row.to_dict())

            selected_details = pd.DataFrame(selected_rows)

            # Create summary for these cases
            hadm_summaries = []
            for hadm_id in selected_hadm_ids:
                hadm_rows = [r for r in selected_rows if r['hadm_id'] == hadm_id]

                models = set(r['model'] for r in hadm_rows)
                chunking_methods = set(r['chunking_method'] for r in hadm_rows)
                prompt_strategies = set(r['prompt_strategy'] for r in hadm_rows)
                agg_methods = set(r['aggregation_method'] for r in hadm_rows)

                # Check if this case was misclassified by both models
                llama_405b_misclassified = any('llama_405b' in str(r['model']).lower() for r in hadm_rows)
                gpt_4o_misclassified = any(('gpt_4o' in str(r['model']).lower() or 'gpt-4o' in str(r['model']).lower())
                                           for r in hadm_rows)

                summary = {
                    'hadm_id': hadm_id,
                    'model': ', '.join(sorted(str(x) for x in models)),
                    'chunking_method': ', '.join(sorted(str(x) for x in chunking_methods)),
                    'prompt_strategy': ', '.join(sorted(str(x) for x in prompt_strategies)),
                    'aggregation_method': ', '.join(sorted(str(x) for x in agg_methods)),
                    'misclass_count': hadm_counts[hadm_id],
                    'misclass_rate': hadm_rates[hadm_id],
                    'phenotype': phenotype,
                    'llama_405b_misclassified': llama_405b_misclassified,
                    'gpt_4o_misclassified': gpt_4o_misclassified,
                    'both_models_misclassified': llama_405b_misclassified and gpt_4o_misclassified
                }
                hadm_summaries.append(summary)

            # Create and save summary
            summary_df = pd.DataFrame(hadm_summaries)
            summary_df.to_csv(f"{output_dir}/{phenotype}_challenging_cases.csv", index=False)

            # Add to combined results
            all_selected_cases.append(summary_df)

            print(f"Successfully processed phenotype: {phenotype} - found {len(selected_hadm_ids)} challenging cases")

        except Exception as e:
            import traceback
            print(f"Error processing phenotype {phenotype}: {e}")
            print(traceback.format_exc())
            continue

    # Combine all phenotypes into one dataframe
    if all_selected_cases:
        try:
            combined_df = pd.concat(all_selected_cases, ignore_index=True)
            combined_df.to_csv(f"{output_dir}/all_challenging_cases.csv", index=False)

            # Create a simple CSV with just unique HADM_IDs for easier testing
            unique_hadm_ids = pd.DataFrame({'hadm_id': combined_df['hadm_id'].unique()})
            unique_hadm_ids.to_csv(f"{output_dir}/unique_challenging_hadm_ids.csv", index=False)

            print(f"Found a total of {len(unique_hadm_ids)} unique challenging cases across all phenotypes")
        except Exception as e:
            print(f"Error creating combined output: {e}")
            combined_df = pd.DataFrame()
    else:
        combined_df = pd.DataFrame()
        print("No challenging cases were found across all phenotypes.")

    return selected_cases, combined_df


def select_easy_cases(proper_classify_file1, proper_classify_file2, phenotypes, cases_per_phenotype=20,
                      output_dir='easy_cases'):
    """
    Selects "easy" cases (correctly classified hadm_ids) for each phenotype.
    Only includes cases that were correctly classified by BOTH gpt_4o AND llama_405b models.

    Args:
        proper_classify_file1: Path to first CSV file containing properly classified cases
        proper_classify_file2: Path to second CSV file containing properly classified cases
        phenotypes: List of phenotypes to analyze
        cases_per_phenotype: Number of cases to select per phenotype (default 20)
        output_dir: Directory to save CSV files

    Returns:
        dict: Dictionary with phenotypes as keys and lists of selected hadm_ids as values
        pandas.DataFrame: Combined dataframe of all selected easy cases
    """
    import os
    import numpy as np
    import pandas as pd
    import random

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load proper classification data from both files
    try:
        proper_df1 = pd.read_csv(proper_classify_file1)
        proper_df1 = proper_df1.rename(columns={
            'Phenotype': 'phenotype',
            'ModelName': 'model',
            'ChunkingMethod': 'chunking_method',
            'PromptStrategy': 'prompt_strategy',
            'AggregationMethod': 'aggregation_method'
        })
    except Exception as e:
        print(f"Error loading proper classification file 1: {e}")
        return {}, pd.DataFrame()

    try:
        proper_df2 = pd.read_csv(proper_classify_file2)
        proper_df2 = proper_df2.rename(columns={
            'Phenotype': 'phenotype',
            'ModelName': 'model',
            'ChunkingMethod': 'chunking_method',
            'PromptStrategy': 'prompt_strategy',
            'AggregationMethod': 'aggregation_method'
        })
    except Exception as e:
        print(f"Error loading proper classification file 2: {e}")
        return {}, pd.DataFrame()

    # Combine both proper classification dataframes
    proper_df = pd.concat([proper_df1, proper_df2], ignore_index=True)
    proper_df = proper_df.drop_duplicates().reset_index(drop=True)

    # Initialize results
    selected_cases = {}
    all_selected_cases = []

    # Process each phenotype
    for phenotype in phenotypes:
        try:
            # Get proper classifications for this phenotype
            phenotype_proper = proper_df[proper_df['phenotype'] == phenotype].copy()

            if phenotype_proper.empty:
                print(f"No proper classifications found for phenotype: {phenotype}")
                continue

            # Find HADM_IDs that were properly classified by llama_405b
            llama_405b_proper_mask = phenotype_proper['model'].str.contains('llama_405b', case=False, na=False)
            llama_405b_proper_hadm_ids = set(phenotype_proper.loc[llama_405b_proper_mask, 'hadm_id'].unique())

            # Find HADM_IDs that were properly classified by gpt_4o
            gpt_4o_proper_mask = (
                    phenotype_proper['model'].str.contains('gpt_4o', case=False, na=False) |
                    phenotype_proper['model'].str.contains('gpt-4o', case=False, na=False)
            )
            gpt_4o_proper_hadm_ids = set(phenotype_proper.loc[gpt_4o_proper_mask, 'hadm_id'].unique())

            # Find HADM_IDs that were properly classified by BOTH models
            both_proper_hadm_ids = llama_405b_proper_hadm_ids.intersection(gpt_4o_proper_hadm_ids)

            print(f"Found {len(both_proper_hadm_ids)} easy cases for phenotype {phenotype}")

            if len(both_proper_hadm_ids) == 0:
                print(f"No easy cases found for phenotype {phenotype}")
                continue

            # Calculate percentage of right classifications for each HADM_ID
            hadm_stats = {}

            # First, count the total configurations in the dataset for normalization
            all_configs = set()
            for _, row in phenotype_proper.iterrows():
                config = (str(row['model']), str(row['chunking_method']),
                          str(row['prompt_strategy']), str(row['aggregation_method']))
                all_configs.add(config)
            total_configs = len(all_configs)

            # Now count correct classifications per HADM_ID
            for hadm_id in both_proper_hadm_ids:
                hadm_rows = phenotype_proper[phenotype_proper['hadm_id'] == hadm_id]

                # Count unique configurations for this HADM_ID
                hadm_configs = set()
                for _, row in hadm_rows.iterrows():
                    config = (str(row['model']), str(row['chunking_method']),
                              str(row['prompt_strategy']), str(row['aggregation_method']))
                    hadm_configs.add(config)

                correct_config_count = len(hadm_configs)
                percent_correct = (correct_config_count / total_configs) * 100

                hadm_stats[hadm_id] = {
                    'correct_config_count': correct_config_count,
                    'percent_correct': percent_correct,
                    'llama_405b_correct': hadm_id in llama_405b_proper_hadm_ids,
                    'gpt_4o_correct': hadm_id in gpt_4o_proper_hadm_ids
                }

            # Rank HADMs by percentage of correct classifications (most consistently correct first)
            ranked_hadm_ids = sorted(hadm_stats.items(), key=lambda x: x[1]['percent_correct'], reverse=True)

            # Select cases prioritizing those with highest correct percentages
            if len(ranked_hadm_ids) > cases_per_phenotype:
                selected_easy_hadm_ids = [hadm_id for hadm_id, _ in ranked_hadm_ids[:cases_per_phenotype]]
            else:
                selected_easy_hadm_ids = [hadm_id for hadm_id, _ in ranked_hadm_ids]

            selected_cases[phenotype] = selected_easy_hadm_ids

            # Create summary for these cases
            hadm_summaries = []
            for rank, (hadm_id, stats) in enumerate(ranked_hadm_ids):
                if hadm_id in selected_easy_hadm_ids:
                    # Get rows for this hadm_id
                    hadm_rows = phenotype_proper[phenotype_proper['hadm_id'] == hadm_id]

                    # Extract unique values for each field
                    models = set(str(m) for m in hadm_rows['model'].unique())
                    chunking_methods = set(str(c) for c in hadm_rows['chunking_method'].unique())
                    prompt_strategies = set(str(p) for p in hadm_rows['prompt_strategy'].unique())
                    agg_methods = set(str(a) for a in hadm_rows['aggregation_method'].unique())

                    # Create summary
                    summary = {
                        'hadm_id': hadm_id,
                        'model': ', '.join(sorted(models)),
                        'chunking_method': ', '.join(sorted(chunking_methods)),
                        'prompt_strategy': ', '.join(sorted(prompt_strategies)),
                        'aggregation_method': ', '.join(sorted(agg_methods)),
                        'phenotype': phenotype,
                        'correct_config_count': stats['correct_config_count'],
                        'total_configs': total_configs,
                        'percent_correct': round(stats['percent_correct'], 2),
                        'rank': rank + 1,  # 1-based ranking
                        'llama_405b_correct': stats['llama_405b_correct'],
                        'gpt_4o_correct': stats['gpt_4o_correct'],
                        'both_models_correct': stats['llama_405b_correct'] and stats['gpt_4o_correct']
                    }
                    hadm_summaries.append(summary)

            # Create and save summary
            summary_df = pd.DataFrame(hadm_summaries)
            summary_df = summary_df.sort_values('percent_correct', ascending=False)
            summary_df.to_csv(f"{output_dir}/{phenotype}_easy_cases.csv", index=False)

            # Add to combined results
            all_selected_cases.append(summary_df)

            print(f"Successfully processed phenotype: {phenotype} - selected {len(selected_easy_hadm_ids)} easy cases")
            print(
                f"  - First case: HADM_ID {selected_easy_hadm_ids[0]} correctly classified in {hadm_stats[selected_easy_hadm_ids[0]]['percent_correct']:.2f}% of configurations")

        except Exception as e:
            import traceback
            print(f"Error processing phenotype {phenotype}: {e}")
            print(traceback.format_exc())
            continue

    # Combine all phenotypes into one dataframe
    if all_selected_cases:
        try:
            combined_df = pd.concat(all_selected_cases, ignore_index=True)
            combined_df.to_csv(f"{output_dir}/all_easy_cases.csv", index=False)

            # Create a simple CSV with just unique HADM_IDs for easier testing
            unique_hadm_ids = pd.DataFrame({'hadm_id': combined_df['hadm_id'].unique()})
            unique_hadm_ids.to_csv(f"{output_dir}/unique_easy_hadm_ids.csv", index=False)

            print(f"Found a total of {len(unique_hadm_ids)} unique easy cases across all phenotypes")
        except Exception as e:
            print(f"Error creating combined output: {e}")
            combined_df = pd.DataFrame()
    else:
        combined_df = pd.DataFrame()
        print("No easy cases were found across all phenotypes.")

    return selected_cases, combined_df
