import pandas as pd
import os
from scripts.constants import *
from scripts.utils import *
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config


# from medcat.utils.vocab import Vocab


# Loading MedCat annotator via individual files
# CDB model being the UMLS model
# cdb = CDB.load('../'+CDB_PATH)
# vocab = Vocab.load('../'+VOCAB_PATH)
# config = Config()
# config.general['spacy_model'] = 'en_core_web_md'
# cat = CAT(cdb=cdb, config=Config(), vocab=Vocab())
CAT_PATH = 'medcat/medmen'
cat = CAT.load_model_pack

def generate_cat():
    # Loading MedCat annotator via pre-trained pack
    cat = CAT.load_model_pack(CAT_PATH)
    # Semantic types : https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
    cui_filters = set()
    for type_ids in TYPE_IDS_FILTER:
        cui_filters.update(cat.cdb.addl_info['type_id2cuis'][type_ids])
    cat.cdb.config.linking['filters']['cuis'] = cui_filters
    return cat

def process_and_annotate_dataframe(df, cat, text_column='text'):
    def annotate_text(row):
        text = row[text_column]
        cui_counts = {}
        if len(text) > 10:
            # Annotate the text
            entities = cat.get_entities(text)

            # Correctly parse the entities JSON structure
            entities_list = list(entities['entities'].values())

            # Extract CUIs from the entities and count them
            for entity in entities_list:
                cui = entity['cui']
                if cui not in cui_counts:
                    cui_counts[cui] = 0
                cui_counts[cui] += 1

        # Convert the dictionary to a string representation
        cui_counts_str = ', '.join(f"{cui}: {count}" for cui, count in cui_counts.items())
        return cui_counts_str

    # Add a new column 'nlp_codes' to the DataFrame
    df['nlp_codes'] = df.apply(annotate_text, axis=1)
    return df


def apply_feature_selection_and_export_cuis(df, cui_column='nlp_codes', cat=None, lower_quantile=0.10,
                                            output_file='cui_description_dict.csv'):
    """
    Apply feature selection based on CUI frequency and export a dictionary of CUI descriptions.

    Parameters:
    - df: DataFrame to apply feature selection to.
    - cui_column: Column containing CUI counts.
    - cat: Annotator object with access to CUI descriptions.
    - lower_quantile: Quantile used to determine the frequency threshold for keeping CUIs.
    - output_file: File path for exporting the CUI description dictionary.

    Returns:
    - DataFrame with low-frequency CUIs removed.
    """
    # Expand the CUI counts into a list of tuples (CUI, count)
    all_cuis = []
    for item in df[cui_column].dropna():
        cuis = item.split(', ')
        for cui in cuis:
            if ': ' in cui:
                cui_code, count = cui.split(': ')
                all_cuis.append((cui_code, int(count)))
            else:
                pass

    # Convert list to DataFrame
    cui_df = pd.DataFrame(all_cuis, columns=['CUI', 'Count'])

    # Aggregate counts by CUI
    cui_counts = cui_df.groupby('CUI').sum()

    # Calculate the threshold for feature selection
    threshold = cui_counts['Count'].quantile(lower_quantile)

    # Determine valid CUIs based on the threshold
    valid_cuis = cui_counts[cui_counts['Count'] > threshold].index

    # Filter the original DataFrame to remove rows with low-frequency CUIs
    def filter_cuis(cui_string):
        filtered_cuis = []
        for cui in cui_string.split(', '):
            if ': ' in cui:
                cui_code, _ = cui.split(': ')
                if cui_code in valid_cuis:
                    filtered_cuis.append(cui)
            else:
                pass
        return ', '.join(filtered_cuis)

    df[cui_column] = df[cui_column].apply(filter_cuis)

    # Create a dictionary to store CUI code and its description
    cui_description_dict = {cui: cat.cdb.get_name(cui) for cui in valid_cuis}

    # Convert the dictionary to a DataFrame
    cui_description_df = pd.DataFrame(list(cui_description_dict.items()), columns=['CUI', 'Description'])

    # Export the DataFrame to a CSV file
    save_csv(cui_description_df, output_file)
    print(f"CUI description dictionary exported to {output_file}")

    return df



def pivot_and_aggregate(df, id_column='hadm_id', cui_column='nlp_codes'):
    # Split the CUI counts into separate rows
    df = df[df[cui_column].notna() & (df[cui_column] != '')]

    s = df[cui_column].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
    s.name = 'cui_count'

    # Join the expanded CUI counts back to the original DataFrame to associate hadm_id
    df_expanded = df[[id_column]].join(s)

    # Split the 'cui_count' into 'CUI' and 'Count'
    df_expanded[['CUI', 'Count']] = df_expanded['cui_count'].str.split(': ', expand=True)

    # Convert 'Count' to numeric, handling None and NaN values
    df_expanded['Count'] = pd.to_numeric(df_expanded['Count'], errors='coerce')
    df_expanded['Count'].fillna(0, inplace=True)  # Fill NaN with 0

    # Pivot the DataFrame to get hadm_id as index and CUIs as columns, summing the counts
    pivoted_df = df_expanded.pivot_table(index=id_column, columns='CUI', values='Count', aggfunc='sum', fill_value=0)

    # Reset the index to turn 'hadm_id' back into a column
    pivoted_df = pivoted_df.reset_index()

    return pivoted_df



#cat = generate_cat()
#print('loaded')
# process_and_annotate_dataframe(df_cohort,cat, "text")
#apply_feature_selection_and_export_cuis(df_cohort, cat=cat, output_file='cui_description_dict.csv')
#pivot_and_aggregate(df_cohort, id_column='hadm_id', cui_column='nlp_codes')
