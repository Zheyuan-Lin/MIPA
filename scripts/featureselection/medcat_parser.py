import pandas as pd
import os
import multiprocessing as mp
from scipy.sparse import coo_matrix
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


def _annotate_texts_chunk(texts):
    """
    Worker function for process_and_annotate_dataframe_parallel. Runs in a separate process
    (see pool initializer below) -- annotates a list of raw strings and returns the same
    'CUI: count, CUI: count' string format as process_and_annotate_dataframe's annotate_text,
    in the same order as the input list.
    """
    global _WORKER_CAT
    results = []
    for text in texts:
        cui_counts = {}
        if isinstance(text, str) and len(text) > 10:
            entities = _WORKER_CAT.get_entities(text)
            for entity in entities['entities'].values():
                cui = entity['cui']
                cui_counts[cui] = cui_counts.get(cui, 0) + 1
        results.append(', '.join(f"{cui}: {count}" for cui, count in cui_counts.items()))
    return results


def _init_worker_cat():
    """
    multiprocessing.Pool initializer: loads one MedCAT model per worker process. Each
    loaded model is ~1.1GB RSS (measured) -- this is why process_and_annotate_dataframe_parallel
    keeps n_workers conservative by default rather than maxing out CPU core count.
    """
    global _WORKER_CAT
    _WORKER_CAT = generate_cat()


def process_and_annotate_dataframe_parallel(df, text_column='text', n_workers=3, chunksize=50):
    """
    Parallel version of process_and_annotate_dataframe. MedCAT's per-document inference is
    independent across documents, so this is an easy, safe speedup for large note counts --
    at train scale (~1.35M notes across DSNOTES+RADNOTES), the serial version was measured
    at ~240ms/note, i.e. ~3.7 DAYS of continuous single-core work. CAT.load_model_pack's
    loaded object isn't picklable/shareable across macOS's default 'spawn' multiprocessing
    start method, so each worker loads its OWN full copy of the model via the pool
    initializer below -- this is why n_workers is kept conservative (default 3, ~3.3GB of
    model copies total) rather than matching the machine's full core count: more workers
    would speed up wall-clock time further but risks running this 8GB machine out of memory
    on top of everything else already using it.

    Returns a copy of df with an 'nlp_codes' column, in the exact same row order as the
    input -- verified identical output to process_and_annotate_dataframe on the same input
    before using this for the real run.
    """
    df = df.reset_index(drop=True)
    texts = df[text_column].tolist()

    # Force 'spawn' instead of the platform default. On Linux the default is 'fork',
    # which deadlocks here: by the time this runs, torch/spacy (imported transitively via
    # medcat.cat.CAT) have already initialized internal thread pools (BLAS/OpenMP) in this
    # process, and fork()'ing with live threads copies their lock state into each child
    # without the owning threads -- the child then hangs forever the first time it touches
    # one of those locks (confirmed: workers sat at 0% CPU indefinitely on an 8-core Linux
    # VM, whereas macOS -- where 'spawn' is already the default -- never hit this).
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers, initializer=_init_worker_cat) as pool:
        # Split into many small chunks (chunksize param, not len(texts)//n_workers) so
        # pool.map load-balances across workers instead of giving one worker a
        # disproportionately slow batch (e.g. a run of unusually long discharge summaries)
        # while the others sit idle.
        chunk_results = pool.map(_annotate_texts_chunk, _chunk_list(texts, chunksize))

    all_results = [item for chunk in chunk_results for item in chunk]
    df['nlp_codes'] = all_results
    return df


def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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

    # Build the pivot directly as a pandas Sparse DataFrame via a scipy COO matrix instead
    # of df.pivot_table(..., fill_value=0) -- same rationale as pivot_cohort in
    # scripts/featureselection/pivot.py: pivot_table builds a dense (admissions x CUIs)
    # array before anything else happens, and the UMLS CUI vocabulary observed across a
    # large note corpus can run into the thousands-to-tens-of-thousands, which at full
    # training-cohort scale (hundreds of thousands of admissions) is the same "tens of
    # billions of cells" problem DIAG hit. coo_matrix sums duplicate (row, col) entries
    # automatically on conversion, which is exactly aggfunc='sum' semantics -- simpler than
    # pivot_cohort's case (which needed aggfunc='max' and a pre-dedup workaround).
    row_cat = pd.Categorical(df_expanded[id_column])
    col_cat = pd.Categorical(df_expanded['CUI'])
    coo = coo_matrix(
        (df_expanded['Count'].values, (row_cat.codes, col_cat.codes)),
        shape=(len(row_cat.categories), len(col_cat.categories)),
    ).tocsr()
    pivoted_df = pd.DataFrame.sparse.from_spmatrix(coo, index=row_cat.categories, columns=col_cat.categories)
    pivoted_df.index.name = id_column

    # Reset the index to turn 'hadm_id' back into a column
    pivoted_df = pivoted_df.reset_index()

    return pivoted_df



#cat = generate_cat()
#print('loaded')
# process_and_annotate_dataframe(df_cohort,cat, "text")
#apply_feature_selection_and_export_cuis(df_cohort, cat=cat, output_file='cui_description_dict.csv')
#pivot_and_aggregate(df_cohort, id_column='hadm_id', cui_column='nlp_codes')
