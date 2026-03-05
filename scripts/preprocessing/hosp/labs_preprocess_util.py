import os
from glob import glob
from uuid import uuid1
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
from dask import dataframe as dd
directory_path = "./data/temp"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

def hadm_imputer(
    charttime: pd.Timestamp,
    hadm_old: Union[str, float],
    hadm_ids_w_timestamps: List[
        Tuple[
            str,
            pd.Timestamp,
            pd.Timestamp,
        ]
    ],
) -> Tuple[Union[str, None], Union[str, None], Union[str, None]]:
    # Check if old HADM ID exists and is not NaN
    if hadm_old is not None and not pd.isna(hadm_old):
        hadm_old = str(hadm_old)  # Ensure HADM ID is treated as a string
        for h_id, adm_time, disch_time in hadm_ids_w_timestamps:
            if h_id == hadm_old:
                return (
                    h_id,
                    adm_time.strftime("%Y-%m-%d %H:%M:%S"),
                    disch_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

    # Prepare list with time differences
    hadm_ids_w_timestamps = [
        (
            hadm_id,
            admittime,
            dischtime,
            (charttime - admittime).total_seconds(),
            (charttime - dischtime).total_seconds(),
        )
        for hadm_id, admittime, dischtime in hadm_ids_w_timestamps
    ]

    # Filter to only include events between admittance and discharge
    hadm_ids_w_timestamps = [
        x for x in hadm_ids_w_timestamps if x[3] >= 0 and x[4] <= 0
    ]

    # Sort by the admit time closest to the lab event charttime
    hadm_ids_w_timestamps.sort(key=lambda x: abs(x[3]))

    # If there are no matches, return None values
    if not hadm_ids_w_timestamps:
        return None, None, None

    hadm_id, admittime, dischtime, _, _ = hadm_ids_w_timestamps[0]
    return (
        hadm_id,
        admittime.strftime("%Y-%m-%d %H:%M:%S"),
        dischtime.strftime("%Y-%m-%d %H:%M:%S"),
    )

def impute_missing_hadm_ids(
    lab_table: pd.DataFrame, subject_hadm_admittime_tracker: defaultdict
) -> pd.DataFrame:
    list_rows_lab = []
    all_lab_cols = lab_table.columns
    for row in lab_table.itertuples():
        existing_data = {
            col_name: row.__getattribute__(col_name) for col_name in all_lab_cols
        }
        new_hadm_id, new_admittime, new_dischtime = hadm_imputer(
            row.charttime,
            row.hadm_id,
            subject_hadm_admittime_tracker.get(
                row.subject_id, []
            ),  # using get as defaultdict will create key if does not exist
        )
        existing_data["hadm_id_new"] = new_hadm_id
        existing_data["admittime"] = new_admittime
        existing_data["dischtime"] = new_dischtime
        list_rows_lab.append(existing_data)
    tab_name = str(uuid1())
    pd.DataFrame(list_rows_lab).to_csv(f"{directory_path}/{tab_name}.csv")


def impute_hadm_ids(
    lab_table: Union[str, pd.DataFrame], admission_table: Union[str, pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(lab_table, str):
        lab_table = pd.read_csv(lab_table)
    if isinstance(admission_table, str):
        admission_table = pd.read_csv(admission_table)

    admission_table.compute() if isinstance(admission_table, dd.DataFrame) else admission_table
    lab_table["charttime"] = pd.to_datetime(lab_table.charttime)
    admission_table["admittime"] = pd.to_datetime(admission_table.admittime)
    admission_table["dischtime"] = pd.to_datetime(admission_table.dischtime)
    # get a dictionary like this->
    """ {
        "sub_id_1": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        "sub_id_2": [["hadm_1", "admittime1", "dischtime1"], ["hadm_2", "admittime2", "dischtime2"]],
        ...
    """
    subject_hadm_admittime_tracker = defaultdict(list)
    for row in admission_table.itertuples():
        subject_hadm_admittime_tracker[row.subject_id].append(
            [row.hadm_id, row.admittime, row.dischtime]
        )
    lab_size = lab_table.shape[0]
    chunks = 100
    tab_size = lab_size // chunks
    lab_table_chunks = []
    for i in range(chunks):
        st, en = i * tab_size, (i + 1) * tab_size
        lab_table_chunks.append(lab_table[st:en])
    if lab_size - chunks * tab_size > 0:
        lab_table_chunks.append(lab_table[chunks * tab_size :])
    # we dont need the original lab table as it is chunkified, hope is to save memory as lab table is huge
    del lab_table
    impute_missing_hadm_ids_w_lookup = partial(
        impute_missing_hadm_ids,
        subject_hadm_admittime_tracker=subject_hadm_admittime_tracker,
    )
    #print(impute_missing_hadm_ids_w_lookup)
    #print(len(lab_table_chunks))
    with Pool(8) as p:
        p.map(impute_missing_hadm_ids_w_lookup, lab_table_chunks)
    all_csvs = glob(f"{directory_path}/*.csv")
    lab_tab = pd.DataFrame()
    for csv in all_csvs:
        lab_tab = pd.concat([lab_tab, pd.read_csv(csv)])
        os.remove(csv)
    return lab_tab

