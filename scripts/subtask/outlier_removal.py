#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


def compute_outlier_imputation(arr, cut_off, left_thresh, impute):
    """
    Compute outlier imputation for a given array based on specified thresholds.

    Parameters:
    - arr: The input array to be processed.
    - cut_off: The upper percentile threshold for identifying outliers.
    - left_thresh: The lower percentile threshold for identifying outliers.
    - impute: A boolean indicating whether to impute outliers with the percentile thresholds (True) or replace them with NaN (False).

    Returns:
    - The processed array with outliers either imputed or replaced with NaN.
    """
    perc_up = np.percentile(arr, left_thresh)
    perc_down = np.percentile(arr, cut_off)

    if impute:
        arr[arr < perc_up] = perc_up
        arr[arr > perc_down] = perc_down
    else:
        arr[arr < perc_up] = np.nan
        arr[arr > perc_down] = np.nan

    return arr


def outlier_imputation(data, id_attribute, value_attribute, cut_off, left_thresh, impute):
    """
    Perform outlier imputation on a DataFrame based on specified thresholds for each group defined by an ID attribute.

    Parameters:
    - data: The input DataFrame to be processed.
    - id_attribute: The name of the column in the DataFrame that defines the grouping for outlier imputation.
    - value_attribute: The name of the column in the DataFrame that contains the values to be imputed.
    - cut_off: The upper percentile threshold for identifying outliers within each group.
    - left_thresh: The lower percentile threshold for identifying outliers within each group.
    - impute: A boolean indicating whether to impute outliers with the percentile thresholds (True) or replace them
    with NaN (False), if so the outliers will be removed from the dataset.

    Returns:
    - The modified DataFrame with outliers imputed or replaced with NaN based on the specified thresholds.
    """
    grouped = data.groupby([id_attribute])[value_attribute]

    for id_number, values in grouped:
        index = values.index
        values = compute_outlier_imputation(values, cut_off, left_thresh, impute)
        data.loc[index, value_attribute] = values

    data = data.dropna(subset=[value_attribute])

    return data
