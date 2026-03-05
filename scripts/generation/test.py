# Loading file
import pandas as pd
test = pd.read_csv('test2.csv')
golden_labels = pd.read_csv('golden_labels.csv')

# remove duplicates based on hadm_id column
print("Unique hadm_id in test:", test['hadm_id'].nunique())
print("Unique hadm_id in golden_labels:", golden_labels['hadm_id'].nunique())

# Which hadm_ids are in test but not in golden_labels
missing_in_gold = set(test['hadm_id']) - set(golden_labels['hadm_id'])
print("HADM_IDs in test but missing in golden_labels:", len(missing_in_gold))

# Which hadm_ids are in golden_labels but not in test
missing_in_test = set(golden_labels['hadm_id']) - set(test['hadm_id'])
print("HADM_IDs in golden_labels but missing in test:", len(missing_in_test))

# Show some examples of missing matches
print("Example missing in golden_labels:", list(missing_in_gold)[:5])
print("Example missing in test:", list(missing_in_test)[:5])
