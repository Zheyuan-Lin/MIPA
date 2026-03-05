# Generate data summary
def summary(dict):
'''
Takes a dictionary of dataframes and returns a summary of the dataframes in CSV format.
This is expected to be executed after load_mimiciv function
'''
    for table_name, table in dataframes.items():
        table

# DICTIONARIES

# Get Chartevent items with labels & category
df = df_d_items
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    # print(category)
    category_list = df[df['category'] == category]
    for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append(
            {'eventtype': 'chart', 'category': category, 'label': item}, ignore_index=True)

# Get Lab items with labels & category
df = df_d_labitems
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    # print(category)
    category_list = df[df['category'] == category]
    for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append(
            {'eventtype': 'lab', 'category': category, 'label': item}, ignore_index=True)

# Get HCPCS items with labels & category
df = df_d_hcpcs
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
    # print(category)
    category_list = df[df['category'] == category]
    for item_idx, item in enumerate(sorted(category_list.long_description.astype(str).unique())):
        df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append(
            {'eventtype': 'hcpcs', 'category': category, 'label': item}, ignore_index=True)


