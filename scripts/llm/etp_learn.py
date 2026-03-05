
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
import json
import os




PHENOTYPES_ETP_LEARNED_THRESHOLDS = {
  "diabetes_type_2": {
    "weights": {
      "diabetes_type_2": 0.5,
      "metformin": 0.2,
      "sulfonylureas": 0.1,
      "glp-1 agonists": 0.3,
      "dpp-4 inhibitors": 0.4,
      "sglt2 inhibitors": 0.2,
      "insulin": 0.05,
      "neuropathy": 0.05,
      "retinopathy": 0.05,
      "nephropathy": 0.05,
      "blood glucose levels": 0.1,
      "HbA1c": 0.15
    },
    "threshold": 0.75
  }


# Assuming PHENOTYPES_ETP_KEYS is predefined with all phenotypes and their respective features

def load_phenotype_data(phenotype):
    file_path = f'data/{phenotype}_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for {phenotype} at {file_path}")
    df = pd.read_csv(file_path)
    return df

def train_model_and_find_threshold(df, features):
    if not set(features).issubset(df.columns):
        missing_features = set(features) - set(df.columns)
        raise ValueError(f"Missing features in the DataFrame for {features}: {missing_features}")
    X = df[features]
    y = df['gold_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * recall * precision / (recall + precision)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    return dict(zip(features, model.coef_[0])), optimal_threshold

def train_all_phenotypes(phenotypes):
    results = {}
    for phenotype, features in phenotypes.items():
        try:
            df = load_phenotype_data(phenotype)
            weights, threshold = train_model_and_find_threshold(df, features)
            results[phenotype] = {
                'weights': weights,
                'threshold': threshold
            }
            print(f"Successfully trained model for {phenotype}")
        except Exception as e:
            print(f"Error training model for {phenotype}: {e}")
    return results

# Example usage
all_phenotype_info = train_all_phenotypes(PHENOTYPES_ETP_KEYS)

# Save to JSON
with open('phenotype_weights_thresholds.json', 'w') as f:
    json.dump(all_phenotype_info, f, indent=4)

print("Saved all phenotype weights and thresholds to 'phenotype_weights_thresholds.json'")