from scripts.model.modeling import train_all_models, train_all_lr, train_all_naive_bayes, train_all_random_forest, train_all_gradient_boosting
from scripts.model.reporting import generate_performance_report
from scripts.constants import MODEL_PATH
from scripts.model.evaluate import evaluate_all, evaluate_classification_model


PHENOTYPES = ["c_diff_complication", "c_diff_past", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
         "metastatic_cancer", "obesity", "rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"]


train_all_models(PHENOTYPES, 'median', 'simple', 'simple')

print("TASK COMPLETED")


