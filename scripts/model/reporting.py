import os
import pandas as pd
from scripts.utils import save_csv

def aggregate_model_performance(model_root='models'):
    """Aggregate all model_performance.csv files into a single DataFrame."""
    records = []
    for phenotype in os.listdir(model_root):
        pheno_path = os.path.join(model_root, phenotype)
        if not os.path.isdir(pheno_path) or phenotype == 'reports':
            continue
        for model_type in os.listdir(pheno_path):
            model_dir = os.path.join(pheno_path, model_type)
            perf_path = os.path.join(model_dir, 'model_performance.csv')
            if os.path.exists(perf_path):
                try:
                    df = pd.read_csv(perf_path)
                    df['phenotype'] = phenotype
                    df['model_type'] = model_type
                    records.append(df)
                except Exception as e:
                    print(f"Error reading {perf_path}: {e}")
    if not records:
        print("No model_performance.csv files found.")
        return pd.DataFrame()
    all_perf = pd.concat(records, ignore_index=True)
    return all_perf

def generate_performance_report(model_root='models'):
    """Generate and save summary reports of model performance."""
    report_dir = os.path.join(model_root, 'reports')
    os.makedirs(report_dir, exist_ok=True)

    all_perf = aggregate_model_performance(model_root)
    if all_perf.empty:
        print("No performance data to report.")
        return

    # Save the full table
    all_perf_path = os.path.join(report_dir, 'all_model_performance.csv')
    save_csv(all_perf, all_perf_path)
    print(f"Saved all model performance to {all_perf_path}")

    # Best model per phenotype
    best_by_pheno = all_perf.sort_values('F1 Score', ascending=False).groupby('phenotype').first().reset_index()
    best_by_pheno_path = os.path.join(report_dir, 'best_model_per_phenotype.csv')
    save_csv(best_by_pheno, best_by_pheno_path)
    print(f"Saved best model per phenotype to {best_by_pheno_path}")

    # Best phenotype per model type
    best_by_model = all_perf.sort_values('F1 Score', ascending=False).groupby('model_type').first().reset_index()
    best_by_model_path = os.path.join(report_dir, 'best_phenotype_per_model_type.csv')
    save_csv(best_by_model, best_by_model_path)
    print(f"Saved best phenotype per model type to {best_by_model_path}")

    # Mean/median/std by model type
    stats_by_model = all_perf.groupby('model_type')[['F1 Score', 'Accuracy', 'Precision', 'Recall']].agg(['mean', 'median', 'std']).reset_index()
    stats_by_model_path = os.path.join(report_dir, 'model_type_statistics.csv')
    save_csv(stats_by_model, stats_by_model_path)
    print(f"Saved model type statistics to {stats_by_model_path}")

    # Mean/median/std by phenotype
    stats_by_pheno = all_perf.groupby('phenotype')[['F1 Score', 'Accuracy', 'Precision', 'Recall']].agg(['mean', 'median', 'std']).reset_index()
    stats_by_pheno_path = os.path.join(report_dir, 'phenotype_statistics.csv')
    save_csv(stats_by_pheno, stats_by_pheno_path)
    print(f"Saved phenotype statistics to {stats_by_pheno_path}")

    # Optionally: add more advanced insights or visualizations here

# Usage: after all models are trained and evaluated, call:
# generate_performance_report('models') 