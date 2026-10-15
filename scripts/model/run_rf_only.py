import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.model.modeling import train_all_random_forest
from scripts.model.evaluate import evaluate_all
from scripts.constants import PHENOTYPES

OUTPUT_DIR = 'data/models/paper_comparison_ml'
OUTPUT_PATH = f'{OUTPUT_DIR}/classic_ml_random_forest_results.csv'
CHECKPOINT_DIR = f'{OUTPUT_DIR}/random_forest_checkpoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_stop_heartbeat = threading.Event()
def _heartbeat():
    start = time.time()
    while not _stop_heartbeat.wait(60):
        print(f"[heartbeat] still running, {time.time()-start:.0f}s elapsed", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()

if os.path.exists(OUTPUT_PATH):
    print(f"[SKIPPING] already completed: {OUTPUT_PATH}", flush=True)
    sys.exit(0)

all_metrics = []
for phenotype in PHENOTYPES:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{phenotype}.csv")
    if os.path.exists(ckpt_path):
        print(f"[SKIPPING {phenotype}] checkpoint already exists", flush=True)
        all_metrics.append(pd.read_csv(ckpt_path))
        continue

    t0 = time.time()
    print(f"=== Training random_forest for {phenotype} ===", flush=True)
    models = train_all_random_forest(feature_reduction=True, phenotypes=[phenotype], imputation_method='simple')
    metrics_df = evaluate_all(models, model_type='random_forest')
    if metrics_df is not None and not metrics_df.empty:
        metrics_df['model_type'] = 'random_forest'
        metrics_df.to_csv(ckpt_path, index=False)
        all_metrics.append(metrics_df)
        print(f"Saved checkpoint for {phenotype} to {ckpt_path}", flush=True)
    print(f"=== {phenotype} DONE in {time.time()-t0:.1f}s ===", flush=True)

combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
combined.to_csv(OUTPUT_PATH, index=False)
print(f"Saved combined random_forest results to {OUTPUT_PATH}", flush=True)
print("TASK COMPLETED", flush=True)
