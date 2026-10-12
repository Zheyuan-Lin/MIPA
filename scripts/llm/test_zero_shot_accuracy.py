"""
Zero-shot baseline: one model, prompt_type='zero_shot', chunking='none' (whole document,
no multi-chunk aggregation) -- no model/strategy/chunking comparison, just the baseline
zero-shot prediction result for every phenotype against golden_labels.csv.

What's being tested: for each (note, phenotype) pair, the model is asked to return
{"phenotype": ..., "status": "1 or 0"}. We compare that status against the human-adjudicated
label in golden_labels.csv and report accuracy/precision/recall/F1 per phenotype.

LLMParser.parse() sends requests SERIALLY (one note at a time, no concurrency). At full
scale (16 phenotypes x up to 1,388 notes = up to ~22,208 calls) this can take many hours.
Progress is checkpointed per batch (see --batch_size) so a crash/interrupt/VPN drop only
loses the in-flight batch, not the whole phenotype -- rerunning the same command resumes
from the last completed batch and skips phenotypes whose summary row is already saved.

Usage:
    # Full baseline run (default): all 16 phenotypes, every golden-labeled note, BMI llama3_70B
    python -m scripts.llm.test_zero_shot_accuracy

    # Narrow it down while validating the setup
    python -m scripts.llm.test_zero_shot_accuracy --phenotypes dementia --max_notes 20

    # OpenRouter instead of the BMI cluster
    python -m scripts.llm.test_zero_shot_accuracy --model llama_8b
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scripts.constants import GOLDEN_NOTES_PATH, GOLDEN_LABELS_PATH, OPENROUTER_URL, OPENROUTER_API_KEY, PHENOTYPES
from scripts.llm.llm_module import PromptEngineering, LLMParser, BMI_ENDPOINTS


def load_golden(phenotype, max_notes=None):
    """
    Merge golden_notes.csv (text) with golden_labels.csv (the 0/1 label column for
    `phenotype`) on note_id, drop rows with a missing/NaN label for this phenotype (not
    every note was adjudicated for every phenotype). max_notes=None (default) uses every
    available labeled note -- the right choice for a baseline number meant to be reported,
    since a small random subsample makes precision/recall swing wildly on these low-prevalence
    phenotypes (see the n=20 runs that produced 0/1/1 flips earlier). Pass --max_notes only to
    narrow scope while iterating on the setup itself.
    """
    notes = pd.read_csv(GOLDEN_NOTES_PATH)
    labels = pd.read_csv(GOLDEN_LABELS_PATH)
    merged = notes.merge(labels[['note_id', phenotype]], on='note_id', how='inner')
    merged = merged.dropna(subset=[phenotype])
    merged[phenotype] = merged[phenotype].astype(int)
    if max_notes is not None and len(merged) > max_notes:
        merged = merged.sample(n=max_notes, random_state=42)
    return merged.reset_index(drop=True)


def run_phenotype(parser, phenotype, df, ckpt_dir, batch_size):
    """
    Batches df through LLMParser.parse(), writing one checkpoint file per batch to ckpt_dir
    immediately after it completes. Batches already on disk from a prior (interrupted) run
    are skipped and reloaded instead of re-sent -- this is what makes the whole script
    resumable. Returns the concatenated result_df for this phenotype (empty if every row
    failed to parse) and the total count of rows whose response couldn't be parsed at all.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    n_total = len(df)
    n_batches = (n_total + batch_size - 1) // batch_size
    results = []
    n_errors = 0

    for i in range(n_batches):
        ckpt_path = os.path.join(ckpt_dir, f"batch_{i:04d}.pkl")
        start, end = i * batch_size, min((i + 1) * batch_size, n_total)
        if os.path.exists(ckpt_path):
            batch_result = pd.read_pickle(ckpt_path)
            print(f"[{phenotype}] batch {i+1}/{n_batches} already done, loaded from checkpoint "
                  f"({len(batch_result)}/{end - start} parsed).")
        else:
            batch_df = df.iloc[start:end]
            chunk_df = pd.DataFrame({'hadm_id': batch_df['note_id'], 'chunk': batch_df['text']})
            batch_result, error_df = parser.parse(chunk_df, phenotype)
            n_batch_errors = 0 if error_df is None else len(error_df)
            n_errors += n_batch_errors
            if batch_result.empty:
                # Every row in this batch failed (e.g. a network/VPN outage) -- do NOT write a
                # checkpoint for it, so the next run retries this batch instead of permanently
                # treating "0 parsed" as "done".
                print(f"[{phenotype}] batch {i+1}/{n_batches} FAILED entirely "
                      f"({n_batch_errors} errors) -- not checkpointed, will retry on next run.")
                continue
            batch_result.to_pickle(ckpt_path)
            print(f"[{phenotype}] batch {i+1}/{n_batches} done: "
                  f"{len(batch_result)}/{end - start} parsed ({n_batch_errors} failures).")
        if not batch_result.empty:
            results.append(batch_result)

    if not results:
        return pd.DataFrame(), n_errors
    return pd.concat(results, ignore_index=True), n_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='bmi_llama3_70b',
                     help="Model key. A scripts.llm.llm_module.BMI_ENDPOINTS key routes to the "
                          "Emory BMI cluster (no API key, needs BMI network/VPN); anything else "
                          "is looked up in LLMParser's OpenRouter model dict (needs OPENROUTER_API_KEY).")
    ap.add_argument('--phenotypes', nargs='+', default=['all'],
                     help="Phenotype names, or 'all' (default) for every phenotype in scripts.constants.PHENOTYPES.")
    ap.add_argument('--max_notes', type=int, default=None,
                     help="Cap notes per phenotype. Default (None) = every golden-labeled note -- "
                          "this is what makes the result a reportable baseline rather than a noisy sample.")
    ap.add_argument('--batch_size', type=int, default=50,
                     help="Notes per checkpoint. Smaller = more resumable but more disk churn.")
    ap.add_argument('--output_dir', default='output/zero_shot_accuracy')
    args = ap.parse_args()

    if args.model not in BMI_ENDPOINTS and not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- add it to .env (see CLAUDE.md), "
                            "or pass a --model from BMI_ENDPOINTS (no key needed, BMI network/VPN only).")

    phenotypes = list(PHENOTYPES) if args.phenotypes == ['all'] else args.phenotypes

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"summary_zero_shot_{args.model}.csv")
    summary_rows = []
    if os.path.exists(summary_path):
        # Resume across phenotypes too: a phenotype whose row is already in a prior summary
        # (from a run that got interrupted after finishing it) is skipped entirely.
        existing_summary = pd.read_csv(summary_path)
        summary_rows = existing_summary.to_dict('records')
        done_phenotypes = set(existing_summary['phenotype'])
        phenotypes = [p for p in phenotypes if p not in done_phenotypes]
        print(f"Resuming: {len(done_phenotypes)} phenotype(s) already completed, "
              f"{len(phenotypes)} remaining.")

    pe = PromptEngineering(prompt_type='zero_shot')
    parser = LLMParser(model_name=args.model, prompt_engineering=pe,
                        api_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)

    for phenotype in phenotypes:
        df = load_golden(phenotype, max_notes=args.max_notes)
        n_pos = int(df[phenotype].sum())
        print(f"\n=== {phenotype}: {len(df)} golden-labeled notes ({n_pos} positive, "
              f"{len(df) - n_pos} negative) ===")
        if df.empty:
            summary_rows.append({'phenotype': phenotype, 'model': args.model, 'n': 0, 'n_parsed': 0,
                                  'accuracy': None, 'precision': None, 'recall': None, 'f1': None})
            continue

        ckpt_dir = os.path.join(args.output_dir, 'checkpoints', args.model, phenotype)
        result_df, n_errors = run_phenotype(parser, phenotype, df, ckpt_dir, args.batch_size)

        if result_df.empty:
            print(f"[{phenotype}] No successfully parsed responses out of {len(df)} notes.")
            summary_rows.append({'phenotype': phenotype, 'model': args.model, 'n': len(df), 'n_parsed': 0,
                                  'accuracy': None, 'precision': None, 'recall': None, 'f1': None})
            continue

        result_df['y_true'] = result_df['hadm_id'].map(dict(zip(df['note_id'], df[phenotype])))
        result_df['y_pred'] = result_df['phenotype_status'].astype(int)
        result_path = os.path.join(args.output_dir, f"{phenotype}_zero_shot_{args.model}_raw.csv")
        result_df.to_csv(result_path, index=False)

        y_true, y_pred = result_df['y_true'], result_df['y_pred']
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        print(f"[{phenotype}] n={len(result_df)}/{len(df)} parsed ({n_errors} failures) "
              f"acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} (TP={tp} FP={fp} FN={fn} TN={tn})")

        summary_rows.append({
            'phenotype': phenotype, 'model': args.model, 'n': len(df), 'n_parsed': len(result_df),
            'n_parse_errors': n_errors, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        })
        # Save after every phenotype, not just at the end -- so a crash mid-run still leaves
        # every already-finished phenotype's result on disk.
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    valid = summary_df.dropna(subset=['f1'])
    if not valid.empty:
        print(f"\n=== Macro-averaged across {len(valid)} phenotype(s) ===")
        print(f"accuracy={valid['accuracy'].mean():.3f} precision={valid['precision'].mean():.3f} "
              f"recall={valid['recall'].mean():.3f} f1={valid['f1'].mean():.3f}")
    print(f"\nSaved summary to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
