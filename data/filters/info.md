# data/filters — File Reference

## `golden_labels.csv`
* **Purpose**: Gold-standard phenotype labels at the *hadm_id* (hospital admission) level.
* **Usage**: Serves as the authoritative label set for supervised learning tasks.

---

## `golden_notes.csv` *(incomplete in repo)*
* **Content**: Discharge-summary notes for each *hadm_id*.
* **Status**: The repository only contains a **single-row sample**.
* **Full dataset**: Must be generated after obtaining **official access to MIMIC-IV 2.2 and running the accompanying **`golden_notes.sql`** query.

---

## `silver_labels.json`
* **Purpose**: Defines **ICD-9 and ICD-10 codes** corresponding to the phenotypes of interest.
* **Role in pipeline**: Consumed by the `mimic_parser` to construct the **training set** for supervised learning models.

---

## `easy_cases.csv`
* **Definition**: Discharge summaries where the phenotype was **correctly classified**:
  * by **both** GPT-4o **and** Llama-405B,
  * **and** correctly identified by the other benchmark models (Llama-3 3B, 8B, 70B, Gemini Flash 2)
  * in **> 70 %** of their configuration runs.
* **Usage**: Enables analysis of entropy for low-difficulty subsets.

---

## `challenging_cases.csv`
* **Definition**: Admissions where the phenotype was **misclassified**:
  * by **both** GPT-4o **and** Llama-405B,
  * **and** misclassified by the other benchmark models (Llama-3 3B, 8B, 70B, Gemini Flash 2)
  * in **> 50 %** of their configuration runs.
* **Usage**: Used to examine entropy in high-difficulty subsets and assess their impact on model uncertainty.

---

### Notes
* Access to the **full discharge summaries** requires proper **PhysioNet/MIMIC-IV 2.2 credentials**.

---

