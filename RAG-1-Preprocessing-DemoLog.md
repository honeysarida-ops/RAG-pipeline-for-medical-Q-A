# Demo Log — RAG-1-Preprocessing.ipynb

**Pipeline stage:** Step 1 of 4 — Preprocessing  
**Dataset:** MedQA-USMLE (medical multiple-choice questions)

---

## 1.1 Load Data

**Input:** Three `.jsonl` files (train / dev / test) from Kaggle  
**Code:**
```python
df = pd.read_json(file_path, lines=True)
```
**Output:** Three Pandas DataFrames (`train_df`, `dev_df`, `test_df`)  
**Commentary:** `lines=True` is the key flag — without it, pandas would try to parse the whole file as a single JSON object and crash on JSONL.

---

## 1.2 Normalize Text

**Input (raw question):**
> `"A 25-year-old man presents with crushing chest pain..."`

**Code:**
```python
text = text.lower()
text = re.sub(r"\s+", " ", text)
return text.strip()
```
**Output:**
> `"a 25-year-old man presents with crushing chest pain..."`

**Commentary:** Lowercasing + collapsing whitespace is standard hygiene. Keeps embeddings consistent — the model won't treat `"Chest Pain"` and `"chest pain"` as different tokens.

---

## 1.3 Format Text

**Input:** A DataFrame row with `question`, `options` dict, `answer`, `meta_info`  
**Code:** `build_document(row)` — assembles a structured string  
**Output:**
```
QUESTION: a 25-year-old man presents with crushing chest pain...
OPTIONS:
A. Aortic dissection
B. Pulmonary embolism
C. Myocardial infarction
D. GERD
CORRECT ANSWER: Myocardial infarction
USMLE LEVEL: step2
```
**Commentary:** Flattening the nested `options` dict into labeled lines is smart — embedding models and LLMs handle flat text far better than raw JSON. The explicit field labels (`QUESTION:`, `CORRECT ANSWER:`) act as semantic anchors during retrieval.

---

## 1.4 Restructure & Save

**Input:** Formatted `document` column in `train_df`  
**Code:** Iterates rows → builds list of dicts → saves as Parquet  
**Output:** `documents.parquet` with columns: `id`, `text`, `answer`, `answer_idx`, `level`  
**Commentary:** Parquet is a sensible choice over CSV here — it preserves types, handles multi-line text cleanly (no delimiter escaping), and is much faster to reload in subsequent pipeline steps.

---

**Overall:** Clean, minimal preprocessing. The pipeline converts messy JSONL records into flat, labeled text documents ready for chunking and embedding in Step 2 (Index Construction).
