# Demo Log — RAG-2-Index Construction.ipynb

**Pipeline stage:** Step 2 of 4 — Index Construction  
**Dataset:** MedQA-USMLE (10,178 training questions)

---

## 2.1 Load Dataset

**Input:** `documents.pkl` — output from RAG-1 Preprocessing  
**Code:**
```python
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)
```
**Output:** List of dicts, each with `id`, `text`, `answer`, `answer_idx`, `level`  
**Commentary:** Pickle is fast for in-pipeline handoffs but not portable — good for Kaggle notebooks where the environment is controlled; a real production system would prefer Parquet or a document store.

---

## 2.2 Chunking

**Input:** `documents[0]["text"]` — a single formatted MCQ string  
**Code:** `chunk_text(text, chunk_size=600)` — splits by character count  
**Output:** Multiple 600-char chunks per question  
**Commentary (design decision):** The author correctly identifies that character-based chunking is *wrong* here. Each MCQ is a self-contained logical unit — splitting mid-question would break the question-options-answer triad. One question = one chunk is the right call for this dataset.

---

## 2.3 Embedding

### 2.3.1 FAISS (Dense / Semantic)

**Input:** 10,178 question strings  
**Code:** `SentenceTransformer("all-MiniLM-L6-v2")` on GPU if available, `batch_size=1267`

```python
embeddings = embedding_model.encode(
    question_texts,
    batch_size=1267,
    normalize_embeddings=True,
    show_progress_bar=True
)
```

**Output:** NumPy array of shape `(10178, 384)`, stored as `faiss_index.bin`  
**Index type:** `IndexFlatIP` — exact inner-product search (works correctly because embeddings are L2-normalised, making inner product equivalent to cosine similarity)  
**Commentary:** `normalize_embeddings=True` is essential when using `IndexFlatIP` — without it, scores would reflect vector magnitude rather than semantic similarity. Batch size of 1267 (~10178/8) is a sensible GPU-friendly choice.

### 2.3.2 BM25 (Sparse / Keyword)

**Input:** Same 10,178 documents, lowercased and whitespace-split  
**Code:** `BM25Okapi(tokenized_docs)` from `rank-bm25`  
**Output:** Statistical BM25 index, saved as `tokenized_docs.pkl`  
**Commentary:** BM25 is a bag-of-words model — no embeddings, pure term frequency/IDF. It complements dense search by catching exact clinical terminology (drug names, anatomical terms) that semantic models sometimes gloss over. Saving `tokenized_docs.pkl` separately is required since BM25 holds no persistent index file of its own; the tokenized corpus *is* the index.

---

**Overall:** The dual-index strategy (FAISS + BM25) sets up a hybrid retrieval system in Step 3. Dense handles semantic similarity; sparse handles lexical precision — together they cover the breadth of USMLE question phrasing.
