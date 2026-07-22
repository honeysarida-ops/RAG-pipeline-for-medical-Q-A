# Demo Log — RAG-3-LLM-Augmented Retrieval.ipynb

**Pipeline stage:** Step 3 of 4 — LLM-Augmented Retrieval  
**Components:** Query expansion → Hybrid search (FAISS + BM25) → RRF fusion → Cross-encoder reranking

---

## 3.1 Baseline LLM — Query Expansion

**Input:**
```
"45-year-old diabetic woman with nephrotic syndrome"
```

**Prompt:** Instructs Gemini 2.5 Flash to generate five search queries covering disease, pathophysiology, histology, pharmacology, and differential diagnosis. Returns JSON only.

**Output:**
```json
["diabetic nephropathy nephrotic syndrome",
 "nodular glomerulosclerosis",
 "Kimmelstiel Wilson lesion",
 "proteinuria diabetes mellitus",
 "renal pathology diabetic kidney disease"]
```

**Commentary:** This is the "LLM augmentation" core of the pipeline — a single clinical vignette is exploded into five targeted sub-queries. Each targets a different knowledge axis (histological finding, eponym, mechanism, symptom cluster), dramatically widening recall. The JSON-only constraint plus manual fence-stripping (`replace("```json", "")`) is a pragmatic workaround for models that wrap JSON in markdown code blocks.

---

## 3.2 Hybrid Search

### 3.2.1 FAISS Dense Search

**Input:** A single query string (e.g. `"urinary tract infection in pregnancy"`)  
**Code:** Encodes query → L2-normalise → `faiss_index.search(q_emb, top_k=10)`  
**Output:**
```python
[{"id": 4821}, {"id": 302}, {"id": 7193}, ...]  # top-10 doc IDs by cosine similarity
```
**Commentary:** `normalize_embeddings=True` must match how the index was built in Step 2. Without it, scores are meaningless against an `IndexFlatIP` index.

### 3.2.2 BM25 Sparse Search

**Input:** Same query string, lowercased and whitespace-split  
**Code:** `bm25.get_scores()` → `np.argsort(scores)[::-1][:top_k]`  
**Output:**
```
ID: 302  | Score: 14.821
Preview: urinary tract infection pregnancy bacteriuria asymptomatic...

ID: 7193 | Score: 12.340
Preview: dysuria frequency urgency pregnant trimester urine culture...
```
**Commentary:** BM25 shines on exact clinical terms — `"dysuria"`, `"bacteriuria"` — where the dense model might retrieve semantically related but lexically distant documents. Score magnitudes are not comparable to FAISS scores, which is why fusion happens via rank (RRF) rather than score combination.

---

## 3.3 Fusion + Reranking

### 3.3.1 Multiple Queries

**Input:** Original question + 5 LLM-expanded queries  
**Output:** 6 retrieval sets, each containing dense + sparse results (up to 20 docs per query)  
**Commentary:** Running all 6 queries independently then merging casts a wide net — the same document appearing in multiple query result sets will accumulate higher RRF scores, naturally surfacing the most consistently relevant docs.

### 3.3.2 Reciprocal Rank Fusion (RRF)

**Input:** 6 retrieval sets  
**Code:** `score[doc_id] += 1.0 / (k + rank + 1)` where `k=60`  
**Output:**
```
Doc ID: 302  | Score: 0.0892
Doc ID: 4821 | Score: 0.0714
Doc ID: 7193 | Score: 0.0653
...
```
**Commentary:** RRF is a robust rank aggregation method that doesn't require score normalisation. The constant `k=60` softens the advantage of very top-ranked documents, preventing a single high-confidence FAISS result from dominating. Documents retrieved by both FAISS and BM25 across multiple queries naturally float to the top.

### 3.3.3 Cross-Encoder Reranking

**Input:** Top candidates from RRF, paired with the original question  
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Code:** `reranker.predict([(question, doc_text), ...])` → sort descending  
**Output:**
```
Rank 1 | Doc ID: 302  | RRF: 0.0892 | Rerank: 8.41 | Text: "23-year-old pregnant woman..."
Rank 2 | Doc ID: 4821 | RRF: 0.0714 | Rerank: 6.87 | Text: "asymptomatic bacteriuria..."
```
**Commentary:** The cross-encoder is the most expensive component but the most precise — it attends jointly to question and document text rather than comparing independent embeddings. Running it only on RRF-fused candidates (not the full 10K corpus) keeps latency manageable. RRF and rerank scores can disagree; rerank score is the final authority.

---

**Overall:** The pipeline layers three complementary signals — semantic (FAISS), lexical (BM25), and cross-attention reranking — with LLM query expansion multiplying the recall opportunity at each layer. The design trades latency for retrieval quality, which is appropriate for a medical QA task where missing the right document is costly.
