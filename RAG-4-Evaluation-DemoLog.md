# Demo Log — RAG-4-Evaluation.ipynb

**Pipeline stage:** Step 4 of 4 — Evaluation  
**Comparison:** Baseline LLM vs. Advanced RAG  
**Metrics:** Recall@5, MRR, NDCG, Exact Match

---

## 4.1 Retrieval Quality

### 4.1.1 Recall@5

**What it measures:** Whether the correct answer appears anywhere in the top-5 retrieved documents.

**Input:**
```python
gold   = "Myocardial infarction"
ranked = ["Aortic dissection", "Pulmonary embolism", "Myocardial infarction", "GERD", ...]
```
**Output:** `1` (hit) or `0` (miss)  
**Aggregated:** `Mean Recall@5` across all test samples

**Commentary:** Recall@5 is a generous ceiling metric — it asks "did we surface the right answer at all?" rather than "did we surface it first?" Useful for checking whether the retrieval stage is fundamentally working before caring about rank.

---

### 4.1.2 MRR (Mean Reciprocal Rank)

**What it measures:** How highly the first correct answer is ranked. Score of `1/rank` — so rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33, etc.

**Input:**
```python
gold   = "Myocardial infarction"
ranked = ["Aortic dissection", "Pulmonary embolism", "Myocardial infarction", ...]
# → reciprocal rank = 1/3 = 0.333
```
**Output:** Per-sample reciprocal rank score, averaged across all test samples

**Commentary:** MRR penalises systems that bury the correct answer. A system with high Recall@5 but low MRR is retrieving the right document but not ranking it first — a sign that the reranking step needs improvement.

---

### 4.1.3 NDCG (Normalised Discounted Cumulative Gain)

**What it measures:** Ranked quality, discounting gains at lower positions. More nuanced than MRR — rewards placing the correct document as high as possible.

**Code:**
```python
y_true  = [1 if x == gold else 0 for x in ranked]
y_score = list(range(len(ranked), 0, -1))  # positional weights
ndcg_score([y_true], [y_score])
```
**Commentary:** For a binary relevance task (one correct answer), NDCG and MRR carry similar information. The positional weight vector `range(len, 0, -1)` assigns the highest weight to rank 1 and decreases linearly — a reasonable proxy for the standard logarithmic discount.

---

## 4.2 Generation Quality

### 4.2.1 Exact Match

**What it measures:** Whether the top-ranked retrieved document exactly matches the gold answer string.

**Input:**
```python
pred = "Myocardial infarction"
gold = "Myocardial infarction"
# → Exact Match: 1
```
**Commentary:** Exact match is strict — case or spacing differences would score 0. For an MCQ dataset with controlled answer strings, this is appropriate and interpretable. For open-ended generation it would be too brittle.

---

## 4.3 End-to-End QA Accuracy

### 4.3.1 Per-Question Breakdown (first 5 samples)

Both Baseline and Advanced RAG print all four metrics side-by-side per question:

```
========== Question 1 ==========
Question: 23-year-old pregnant woman with dysuria at 22 weeks...
Recall@5: 1
MRR:      0.3333
NDCG:     0.6309
Exact Match: 0
```
**Commentary:** Showing per-question breakdown surfaces failure modes that aggregate numbers hide — e.g. a question where Recall@5=1 but Exact Match=0 reveals the right document was retrieved but not ranked first.

### 4.3.2 Aggregate Summary

Final printout compares Baseline LLM vs. Advanced RAG across all four metrics:

```
              Baseline LLM   Advanced RAG
Recall@5:        x.xxxx         x.xxxx
MRR:             x.xxxx         x.xxxx
NDCG:            x.xxxx         x.xxxx
Exact Match:     x.xxxx         x.xxxx
```

**Commentary:** This head-to-head is the payoff of the whole pipeline. Improvements in Advanced RAG validate the cost of query expansion, hybrid search, RRF fusion, and cross-encoder reranking. Any metric where Advanced RAG underperforms would indicate that the added complexity introduced noise rather than signal — worth checking especially for Exact Match, which is most sensitive to reranking errors.

---

**Overall:** The evaluation framework is clean and consistent — same four metrics applied identically to both systems, with per-question and aggregate views. One gap worth noting: there is no latency or cost measurement. Given that Advanced RAG makes 6× the retrieval calls plus an LLM query-expansion API call per question, a throughput comparison would complete the picture.
