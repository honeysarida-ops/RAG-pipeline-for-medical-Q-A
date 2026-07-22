# 🔍 3. Retrieval

## ❓ Query processing

A single user query is expanded into multiple queries using prompting, drawing on medical knowledge sourced from Google's Generative AI — giving the pipeline more angles to search from than one literal query would.

## 🔗 Merging: Weighted Reciprocal Rank Fusion (RRF)

BM25 and FAISS each return their own ranked list of candidates. These are merged using a **weighted RRF score**:

$$
\text{WeightedRRF}(d) = \sum_{r \in R} w_r \cdot \frac{1}{k + \text{rank}_r(d)}
$$

Where:
- `d` — a document
- `R` — the set of ranked lists being fused (BM25, FAISS)
- `rank_r(d)` — the rank of document `d` in ranking `r`
- `w_r` — the weight for ranking `r` (0–1; always 1 in non-weighted/traditional RRF)
- `k` — a constant (typically 60) controlling fusion behaviour

RRF combines two ranked lists into one relevancy-ordered list, without needing the raw similarity scores from each method to sit on the same scale (Sewell, 2024).

## 🎯 Re-ranking: Cross-encoder

The top candidates from RRF are passed to a **cross-encoder** re-ranking model, which feeds the query and each candidate document into the transformer **together**, rather than encoding them separately. This lets the model pick up on sentence-level semantic interaction between the question and the evidence — a more expensive but more accurate final filter (Jin et al., 2023).

## 🤖 Generation-stage LLM

Retrieved and re-ranked context is passed to **Gemini-2.5-flash**, chosen over relying on ChatGPT's built-in retrieval, which struggled to surface up-to-date, domain-specific medical knowledge on its own.

---
**Previous:** [← 2. Indexing](02-indexing.md) · **Next:** [4. Evaluation →](04-evaluation.md)
