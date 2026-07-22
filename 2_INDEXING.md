# 🗂️ 2. Indexing

Two complementary indexes were built over the question corpus, since dense and sparse retrieval each catch different kinds of relevance.

## 🧠 Dense retrieval (FAISS)

- Embedding model: `all-mpnet-base-v2` (mini variant, to keep load times down)
- Indexed with **FAISS** for fast approximate-nearest-neighbour search over the embedding space
- Good at semantic/contextual similarity — catching questions phrased differently from the source text but meaning the same thing
- ⚠️ Trade-off: dense embeddings risk hallucination on highly technical jargon (e.g. specific antibody or pathogen names) the model wasn't well-trained to represent

## 🔑 Sparse retrieval (BM25)

- Built using `rank_bm25`
- Highly sensitive to exact keyword and phrase matches — strong at retrieving exact clinical terms (e.g. bacteria names, disease symptoms) that dense embeddings can miss
- A standard baseline for open-domain QA retrieval (Robertson and Zaragoza, 2009); most sparse-vector approaches degrade sharply once exact keywords are missing (Kim, 2025)

## 🤝 Hybrid search

BM25 and FAISS were run **simultaneously**, not as substitutes for each other — pairing exact keyword precision with semantic flexibility. This hybrid setup is the foundation the retrieval-stage fusion (RRF) builds on.

## ⚡ Batch processing

Embedding ran as one large batch job across all 10,178 questions, to keep indexing time manageable at this scale.

---
**Previous:** [← 1. Data Sourcing](1_DATA_SOURCING.md) · **Next:** [3. Retrieval →](3_RETRIEVAL.md)
