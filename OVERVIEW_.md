# 🩺 RAG Pipeline for Medical Q&A (USMLE-Style Questions)

A retrieval-augmented generation (RAG) system built to answer medical exam-style questions — using hybrid search (BM25 + FAISS), weighted Reciprocal Rank Fusion (RRF), and cross-encoder re-ranking to boost retrieval quality over a baseline LLM.

## 💡 Project overview

LLMs used in the medical domain often lean on outdated internal knowledge, and fine-tuning them to stay current is expensive. This project takes a different approach: **retrieval-augmented generation** — grounding the model's answers in an external, updatable medical knowledge base instead of relying purely on what it memorised during training.

| | |
|---|---|
| 🏥 **Domain** | Medical (USMLE-style question answering) |
| 📊 **Dataset** | MedQA-USMLE — 10,178 English-language multiple-choice questions, each with one correct answer out of four candidates |
| ❓ **Core question** | Can hybrid retrieval + re-ranking meaningfully improve answer quality over a baseline LLM — and where does it still fall short? |

## 🗺️ How this is organised

This project follows a 4-stage pipeline. Click through in order, or jump to whatever's most relevant:

1. 📥 [Data Sourcing](1_DATA_SOURCING.md) — the dataset and why medical Q&A
2. 🗂️ [Indexing](2_INDEXING.md) — building searchable dense + sparse indexes
3. 🔍 [Retrieval](3_RETRIEVAL.md) — fusing and re-ranking search results
4. 📈 [Evaluation](4_EVALUATION.md) — results, limitations, and what I'd improve next

---
**Next:** [1. Data Sourcing →](1_DATA_SOURCING.md)
