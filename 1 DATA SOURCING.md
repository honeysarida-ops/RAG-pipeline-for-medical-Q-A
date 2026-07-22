# 📥 1. Data Sourcing

## 🏥 Why medical Q&A?

Basic LLMs in the medical domain (e.g. plain ChatGPT responses) rely on outdated internal knowledge, whereas RAG grounds responses in external, updatable sources (Amugongo et al., 2025). Fine-tuning an open-source model like LLaMA is also significantly more costly than injecting external knowledge directly into the prompt (Shi et al., 2025) — the core rationale for choosing retrieval over fine-tuning here.

## 📊 The dataset

- **Source:** MedQA-USMLE — a large-scale, open-domain question-answering dataset (Voorhees, 1999) drawn from real medical exams
- **Format:** available in both `.jsonl` and text formats, in English and Chinese
- **Scope for this build:** English-language subset only
- **Size:** 10,178 questions
- **Structure:** each question has exactly one correct answer, selected from four candidates

## 🧹 Pre-processing

- Converted unformatted text data into a consistent schema across `train`, `dev`, and `test` splits
- Added metadata and improved data granularity ahead of indexing
- Skipped heavy chunking — MedQA questions are naturally short, self-contained units, so each question was treated as one retrieval unit rather than needing to be split further

---
**Next:** [2. INDEXING →](02-indexing.md)
