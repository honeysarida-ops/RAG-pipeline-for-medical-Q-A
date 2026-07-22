# 📈 4. Evaluation

The pipeline was evaluated against a baseline LLM (no retrieval) across two groups of metrics: retrieval quality and answer quality.

## 🎯 Retrieval quality metrics

| Metric | Result vs. baseline | Suggested improvement |
|---|---|---|
| Precision | ✅ +56% | Improve re-ranking to reduce irrelevant results |
| Recall | ⚠️ −21.6%* | Enhance domain-specific embeddings and query understanding |
| MRR | ✅ +13.83% | Optimise retrieval depth and evidence selection |
| NDCG | ✅ +4.59% | Improve ranking models and relevance scoring |
| Exact Match | ✅ +82% | Incorporate clinical reasoning into answer-focused retrieval |
| **Overall performance** | ✅ **+65% over baseline** | Further optimisation needed for medical-domain complexity |

**\*On the recall drop:** this isn't a genuine regression — it reflects a scope difference. Baseline recall is artificially inflated because the correct answer is always one of 4–5 multiple-choice options, while the RAG pipeline retrieves against the full 10,178-document corpus. The two recall figures measure fundamentally different things and shouldn't be compared directly.

**What precision and recall are doing across the two retrieval stages:**
- **Stage 1 (hybrid search):** optimises for *recall* — casting a wide net across the corpus so relevant documents aren't missed, while hybrid BM25+FAISS matching also improves exact-match accuracy
- **Stage 2 (re-ranking):** optimises for *precision* — ensuring the top-k documents surfaced to the "clinician" (or downstream LLM) are the most useful, most relevant subset

**On MRR, NDCG, and Exact Match:** all three measure quality from different angles — MRR reflects how quickly the pipeline surfaces a correct/relevant result, NDCG reflects the overall quality of the ranking order, and Exact Match reflects how often the final answer is correct.

## 🩻 Answer quality metrics

| Aspect | Performance | Interpretation |
|---|---|---|
| Correctness | 🟢 Large improvement | Retrieved information better supported the correct answer |
| Grounding | 🟢 Large improvement | Answers were better supported by retrieved documents, with fewer unrelated responses |
| Completeness | 🟡 Moderate improvement | More relevant information was retrieved, but some answers still lacked enough detail for complex, multi-part medical questions |

## ⚠️ Known limitations

| Challenge | Impact | Future direction |
|---|---|---|
| Complex medical terminology | Limits retrieval accuracy | Medical-specific embedding models |
| Similar disease concepts | Causes retrieval ambiguity | Knowledge graphs + entity linking |
| Clinical reasoning requirements | Relevant evidence doesn't always support correct answer selection | Reasoning-aware retrieval |
| Ranking quality | Correct evidence isn't always top-ranked | Advanced re-ranking methods |
| Multi-step questions | Require synthesising multiple sources | Multi-hop retrieval techniques |

## 💭 Reflection

Hybrid search (BM25 + FAISS + RRF) improved every retrieval metric over baseline — supporting the case for grounding medical Q&A in external, updatable knowledge rather than relying solely on a model's internal training data. The biggest remaining gap is **multi-hop reasoning**: many medical questions require connecting evidence across multiple documents, which single-pass hybrid retrieval doesn't fully solve.

Other directions I'd explore next: 🔁 self-refinement loops (iterative re-retrieval before finalising an answer), ⚡ runtime efficiency improvements at indexing time, and 🔒 ethical considerations around bias and data privacy when routing real patient data through external LLM APIs.

---
**Previous:** [← 3. Retrieval](03-retrieval.md)
