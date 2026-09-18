# Evidence Before Answers

### Evidence-Completeness-Aware Customer Support RAG

> **Evidence before answers.**
>
> A customer-support retrieval system that combines semantic relevance, issue/state compatibility, and observable evidence completeness to select historical conversations before drafting a response.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-orange)](https://www.sbert.net/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)](https://fastapi.tiangolo.com/)

---

## TL;DR

This project investigates a simple question:

> **Does historical evidence quality improve support retrieval beyond semantic similarity alone?**

Using the Customer Support on Twitter dataset, the project reconstructs AmazonHelp conversations and builds an evidence-aware retrieval pipeline.

The system evaluates historical conversations using four observable evidence dimensions:

- **D1 — Context specificity**
- **D2 — Action specificity**
- **D3 — DM/redirect occurrence**
- **D4 — Observable customer outcome**

These dimensions are combined into an **Evidence Completeness (EC)** score.

The retrieval pipeline then combines:

**Relevance → Evidence → Decision → Answer**

The central finding is that adding evidence completeness substantially changes which historical conversations are selected. However, the experiments also show why **evidence quality must not be confused with issue relevance, resolution correctness, or answer correctness**.

---

# 1. Problem Framing

Customer-support RAG systems often retrieve conversations primarily by semantic similarity.

That creates an important failure mode:

> A conversation can be semantically similar to the current customer issue but contain little observable evidence about what support actually did or what happened afterward.

For example, a historical conversation may contain:

- the same product,
- a similar complaint,
- similar wording,

but only:

- an apology,
- a generic redirect,
- no observable action,
- and no customer outcome.

This project therefore introduces an additional retrieval signal:

> **How much observable support evidence does this historical conversation contain?**

The goal is not to assume that high evidence automatically means a correct answer.

Instead, the system explicitly separates:

1. **Semantic relevance**
2. **Issue/state compatibility**
3. **Evidence completeness**
4. **Final response confidence**

---

# 2. What Good Means

A useful support retrieval system should retrieve historical examples that are:

### Relevant

The historical conversation should address the same or closely related customer issue.

### Actionable

The conversation should contain observable support actions such as:

- specific instructions,
- investigation routes,
- support channels,
- concrete troubleshooting,
- refund/return guidance,
- escalation paths.

### Evidence-backed

The conversation should contain observable evidence about what happened after support intervention.

### State-aware

The system should distinguish fine-grained states such as:

- delayed,
- cancelled,
- not received,
- damaged,
- replacement,
- refund pending,
- payment declined.

### Honest about uncertainty

When evidence is weak or relevance is insufficient, the system should not force an answer.

---

# 3. What This Project Does Not Build

This project intentionally does **not** claim to build:

- a production customer-support agent,
- a fully autonomous support system,
- a perfect resolution classifier,
- a general-purpose LLM customer-service model,
- a human-equivalent support representative,
- a guaranteed-correct answer generator.

The main research focus is **retrieval quality and evidence completeness**, with a decision layer demonstrating how the signals can be used for answer/clarification/escalation decisions.

---

# 4. System Architecture

```text
                         Customer Query
                              |
                              v
                    +----------------------+
                    | Intent / State       |
                    | Detection            |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Semantic Retrieval   |
                    | FAISS + MiniLM       |
                    +----------+-----------+
                               |
                         Top candidate pool
                               |
                               v
                    +----------------------+
                    | Evidence Features    |
                    | D1 / D2 / D3 / D4    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Evidence-Aware       |
                    | Reranking             |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Decision Engine      |
                    | AUTO / CLARIFY /     |
                    | ESCALATE             |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Grounded Response    |
                    +----------------------+