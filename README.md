# Evidence Before Answers

### Evidence-Completeness-Aware Customer Support RAG

> **A retrieval system for customer support that does not treat “similar” as “safe to answer from.”**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00A67E)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-FF6F00)](https://www.sbert.net/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-TWCS-111827)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
[![Evaluation](https://img.shields.io/badge/Human%20Evaluation-150%20Examples-2563EB)](./output/)

---

## TL;DR

Traditional support RAG asks:

> **“Which historical conversation looks most similar?”**

This project asks a stricter question:

> **“Which historical conversation is relevant enough, and contains enough observable evidence, to safely ground an answer?”**

I built an **evidence-completeness-aware retrieval and decision pipeline** over AmazonHelp conversations from the Customer Support on Twitter dataset.

The system combines:

**semantic similarity + intent + issue state + evidence completeness**

and supports three outcomes:

```text
AUTO_RESPOND
ASK_CLARIFICATION
ESCALATE
```

The central engineering principle is:

> **Relevance first. Evidence second. Answer third.**

---

# 1. Why this problem?

Customer-support RAG has a subtle failure mode:

A historical conversation can be **semantically similar but operationally useless**.

For example, two conversations may both mention:

* a refund,
* a delivery problem,
* a damaged product,
* a payment issue,

while only one contains an actual support action, useful instructions, or an observable outcome.

Similarity alone cannot distinguish these cases.

This project therefore investigates:

> **Does observable evidence completeness improve the quality of historical conversations selected for support grounding compared with similarity-only retrieval?**

The answer is nuanced:

**Yes for evidence quality — but not automatically for relevance.**

That distinction is the most important finding of the project.

---

# 2. What I built

```text
                    CUSTOMER QUERY
                          │
                          ▼
                ┌───────────────────┐
                │ Intent Detection   │
                │ State Detection    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Semantic Retrieval │
                │ FAISS + MiniLM     │
                └─────────┬─────────┘
                          │
                    Top semantic
                    candidates
                          │
                          ▼
          ┌──────────────────────────────┐
          │ Evidence Completeness Layer  │
          │                              │
          │ D1 Context specificity       │
          │ D2 Action specificity        │
          │ D3 DM redirect               │
          │ D4 Observable outcome        │
          └──────────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Evidence-aware       │
              │ reranking            │
              └──────────┬───────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │ Decision / Safety Layer       │
          │                              │
          │ AUTO_RESPOND                 │
          │ ASK_CLARIFICATION            │
          │ ESCALATE                     │
          └──────────────┬───────────────┘
                         │
                         ▼
                 GROUNDED RESPONSE
```

The architecture deliberately separates:

* **relevance**
* **evidence quality**
* **decision confidence**

rather than collapsing everything into one similarity score.

---

# 3. The core idea: Evidence Completeness

Each historical conversation receives four observable evidence dimensions.

### D1 — Context Specificity

Does the customer provide enough concrete information about the issue?

Examples include:

* specific product/service context;
* domain-specific terms;
* identifying information;
* numbers or other concrete details.

---

### D2 — Action Specificity

What did the support agent actually do?

The system distinguishes between:

```text
generic / redirect
        ↓
specific instruction
        ↓
specific guidance
        ↓
specific action
```

An apology or acknowledgement alone is not treated as a meaningful support action.

---

### D3 — DM Redirect

Was the customer redirected to private messaging before a concrete public support action?

This matters because a conversation containing only:

> “Please DM us.”

provides much less observable evidence than one containing an actual support action and outcome.

---

### D4 — Observable Outcome

What happened after the support interaction?

The model captures:

```text
positive reaction
negative reaction
neutral reaction
silence after action
no action observed
```

The goal is not to claim that “positive reaction = resolution.”

Instead, it captures whether there is observable evidence after the support interaction.

---

# 4. Evidence score

The prototype combines the four dimensions into:

```text
EC =
    0.30 × D1
  + 0.30 × D2
  + 0.30 × D4
  + 0.10 × (1 − D3)
```

Evidence tiers:

| Tier   |        Score |
| ------ | -----------: |
| HIGH   |       ≥ 0.75 |
| MEDIUM | 0.50 – 0.749 |
| LOW    |       < 0.50 |

This is intentionally described as an **evidence-quality heuristic**, not a probability of successful resolution.

---

# 5. Dataset reconstruction

The project uses the **Customer Support on Twitter (TWCS)** dataset.

Raw dataset:

```text
2,811,774 tweets
7 columns
```

The raw relationship fields are:

```text
tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id
```

Instead of assuming adjacent rows form conversations, I reconstructed conversation components using both:

```text
in_response_to_tweet_id
response_tweet_id
```

This produced:

| Reconstruction stage             |      Count |
| -------------------------------- | ---------: |
| Tweets indexed                   |  2,811,774 |
| Conversation components          |    798,197 |
| AmazonHelp-containing components |     82,556 |
| Final AmazonHelp corpus          | **78,781** |
| Average turns                    |       4.29 |
| 4+ turn conversations            |     36,509 |

The final corpus removes conversations with multiple customers or obvious mixed-brand contamination.

The original `twcs.csv` is intentionally **not included in Git** because of its size and dataset licensing/distribution considerations.

---

# 6. Retrieval architecture

## Embedding model

```text
all-MiniLM-L6-v2
```

Each historical customer query is converted into a normalized embedding.

## Vector index

```text
FAISS IndexFlatIP
```

Index size:

```text
78,781 vectors
384 dimensions
```

The system retrieves semantically similar conversations and then applies additional relevance/evidence signals.

---

# 7. Retrieval policies evaluated

Four policies were compared.

### Baseline A — Semantic Only

```text
score = semantic_similarity
```

This represents conventional similarity-based historical retrieval.

### Baseline B — Semantic + Intent

Adds compatibility between the incoming query and historical intent.

### Proposed — Semantic + Evidence

```text
score =
    0.70 × semantic_similarity
  + 0.30 × evidence_completeness
```

### Full system

```text
score =
    0.55 × semantic_similarity
  + 0.15 × intent_compatibility
  + 0.10 × state_compatibility
  + 0.20 × evidence_completeness
```

The full system is intentionally more conservative because support relevance is not only about textual similarity.

---

# 8. Human evaluation

A **150-example human-verified evaluation set** was created.

The final annotation set contains:

| Human label                | YES | NO |
| -------------------------- | --: | -: |
| Evidence complete          |  75 | 75 |
| Action observed            | 105 | 45 |
| Customer reaction observed | 104 | 46 |
| DM redirect                |  60 | 90 |

Validation:

```text
150 / 150 annotations complete
0 duplicate conversation IDs
0 incomplete annotation rows
```

The labels were produced through **human verification of machine-assisted pre-labels**.

This is intentionally disclosed rather than presented as fully blind annotation.

---

# 9. Human alignment of the evidence model

A logistic regression model was trained using:

```text
D1 context specificity
D2 action specificity
D3 DM redirect
D4 observable outcome
```

Target:

```text
human evidence_complete label
```

### 5-fold cross-validation

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **62.67%** |
| Precision | **62.67%** |
| Recall    | **62.67%** |
| F1        | **62.67%** |
| ROC-AUC   | **0.6429** |

This is not presented as support-agent accuracy.

It measures whether the engineered evidence dimensions contain measurable signal corresponding to human evidence-completeness judgments.

The result is deliberately imperfect — which is useful.

It indicates that the heuristic captures meaningful structure, but is not yet a substitute for human judgment.

---

# 10. The main experiment

## Leave-one-out evaluation

To prevent trivial self-retrieval:

* the source conversation for each human example was excluded;
* the top 20 semantic candidates were used as the candidate pool;
* every policy selected from the remaining candidates.

### Results

| Retrieval policy  |    Mean EC | MEDIUM/HIGH |       HIGH | Mean similarity |
| ----------------- | ---------: | ----------: | ---------: | --------------: |
| Semantic-only     |     0.5233 |      49.33% |     21.33% |      **0.7772** |
| Semantic + intent |     0.5425 |      54.67% |     20.67% |          0.7651 |
| Semantic + EC     | **0.7824** |  **90.67%** | **75.33%** |          0.7489 |
| Full system       |     0.7253 |      86.00% |     60.67% |          0.7457 |

### Change vs semantic-only

**Semantic + EC:**

```text
Mean EC:
0.5233 → 0.7824
+0.2591

MEDIUM/HIGH:
49.33% → 90.67%
+41.33 percentage points

HIGH:
21.33% → 75.33%
+54.00 percentage points
```

At the same time:

```text
Mean semantic similarity:
0.7772 → 0.7489
```

This is an important trade-off.

The proposed retrieval policy is **not simply finding more similar conversations**.

It is deliberately moving toward conversations containing stronger observable support evidence.

---

# 11. The result I do NOT claim

It would be misleading to say:

> “The system has 90.67% retrieval accuracy.”

It does not.

The 90.67% number means:

> **90.67% of selected historical conversations were MEDIUM or HIGH according to the evidence-completeness heuristic.**

It does **not** establish:

* exact issue relevance;
* correct resolution;
* response correctness;
* customer satisfaction;
* unsupported-claim rate.

This distinction is central to the evaluation.

---

# 12. What the experiment actually demonstrates

The strongest supported conclusion is:

> **Evidence-aware retrieval substantially changes the evidence profile of retrieved historical conversations.**

It does not yet prove:

> **Evidence-aware retrieval always retrieves the most relevant resolution.**

This led directly to the failure analysis.

---

# 13. Failure analysis

Five recurring failure modes were identified.

## 13.1 Evidence can override relevance

A highly documented historical conversation can concern the wrong issue.

Example pattern:

```text
Current issue:
wrong product / replacement problem

Retrieved:
counterfeit-product / related complaint

Evidence quality:
HIGH

Issue relevance:
imperfect
```

### Lesson

Evidence quality cannot substitute for issue relevance.

---

## 13.2 Context-poor follow-ups

Messages such as:

```text
"Awaiting your response"

"@115823"

"Obrigado."

"Still waiting to hear back from you"
```

are difficult to retrieve against accurately without their preceding context.

### Lesson

A short message should not automatically be treated as a complete retrieval query.

The system should detect insufficient context and prefer clarification or conversation-history expansion.

---

## 13.3 Fine-grained state mismatch

Broad intent is insufficient.

For example:

```text
delivery
├── late
├── cancelled
├── delivered_not_received
└── agent_issue
```

Similarly:

```text
return
├── damaged
├── pickup_reschedule
└── refund_pending
```

### Lesson

State compatibility needs stronger influence than broad intent matching.

---

## 13.4 High evidence ≠ relevant resolution

A conversation can contain:

* a concrete action;
* an observable outcome;
* a follow-up;
* strong support evidence;

and still address the wrong customer problem.

### Lesson

Evidence completeness should be treated as a **quality-of-grounding signal**, not a relevance metric.

---

## 13.5 Reranking can displace a strong semantic match

In some examples, semantic retrieval already identifies the right issue.

Aggressive EC reranking can move that conversation below a more evidence-rich but less relevant candidate.

### Lesson

The better architecture is:

```text
Semantic relevance
        ↓
Intent/state gate
        ↓
Evidence-aware reranking
        ↓
Decision threshold
```

rather than allowing evidence to freely dominate relevance.

---

# 14. Decision engine

The retrieval system feeds a conservative decision layer.

### AUTO_RESPOND

Requires strong semantic, intent, and evidence signals.

### ASK_CLARIFICATION

Used when the issue is sufficiently identifiable but the evidence is not strong enough for confident automation.

### ESCALATE

Used when relevance or evidence falls below the configured thresholds.

Current production thresholds:

```text
AUTO_MIN_SEMANTIC = 0.72
AUTO_MIN_INTENT   = 0.75
AUTO_MIN_EC       = 0.75

CLARIFY_MIN_SEMANTIC = 0.65
CLARIFY_MIN_INTENT   = 0.50
CLARIFY_MIN_EC       = 0.50
```

The system therefore treats uncertainty as a decision signal rather than forcing every request into an answer.

---

# 15. Example decision behavior

### Strong evidence

```text
Query:
"My product arrived damaged."

Retrieved evidence:
HIGH

Intent:
compatible

State:
compatible

Decision:
AUTO_RESPOND
```

### Strong relevance but incomplete evidence

```text
Query:
"My order says delivered but I never received it."

Retrieved evidence:
relevant but below evidence threshold

Decision:
ASK_CLARIFICATION
```

### Insufficient context

```text
Query:
"@AmazonHelp"

Decision:
ESCALATE / ASK_CLARIFICATION
```

The exact decision depends on the retrieved candidate scores and compatibility signals.

---

# 16. Engineering decisions

### Why FAISS?

The corpus contains 78K+ conversations.

A vector index makes semantic retrieval fast and reproducible without requiring a hosted vector database.

### Why `all-MiniLM-L6-v2`?

It provides a practical balance between:

* embedding quality;
* local execution;
* resource requirements;
* reproducibility.

### Why deterministic evidence features?

The first version prioritizes interpretability.

Every evidence score can be decomposed into:

```text
D1
D2
D3
D4
```

This makes failure analysis possible.

### Why not use EC alone?

Because the experiments demonstrated that EC can retrieve highly documented but irrelevant conversations.

That failure is not hidden; it is part of the system design lesson.

---

# 17. Tech stack

| Layer              | Technology                               |
| ------------------ | ---------------------------------------- |
| Language           | Python                                   |
| Data processing    | pandas, NumPy                            |
| Embeddings         | Sentence Transformers                    |
| Embedding model    | `all-MiniLM-L6-v2`                       |
| Vector search      | FAISS                                    |
| ML alignment model | scikit-learn Logistic Regression         |
| Retrieval          | Semantic + intent + state + EC           |
| Evaluation         | Python evaluation harness + human labels |
| Dataset            | Customer Support on Twitter / TWCS       |
| Version control    | Git / GitHub                             |

---

# 18. Repository structure

```text
Hiver-AI/
│
├── data/
│   └── README.md
│
├── src/
│   ├── reconstruct_amazonhelp.py
│   ├── compute_evidence_features.py
│   ├── build_ec_dataset.py
│   ├── prepare_retrieval_corpus.py
│   ├── build_semantic_index.py
│   ├── retrieve.py
│   ├── decision_engine.py
│   ├── create_annotation_sample.py
│   ├── validate_annotations.py
│   ├── train_evidence_model.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_human_alignment.py
│   ├── evaluate_leave_one_out.py
│   └── analyze_failure_modes.py
│
├── output/
│   ├── human_verification_150.csv
│   ├── human_annotations_clean.csv
│   ├── ec_model_predictions.csv
│   ├── human_alignment_leave_one_out.json
│   ├── human_alignment_leave_one_out_details.csv
│   └── failure_analysis_candidates.csv
│
├── requirements.txt
├── REPORT.md
├── README.md
└── .gitignore
```

Large generated artifacts and the raw dataset are intentionally excluded from Git where appropriate.

---

# 19. Reproducibility

## 1. Clone

```bash
git clone https://github.com/shahithmohamed202-collab/Hiver-AI.git
cd Hiver-AI
```

## 2. Create environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Add the dataset

Download the Customer Support on Twitter dataset and place:

```text
data/twcs.csv
```

The raw dataset is intentionally not committed to the repository.

---

# 20. Reproduce the pipeline

### Reconstruct AmazonHelp conversations

```powershell
python src/reconstruct_amazonhelp.py
```

### Compute evidence features

```powershell
python src/compute_evidence_features.py
```

### Build evidence dataset

```powershell
python src/build_ec_dataset.py
```

### Prepare retrieval corpus

```powershell
python src/prepare_retrieval_corpus.py
```

### Build semantic index

```powershell
python src/build_semantic_index.py
```

The semantic index is expensive relative to the other preprocessing steps, so the generated index can be reused when the corpus/order has not changed.

---

# 21. Reproduce evaluation

### Validate the human evaluation set

```powershell
python src/validate_annotations.py
```

### Train the human-alignment model

```powershell
python src/train_evidence_model.py
```

### Run leave-one-out evaluation

```powershell
python src/evaluate_leave_one_out.py
```

### Extract failure-analysis candidates

```powershell
python src/analyze_failure_modes.py
```

The final evaluation intentionally excludes each example's source conversation to prevent trivial self-retrieval.

---

# 22. Evaluation artifacts

The most important generated artifacts are:

```text
output/
├── human_annotations_clean.csv
├── ec_model_predictions.csv
├── human_alignment_leave_one_out.json
├── human_alignment_leave_one_out_details.csv
└── failure_analysis_candidates.csv
```

### `human_annotations_clean.csv`

Validated 150-example human evaluation set.

### `ec_model_predictions.csv`

Cross-validated predictions from the evidence-alignment model.

### `human_alignment_leave_one_out.json`

Primary retrieval comparison and aggregate evaluation results.

### `human_alignment_leave_one_out_details.csv`

Per-example retrieval details for auditing.

### `failure_analysis_candidates.csv`

Candidate examples used to identify systematic retrieval failure modes.

---

# 23. What I would improve with one more week

The current experiment separates **evidence completeness** from **semantic similarity**, but the next evaluation should explicitly label retrieval relevance.

### Next evaluation

For each retrieved conversation, add:

```text
EXACT_MATCH
SAME_ISSUE_DIFFERENT_STATE
RELATED
IRRELEVANT
```

Then report four independent dimensions:

```text
Retrieval relevance
        +
Evidence completeness
        +
Response groundedness
        +
Resolution correctness
```

This would prevent a single headline metric from hiding important failure modes.

---

# 24. Next-generation architecture

The main lesson from the current experiment leads to this architecture:

```text
                    CUSTOMER MESSAGE
                           │
                           ▼
                 CONTEXT SUFFICIENCY
                           │
              ┌────────────┴────────────┐
              │                         │
        insufficient                sufficient
              │                         │
              ▼                         ▼
       ASK CLARIFICATION        INTENT + STATE
                                        │
                                        ▼
                              SEMANTIC RETRIEVAL
                                        │
                                        ▼
                              RELEVANCE GATING
                                        │
                                        ▼
                            EVIDENCE RERANKING
                                        │
                                        ▼
                               GROUNDEDNESS
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                         ▼              ▼              ▼
                    AUTO_REPLY   CLARIFICATION     ESCALATE
```

This is the direction I would take toward a production-quality support agent.

---

# 25. Key takeaways

### Finding 1

Similarity-only retrieval often selects historically similar conversations with weak observable support evidence.

### Finding 2

Adding evidence completeness substantially shifts retrieval toward conversations with stronger observable evidence.

### Finding 3

That improvement is measurable:

```text
Mean EC
0.5233 → 0.7824

MEDIUM/HIGH
49.33% → 90.67%

HIGH
21.33% → 75.33%
```

### Finding 4

Evidence completeness is not relevance.

A high-quality historical resolution for the wrong issue is still the wrong evidence.

### Finding 5

The strongest architecture is therefore not:

```text
SIMILARITY → ANSWER
```

but:

```text
RELEVANCE → EVIDENCE → DECISION → ANSWER
```

---

# 26. Final statement

This project started with a simple assumption:

> Better semantic similarity should produce better support grounding.

The evaluation showed that this is incomplete.

A support system also needs to understand whether a historical conversation contains **observable evidence that can safely support the next response**.

The resulting system therefore treats retrieval as more than nearest-neighbor search.

It is a constrained decision problem:

> **Find a relevant historical case, determine whether its evidence is sufficient, and only then decide whether an automated answer is justified.**

That is the core idea behind **Evidence Before Answers**.

---

## Author

**Shahith Mohamed K**

B.Tech Artificial Intelligence & Data Science
Sri Eshwar College of Engineering

**Focus:** AI/ML · Generative AI · Retrieval Systems · Full-Stack Engineering · Data Science

---

### Evaluation note

All reported numbers in this README come from the project's recorded evaluation runs. Metrics are intentionally described according to what they actually measure; evidence-completeness improvement is not presented as answer accuracy or resolution accuracy.
