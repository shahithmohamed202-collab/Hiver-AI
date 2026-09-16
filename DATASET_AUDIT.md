\# Dataset Audit



\## 1. Dataset



This project uses the Customer Support on Twitter (TWCS) dataset.



The raw dataset is stored locally as:



`data/twcs.csv`



The raw file is treated as immutable and is not modified by the pipeline.



\## 2. Raw Dataset Schema



The original dataset contains seven columns:



\- `tweet\_id`

\- `author\_id`

\- `inbound`

\- `created\_at`

\- `text`

\- `response\_tweet\_id`

\- `in\_response\_to\_tweet\_id`



These fields provide the tweet identifiers, authorship, direction of the message, timestamp, message text, and conversation relationships.



\## 3. Raw Dataset Statistics



| Property | Value |

|---|---:|

| Rows | 2,811,774 |

| Columns | 7 |

| Approximate file size | 493 MB |

| Unique tweet IDs | 2,811,774 |

| Valid parent links | 2,013,577 |

| Valid response links | 2,013,577 |



The dataset contains a large collection of customer-support interactions across multiple support accounts.



\## 4. Conversation Reconstruction



The raw dataset does not directly provide clean conversation-level records.



The reconstruction pipeline therefore builds conversation components using:



\- `in\_response\_to\_tweet\_id`

\- `response\_tweet\_id`



Both relationship directions are considered.



A union-find structure is used to connect tweets belonging to the same conversation component.



\### Reconstruction result



| Metric | Value |

|---|---:|

| Conversation components | 798,197 |

| Components containing AmazonHelp | 82,556 |

| Retained AmazonHelp conversations | 78,781 |

| Removed: no customer | 0 |

| Removed: multiple customers | 3,519 |

| Removed: mixed support brand | 256 |

| Average retained turns | 4.29 |

| Conversations with 4+ tweets | 36,509 |

| Percentage with 4+ tweets | 46.34% |



The approximately 82.5K AmazonHelp-containing components represent the initial candidate pool. After conversation-quality filtering, 78,781 conversations are retained for experimentation.



\## 5. Conversation Quality Filters



The reconstruction process retains conversations that satisfy the following conditions:



1\. The conversation contains the AmazonHelp support account.

2\. At least one customer message is present.

3\. Exactly one customer is associated with the conversation.

4\. The component does not contain an obvious mixture of unrelated support brands.

5\. Tweets are ordered chronologically.



The filtering is intended to reduce ambiguous conversation structures before evidence extraction and retrieval evaluation.



\## 6. Evidence Extraction



Each retained conversation is analyzed using four evidence dimensions.



\### D1 — Context Specificity



Measures how specifically the customer's opening query describes the issue.



Signals include:



\- content-word count

\- domain-specific terminology

\- useful numeric information



\### D2 — Action Specificity



Measures whether the support response contains an identifiable action or actionable guidance.



The classification includes:



\- generic or redirect

\- specific instruction

\- specific guidance

\- specific completed action



\### D3 — DM / Redirect Occurrence



Detects whether the support conversation redirects the customer to direct messages before a specific action is observed.



This is treated as a potential reduction in observable evidence.



\### D4 — Observable Outcome



Classifies the observable conversation outcome as:



\- positive reaction

\- negative reaction

\- neutral reaction

\- silence after action

\- no action observed



\## 7. Evidence Completeness Score



The project combines the four dimensions into a heuristic Evidence Completeness score:



`EC = 0.30\*D1 + 0.30\*D2 + 0.30\*D4\_value + 0.10\*(1-D3)`



D4 values:



| Outcome | Value |

|---|---:|

| Positive reaction | 1.00 |

| Silence after action | 0.70 |

| Neutral reaction | 0.55 |

| Negative reaction | 0.20 |

| No action observed | 0.00 |



Evidence tiers:



\- HIGH: EC >= 0.75

\- MEDIUM: 0.50 <= EC < 0.75

\- LOW: EC < 0.50



The EC score is a heuristic research feature. It is not a probability and has not been human-validated in this experiment.



\## 8. Evidence Statistics



Across the 78,781 retained conversations:



| Metric | Value |

|---|---:|

| Average D1 | 0.7142 |

| Average D2 | 0.3448 |

| Average D3 | 0.0207 |

| Average EC | 0.4851 |



\### D2 distribution



| Classification | Count | Percentage |

|---|---:|---:|

| Generic / redirect | 27,904 | 35.42% |

| Specific instruction | 36,335 | 46.12% |

| Specific guidance | 11,785 | 14.96% |

| Specific action | 2,757 | 3.50% |



\### D4 distribution



| Outcome | Count | Percentage |

|---|---:|---:|

| No action observed | 52,143 | 66.19% |

| Positive reaction | 10,131 | 12.86% |

| Silence after action | 9,600 | 12.19% |

| Negative reaction | 6,907 | 8.77% |



\### Evidence tier distribution



| Tier | Count | Percentage |

|---|---:|---:|

| HIGH | 10,147 | 12.88% |

| MEDIUM | 23,401 | 29.70% |

| LOW | 45,233 | 57.42% |



\## 9. Retrieval Corpus



The retrieval corpus contains all 78,781 retained conversations.



The corpus stores:



\- customer query

\- selected agent response

\- complete conversation text

\- turn count

\- D1–D4 evidence features

\- Evidence Completeness score

\- evidence tier

\- selected action metadata



Additional derived statistics:



| Metric | Count |

|---|---:|

| Responses with explicit action | 13,653 |

| DM-only responses | 444 |

| Generic-only responses | 17,144 |



\## 10. Semantic Retrieval



Customer queries are embedded using:



`sentence-transformers/all-MiniLM-L6-v2`



Embedding dimension:



`384`



Embeddings are normalized and stored in a FAISS inner-product index.



Files:



\- `output/semantic\_index.faiss`

\- `output/semantic\_index\_metadata.pkl`



The existing index contains exactly 78,781 vectors and matches the current retrieval corpus ordering.



The index should not be rebuilt unless the corpus ordering or embedded query text changes.



\## 11. Derived Dataset Files



The pipeline produces:



\### Conversation dataset



`output/amazonhelp\_conversations.jsonl`



Contains reconstructed AmazonHelp conversations.



\### Evidence features



`output/amazonhelp\_evidence\_features.jsonl`



Contains D1–D4 and Evidence Completeness measurements.



\### Evidence-aware retrieval dataset



`output/ec\_retrieval\_dataset.jsonl`



Combines reconstructed conversations with evidence features.



\### Retrieval corpus



`output/retrieval\_corpus.jsonl`



Contains the records used by the retrieval and decision-engine stages.



\### Semantic index



`output/semantic\_index.faiss`



FAISS vector index for semantic retrieval.



\### Index metadata



`output/semantic\_index\_metadata.pkl`



Stores model and corpus metadata associated with the FAISS index.



\## 12. Data Integrity Checks



The current pipeline verifies:



\- raw tweet IDs are unique

\- reconstructed conversations contain usable customer messages

\- retained evidence records have matching conversations

\- retrieval corpus contains 78,781 records

\- FAISS index contains 78,781 vectors

\- FAISS metadata ordering matches the retrieval corpus

\- no conversation records are missing during EC dataset construction



\## 13. Dataset Limitations



Several limitations should be considered.



\### Public support interactions are incomplete



Twitter conversations may move into private messages, meaning the publicly observable conversation may not contain the final resolution.



\### Evidence is observational



The absence of an observable outcome does not necessarily mean that the customer's issue remained unresolved.



\### Heuristic labels



D1–D4 are deterministic heuristic features rather than manually validated ground-truth labels.



\### Single support account



The current experiment focuses on AmazonHelp rather than evaluating multiple support brands.



\### Historical language



The dataset contains historical support interactions. Current customer-support behavior may differ.



\### Retrieval evaluation



The current evaluation uses manually designed queries and does not contain a human-annotated relevance benchmark.



Therefore, retrieval results should be interpreted as an engineering experiment rather than a definitive measurement of production accuracy.



\## 14. Reproducibility



The main processing stages are:



```text

twcs.csv

&#x20;   |

&#x20;   v

Conversation reconstruction

&#x20;   |

&#x20;   v

amazonhelp\_conversations.jsonl

&#x20;   |

&#x20;   v

Evidence extraction

&#x20;   |

&#x20;   v

amazonhelp\_evidence\_features.jsonl

&#x20;   |

&#x20;   v

EC dataset construction

&#x20;   |

&#x20;   v

ec\_retrieval\_dataset.jsonl

&#x20;   |

&#x20;   v

Retrieval corpus preparation

&#x20;   |

&#x20;   v

retrieval\_corpus.jsonl

&#x20;   |

&#x20;   v

Semantic embedding + FAISS

&#x20;   |

&#x20;   v

semantic\_index.faiss

