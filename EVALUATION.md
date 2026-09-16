\# Evaluation



\## 1. Evaluation Objective



This project evaluates whether incorporating \*\*Evidence Completeness (EC)\*\* into historical customer-support retrieval can select responses with stronger observable evidence than semantic-similarity-only retrieval.



The central research question is:



> \*\*Does incorporating observable evidence completeness into historical conversation retrieval reduce the reliance on similarity alone when selecting a response that can safely ground a customer-support answer?\*\*



The evaluation compares four retrieval strategies:



1\. \*\*semantic\_only\*\* — ranks candidates using semantic similarity only.

2\. \*\*semantic\_intent\*\* — combines semantic similarity with intent compatibility.

3\. \*\*semantic\_ec\*\* — combines semantic similarity with Evidence Completeness.

4\. \*\*full\*\* — combines semantic similarity, intent compatibility, and Evidence Completeness.



The purpose of this experiment is to study the retrieval behavior of the proposed approach, not to claim production-level accuracy.



\---



\## 2. Dataset Used



The evaluation uses the reconstructed \*\*AmazonHelp\*\* subset of the Twitter Customer Support (TWCS) dataset.



\### Corpus



\- Raw TWCS rows: \*\*2,811,774\*\*

\- AmazonHelp-containing graph components: \*\*82,556\*\*

\- Final retained AmazonHelp conversations: \*\*78,781\*\*

\- Conversations with 4+ turns: \*\*36,509\*\*

\- Average conversation length: \*\*4.29 turns\*\*



The final retrieval corpus contains one customer query and a selected historical agent response for each retained conversation.



The raw dataset is not modified by the evaluation process.



\---



\## 3. Evidence Completeness



Each historical conversation is assigned an Evidence Completeness score using four observable dimensions:



\### D1 — Context Specificity



Measures how specifically the customer describes the issue.



Examples of stronger context include:



\- concrete product or order references

\- domain-specific issue terms

\- numerical details

\- specific problem descriptions



\### D2 — Action Specificity



Measures how specific and actionable the agent response is.



The implementation distinguishes between:



\- generic or redirect responses

\- specific instructions

\- specific guidance

\- observable completed actions



\### D3 — DM / Redirect Occurrence



Identifies whether the conversation redirects the customer to a private/direct-message channel before a specific action is observed.



This matters because a redirect alone does not provide observable evidence that the underlying issue was resolved.



\### D4 — Observable Outcome



Classifies the observable outcome after the agent response:



\- `positive\_reaction`

\- `silence\_after\_action`

\- `neutral\_reaction`

\- `negative\_reaction`

\- `no\_action\_observed`



\### EC Formula



The current experimental Evidence Completeness score is:



```text

EC =

&#x20;   0.30 × D1

&#x20; + 0.30 × D2

&#x20; + 0.30 × D4\_value

&#x20; + 0.10 × (1 − D3)

