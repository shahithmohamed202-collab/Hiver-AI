\# Evidence-Completeness-Aware Customer Support RAG



> \*\*Evidence before answers.\*\*



An evidence-aware retrieval system for customer-support conversations that goes beyond semantic similarity by considering whether a retrieved historical conversation contains enough \*\*observable evidence\*\* to safely support a response.



Built for the Hiver SDE Intern Take-Home Assignment using the \*\*Twitter Customer Support Conversation (TWCS)\*\* dataset and the AmazonHelp support account.



\---



\## 1. The Problem



Traditional Retrieval-Augmented Generation (RAG) systems usually retrieve historical conversations based primarily on semantic similarity.



That creates an important problem in customer support:



> A conversation can be highly similar to the current customer query while still containing little observable evidence that the suggested resolution actually worked.



For example, a historical conversation may contain:



```text

Customer: My order hasn't arrived.

Agent: Please send us a DM.

