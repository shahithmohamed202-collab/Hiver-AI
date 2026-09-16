\# Architecture



\## 1. System Overview



This project implements an \*\*Evidence-Completeness-Aware Customer Support RAG\*\* system.



The core idea is:



> \*\*Evidence before answers.\*\*



Traditional retrieval-augmented generation systems often retrieve historical responses primarily according to semantic similarity. This project adds another question:



> \*\*Does the retrieved historical conversation contain enough observable evidence to safely ground a response?\*\*



The system therefore combines:



\- semantic similarity

\- intent compatibility

\- issue-state compatibility

\- Evidence Completeness (EC)



The resulting pipeline can retrieve historical support interactions and determine whether the evidence is strong enough to:



\- automatically respond

\- ask for clarification

\- escalate



\---



\# 2. High-Level Pipeline



```text

&#x20;                   TWCS Dataset

&#x20;                        |

&#x20;                        v

&#x20;             Conversation Reconstruction

&#x20;                        |

&#x20;                        v

&#x20;             AmazonHelp Conversation Set

&#x20;                        |

&#x20;                        v

&#x20;             Evidence Feature Extraction

&#x20;                        |

&#x20;                        v

&#x20;             Evidence Completeness Score

&#x20;                        |

&#x20;                        v

&#x20;               Retrieval Corpus

&#x20;                        |

&#x20;                        +------------------+

&#x20;                        |                  |

&#x20;                        v                  v

&#x20;                 Customer Query      Historical Queries

&#x20;                        |                  |

&#x20;                        v                  v

&#x20;                 Intent / State      FAISS Embeddings

&#x20;                        |                  |

&#x20;                        +--------+---------+

&#x20;                                 |

&#x20;                                 v

&#x20;                        Candidate Retrieval

&#x20;                                 |

&#x20;                                 v

&#x20;                   Semantic / Intent / EC

&#x20;                        Ranking Variants

&#x20;                                 |

&#x20;                                 v

&#x20;                        Final Candidate

&#x20;                                 |

&#x20;                                 v

&#x20;                      Decision Engine

&#x20;                                 |

&#x20;                  +--------------+--------------+

&#x20;                  |              |              |

&#x20;                  v              v              v

&#x20;            AUTO\_RESPOND   ASK\_CLARIFICATION  ESCALATE

