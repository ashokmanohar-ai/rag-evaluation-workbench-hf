---
title: RAG Evaluation Workbench
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Browser-native RAG retrieval evaluation workbench.
---

# RAG Evaluation Workbench

A browser-native RAG quality engineering workbench by **Ashok Kumar Manohar**.

It evaluates the retrieval layer independently from answer generation using labelled relevance sets and standard information-retrieval metrics.

- True BM25 (`k1`, `b`, IDF and length normalization)
- Local semantic embeddings in the browser with Hugging Face Transformers.js
- Hybrid lexical + semantic retrieval
- Precision@K, Recall@K, MRR, nDCG@K and Hit Rate
- Per-query ranked evidence
- Downloadable evidence JSON with SHA-256 provenance
- No LLM API key and no backend required

Source: https://github.com/ashokmanohar-ai/rag-evaluation-workbench-hf
