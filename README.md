---
title: RAG Evaluation Workbench
emoji: 🔎
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Reproducible BM25, semantic and hybrid RAG retrieval evaluation with IR metrics.
---

# RAG Evaluation Workbench

A recruiter-facing, keyless RAG quality engineering Space for comparing **BM25**, **local semantic retrieval**, and **hybrid retrieval** against a labelled relevance set.

## What it proves

- True BM25 with `k1`, `b`, IDF and document-length normalization
- Local semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid lexical + semantic ranking
- Chunk-level retrieval with document-level evaluation
- Precision@K, Recall@K, MRR, nDCG@K, Hit Rate
- P50/P95 retrieval latency
- Per-query failure analysis
- Downloadable evidence JSON with run ID and SHA-256 evidence hash
- No paid LLM API and no `.env` file required

## Expected input

### Corpus CSV

```csv
doc_id,text
D1,"Document text..."
D2,"Another document..."
```

### Labelled evaluation CSV

```csv
query,relevant_doc_ids
"What is BM25?",D2
"Which documents are relevant?",D1|D3
```

## Architecture

```text
Corpus + labelled relevance set
        ↓
Chunking
        ↓
BM25 / Semantic / Hybrid
        ↓
Ranked evidence
        ↓
Precision@K · Recall@K · MRR · nDCG · Hit Rate
        ↓
Evidence JSON + SHA-256 provenance
```

## Why this exists

RAG teams often evaluate only the generated answer. This project isolates retrieval quality first. That makes failures explainable and lets teams decide whether the problem is chunking, retrieval, ranking, relevance labels, or generation.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Hugging Face Space

The repository is deployed automatically to Hugging Face Spaces from GitHub Actions when `main` changes. The workflow expects a GitHub Actions secret named `HF_TOKEN` with write permission to the Hugging Face namespace `ashokrulz20`.

Target Space:

`ashokrulz20/rag-evaluation-workbench`

The default BM25 mode starts without downloading an embedding model. Semantic and Hybrid modes lazily load the configured sentence-transformer model.

## Portfolio positioning

**Ashok Kumar Manohar — Test Architect | AI Quality Engineer | Agentic AI | RAG Evaluation | FDE**

Use this Space as AI-native evidence alongside the Vercel RAGOps Studio and GitHub source repositories.
