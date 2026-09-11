from __future__ import annotations

import csv
import io
import json
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from rag_core import (
    BM25Index,
    chunk_documents,
    evidence_hash,
    make_run_id,
    metrics_for_query,
    minmax,
    rank_chunks,
    unique_doc_ranking,
)

DEFAULT_CORPUS = """doc_id,text
FHIR-01,"A FHIR CapabilityStatement describes server capabilities including FHIR version, supported resources, interactions, formats, search parameters and operations. Quality engineering can compare declared capability with live behavior."
FHIR-02,"FHIR resource validation can use the $validate operation. Servers return an OperationOutcome containing issues with severity, code, diagnostics and locations. Terminology validation can use $validate-code against a CodeSystem or ValueSet."
MCP-01,"Enterprise MCP governance should classify tool actions and enforce outcomes such as ALLOW, APPROVAL, BLOCK, RETRY and ESCALATE. Destructive or cross-tenant operations should be blocked or require explicit approval."
AGENT-01,"Agent evaluation should measure task success, tool correctness, safety, latency and flakiness over repeated trials. Hard release gates prevent unsafe or unreliable agent versions from reaching production."
RAG-01,"Retrieval evaluation measures whether relevant evidence appears in the ranked result set. Common information retrieval metrics include Precision at K, Recall at K, Hit Rate, Mean Reciprocal Rank and normalized Discounted Cumulative Gain."
RAG-02,"BM25 is a lexical ranking function based on term frequency, inverse document frequency and document length normalization. The k1 parameter controls term saturation while b controls length normalization."
RAG-03,"Semantic retrieval encodes queries and passages into vectors and ranks by cosine similarity. Hybrid retrieval combines lexical BM25 evidence with semantic similarity to improve robustness across exact terminology and paraphrases."
AIQX-01,"AIQX aggregates evidence from agent evaluation, AgentOps, RAGOps and healthcare interoperability checks. A weighted release gate can produce PASS, CONDITIONAL GO or HOLD while retaining source provenance, timestamps and evidence hashes."
FDE-01,"A forward deployed engineer translates customer requirements into architecture alternatives, trade-offs, risks, assumptions, proof-of-concept exit criteria and an architecture decision record."
VERCEL-01,"A production delivery path can use source control, pull requests, preview deployments, automated checks and production promotion. Observability should include latency, errors, deployment health and evidence provenance."
"""

DEFAULT_EVAL = """query,relevant_doc_ids
How do I verify what a FHIR server supports?,FHIR-01
What operation returns validation issues for a Patient resource?,FHIR-02
Which MCP outcome should protect destructive actions?,MCP-01
How should repeated AI agent trials be evaluated?,AGENT-01
Which metrics show ranked retrieval quality?,RAG-01
What do k1 and b mean in BM25?,RAG-02
How does hybrid retrieval combine exact terms and semantic similarity?,RAG-02|RAG-03
How can multiple AI quality signals become a release decision?,AIQX-01
What should an FDE produce from a customer requirement?,FDE-01
How should a production deployment workflow be structured?,VERCEL-01
"""


def parse_csv_text(text: str, required: set[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    return [dict(r) for r in reader if any((v or '').strip() for v in r.values())]


def load_csv_source(file_path, text: str, required: set[str]) -> list[dict]:
    if file_path:
        path = Path(file_path)
        return parse_csv_text(path.read_text(encoding="utf-8-sig"), required)
    return parse_csv_text(text, required)


@lru_cache(maxsize=3)
def get_encoder(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


def semantic_scores(model_id: str, chunks: list[dict], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model = get_encoder(model_id)
    chunk_texts = [c["text"] for c in chunks]
    c_emb = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(q_emb), np.asarray(c_emb)


def metric_cards(agg: dict, retriever: str, run_id: str, evidence_sha: str) -> str:
    def pct(x): return f"{100*x:.1f}%"
    return f"""
    <div class='metric-grid'>
      <div class='metric'><span>Recall@K</span><b>{pct(agg['recall_at_k'])}</b></div>
      <div class='metric'><span>Precision@K</span><b>{pct(agg['precision_at_k'])}</b></div>
      <div class='metric'><span>MRR</span><b>{agg['mrr']:.3f}</b></div>
      <div class='metric'><span>nDCG@K</span><b>{agg['ndcg_at_k']:.3f}</b></div>
      <div class='metric'><span>Hit Rate</span><b>{pct(agg['hit_rate'])}</b></div>
      <div class='metric'><span>P50 latency</span><b>{agg['p50_latency_ms']:.1f} ms</b></div>
    </div>
    <div class='run-meta'><b>{retriever}</b> · Run <code>{run_id}</code> · Evidence <code>{evidence_sha[:16]}</code></div>
    """


def run_evaluation(corpus_file, corpus_text, eval_file, eval_text, retriever, top_k, chunk_words,
                   overlap_words, k1, b, semantic_model, semantic_weight):
    try:
        docs = load_csv_source(corpus_file, corpus_text, {"doc_id", "text"})
        eval_rows = load_csv_source(eval_file, eval_text, {"query", "relevant_doc_ids"})
        chunks = chunk_documents(docs, int(chunk_words), int(overlap_words))
        if not chunks:
            raise ValueError("No chunks were produced from the corpus.")
        if not eval_rows:
            raise ValueError("Evaluation set is empty.")

        bm25 = BM25Index(chunks, float(k1), float(b))
        queries = [r["query"].strip() for r in eval_rows]
        needs_semantic = retriever in {"Semantic (MiniLM)", "Hybrid BM25 + Semantic"}
        if needs_semantic:
            q_emb, c_emb = semantic_scores(semantic_model.strip(), chunks, queries)
        else:
            q_emb = c_emb = None

        per_query = []
        evidence_rows = []
        latencies = []

        for qi, row in enumerate(eval_rows):
            query = row["query"].strip()
            relevant = [x.strip() for x in row["relevant_doc_ids"].replace(";", "|").split("|") if x.strip()]
            started = time.perf_counter()
            bm = bm25.score(query)
            if retriever == "BM25":
                scores = bm
            else:
                sem = np.dot(c_emb, q_emb[qi])
                if retriever == "Semantic (MiniLM)":
                    scores = sem
                else:
                    scores = (1.0 - float(semantic_weight)) * minmax(bm) + float(semantic_weight) * minmax(sem)
            ranked = rank_chunks(chunks, scores)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            doc_rank = unique_doc_ranking(ranked)
            m = metrics_for_query(doc_rank, relevant, int(top_k))
            top_docs = doc_rank[:int(top_k)]
            per_query.append({
                "query": query,
                "relevant_doc_ids": " | ".join(relevant),
                "top_docs": " → ".join(top_docs),
                f"precision@{int(top_k)}": round(m["precision_at_k"], 4),
                f"recall@{int(top_k)}": round(m["recall_at_k"], 4),
                "RR": round(m["rr"], 4),
                f"nDCG@{int(top_k)}": round(m["ndcg_at_k"], 4),
                "hit": int(m["hit_rate"]),
                "latency_ms": round(elapsed, 2),
            })
            for rank, hit in enumerate(ranked[:max(int(top_k), 5)], start=1):
                evidence_rows.append({
                    "query": query,
                    "rank": rank,
                    "doc_id": hit["doc_id"],
                    "chunk_id": hit["chunk_id"],
                    "score": round(hit["score"], 6),
                    "relevant": "YES" if hit["doc_id"] in set(relevant) else "NO",
                    "text": hit["text"],
                })

        pq = pd.DataFrame(per_query)
        p_col = f"precision@{int(top_k)}"
        r_col = f"recall@{int(top_k)}"
        n_col = f"nDCG@{int(top_k)}"
        agg = {
            "precision_at_k": float(pq[p_col].mean()),
            "recall_at_k": float(pq[r_col].mean()),
            "mrr": float(pq["RR"].mean()),
            "ndcg_at_k": float(pq[n_col].mean()),
            "hit_rate": float(pq["hit"].mean()),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "queries": len(eval_rows),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        run_id = make_run_id()
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "retriever": retriever,
            "configuration": {
                "top_k": int(top_k), "chunk_words": int(chunk_words), "overlap_words": int(overlap_words),
                "bm25_k1": float(k1), "bm25_b": float(b), "semantic_model": semantic_model,
                "semantic_weight": float(semantic_weight),
            },
            "aggregate": agg,
            "per_query": per_query,
        }
        sha = evidence_hash(payload)
        payload["evidence_sha256"] = sha
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="rag-eval-", delete=False, encoding="utf-8")
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.close()

        aggregate_df = pd.DataFrame([
            ["Precision@K", round(agg["precision_at_k"], 4)],
            ["Recall@K", round(agg["recall_at_k"], 4)],
            ["MRR", round(agg["mrr"], 4)],
            ["nDCG@K", round(agg["ndcg_at_k"], 4)],
            ["Hit Rate", round(agg["hit_rate"], 4)],
            ["P50 latency (ms)", round(agg["p50_latency_ms"], 2)],
            ["P95 latency (ms)", round(agg["p95_latency_ms"], 2)],
            ["Documents", agg["documents"]], ["Chunks", agg["chunks"]], ["Queries", agg["queries"]],
        ], columns=["Metric", "Value"])

        return metric_cards(agg, retriever, run_id, sha), aggregate_df, pq, pd.DataFrame(evidence_rows), tmp.name
    except Exception as e:
        raise gr.Error(str(e))


CSS = """
.gradio-container {max-width: 1500px !important;}
.hero {background: linear-gradient(135deg,#07111f,#0d2440); color:white; padding:28px; border-radius:22px; margin-bottom:18px;}
.hero h1 {font-size: 34px; margin:0 0 8px 0;}
.hero p {color:#bfd2e6; max-width:920px; margin:0; line-height:1.6;}
.badge {display:inline-block;padding:5px 9px;border-radius:999px;background:#0e7490;color:#ecfeff;font-size:12px;font-weight:700;margin-bottom:10px;}
.metric-grid {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:10px 0;}
.metric {border:1px solid #dbe4ee;border-radius:14px;padding:14px;background:white;box-shadow:0 6px 20px #0f172a0a;}
.metric span {display:block;color:#64748b;font-size:12px}.metric b{display:block;font-size:24px;margin-top:5px;color:#0f172a}
.run-meta{margin:10px 0 2px;color:#475569;font-size:12px}.run-meta code{background:#eef2ff;padding:2px 5px;border-radius:5px}
@media(max-width:900px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
"""

with gr.Blocks(title="RAG Evaluation Workbench") as demo:
    gr.HTML("""
    <div class='hero'>
      <span class='badge'>REPRODUCIBLE RAG QUALITY ENGINEERING</span>
      <h1>RAG Evaluation Workbench</h1>
      <p>Benchmark the retrieval layer before trusting generation. Compare true BM25, local semantic retrieval and hybrid search against a labelled relevance set using Precision@K, Recall@K, MRR, nDCG and Hit Rate — no API keys required.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Run Evaluation"):
            with gr.Row():
                with gr.Column(scale=3):
                    corpus_file = gr.File(label="Corpus CSV (optional)", file_types=[".csv"], type="filepath")
                    corpus_text = gr.Textbox(value=DEFAULT_CORPUS, lines=12, label="Corpus CSV", info="Required columns: doc_id,text. Uploaded file takes precedence.")
                with gr.Column(scale=2):
                    eval_file = gr.File(label="Evaluation CSV (optional)", file_types=[".csv"], type="filepath")
                    eval_text = gr.Textbox(value=DEFAULT_EVAL, lines=12, label="Labelled Evaluation Set", info="Required columns: query,relevant_doc_ids. Use | for multiple relevant documents.")

            with gr.Row():
                retriever = gr.Radio(["BM25", "Semantic (MiniLM)", "Hybrid BM25 + Semantic"], value="BM25", label="Retriever")
                top_k = gr.Slider(1, 10, 3, step=1, label="Top K")
                chunk_words = gr.Slider(20, 250, 80, step=10, label="Chunk size (words)")
                overlap_words = gr.Slider(0, 60, 15, step=5, label="Overlap (words)")
            with gr.Accordion("Advanced retrieval controls", open=False):
                with gr.Row():
                    k1 = gr.Slider(0.5, 3.0, 1.5, step=0.1, label="BM25 k1")
                    b = gr.Slider(0.0, 1.0, 0.75, step=0.05, label="BM25 b")
                    semantic_weight = gr.Slider(0.0, 1.0, 0.55, step=0.05, label="Hybrid semantic weight")
                semantic_model = gr.Textbox(value="sentence-transformers/all-MiniLM-L6-v2", label="Semantic embedding model")

            run_btn = gr.Button("Run Retrieval Benchmark", variant="primary")
            cards = gr.HTML()
            with gr.Row():
                aggregate = gr.Dataframe(label="Aggregate Metrics", interactive=False)
                export_file = gr.File(label="Download Evidence JSON")

            per_query = gr.Dataframe(label="Per-query Metrics", interactive=False, wrap=True)
            evidence = gr.Dataframe(label="Ranked Retrieval Evidence", interactive=False, wrap=True)

            run_btn.click(
                run_evaluation,
                inputs=[corpus_file, corpus_text, eval_file, eval_text, retriever, top_k, chunk_words,
                        overlap_words, k1, b, semantic_model, semantic_weight],
                outputs=[cards, aggregate, per_query, evidence, export_file],
                api_name="evaluate_retrieval",
            )

        with gr.Tab("Metric Guide"):
            gr.Markdown("""
### What the workbench measures

| Metric | What it answers |
|---|---|
| **Precision@K** | Of the top-K retrieved documents, how many are actually relevant? |
| **Recall@K** | Of all known relevant documents, how many did retrieval find in the top K? |
| **MRR** | How early does the first relevant document appear? |
| **nDCG@K** | How good is the ranked ordering, with stronger reward for relevant results near the top? |
| **Hit Rate** | Did the retriever find at least one relevant document in the top K? |
| **P50/P95 latency** | What retrieval latency should an engineering team expect across evaluation queries? |

**Important:** this Space evaluates **retrieval**, not answer generation. That separation is intentional: weak retrieval cannot be rescued reliably by a generator, and retrieval metrics remain reproducible without paid LLM APIs.
            """)

        with gr.Tab("Architecture"):
            gr.Markdown("""
```text
Labelled corpus + relevance set
          │
          ▼
   Chunking configuration
          │
    ┌─────┼───────────┐
    ▼     ▼           ▼
  BM25  Semantic    Hybrid
    │     │           │
    └─────┼───────────┘
          ▼
 Ranked document evidence
          │
          ▼
Precision@K · Recall@K · MRR · nDCG · Hit Rate · latency
          │
          ▼
Evidence JSON + SHA-256 provenance
```

**Design principles**
- No external LLM key required.
- BM25 is implemented with term frequency, IDF, k1 and b length normalization.
- Semantic retrieval loads a Hugging Face sentence-transformer lazily.
- Hybrid retrieval combines min-max-normalized BM25 and cosine similarity scores.
- Evaluation is document-level even when retrieval is chunk-level, preventing duplicate chunks from inflating metrics.
- Every run emits an evidence artifact containing configuration, metrics, per-query results and a SHA-256 evidence hash.
            """)

        with gr.Tab("Recruiter / Interview Walkthrough"):
            gr.Markdown("""
### 30-second explanation
> I built this workbench to test the retrieval layer of a RAG system independently from the LLM. A labelled query-to-document relevance set lets me compare BM25, semantic and hybrid retrieval using standard IR metrics, then export reproducible evidence for release decisions.

### What to demonstrate
1. Run the default **BM25** benchmark.
2. Change `k1`, `b`, chunk size or Top-K and show metrics move.
3. Switch to **Semantic** or **Hybrid** to demonstrate local embedding retrieval.
4. Inspect per-query failures instead of relying only on averages.
5. Download the evidence JSON and show the run ID + SHA-256 provenance.

### Production extension
For an enterprise system I would add versioned datasets, model/embedding registry, reranker evaluation, regression baselines, confidence intervals, experiment persistence, CI release gates and live observability.
            """)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"), css=CSS)
