from rag_core import BM25Index, chunk_documents, metrics_for_query, rank_chunks, unique_doc_ranking

docs = [
    {"doc_id": "D1", "text": "BM25 lexical retrieval uses term frequency and document length"},
    {"doc_id": "D2", "text": "semantic vector retrieval uses embeddings and cosine similarity"},
]
chunks = chunk_documents(docs, 20, 0)
idx = BM25Index(chunks, 1.5, 0.75)
ranked = rank_chunks(chunks, idx.score("BM25 lexical retrieval"))
docs_ranked = unique_doc_ranking(ranked)
assert docs_ranked[0] == "D1"
m = metrics_for_query(docs_ranked, ["D1"], 1)
assert m["recall_at_k"] == 1.0
assert m["precision_at_k"] == 1.0
assert m["rr"] == 1.0
print("core tests passed")
