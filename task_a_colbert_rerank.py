from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


COLBERT_MODEL_NAME_DEFAULT = "jinaai/jina-colbert-v2"
COLBERT_QUERY_PREFIX_DEFAULT = "[QueryMarker]"
COLBERT_DOCUMENT_PREFIX_DEFAULT = "[DocumentMarker]"

_MODEL_CACHE: Dict[Tuple[str, str, int, int], object] = {}


def _load_pylate():
    try:
        from pylate import models, rank
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise RuntimeError(
            "PyLate is required for ColBERT reranking. Install it in the notebook environment with `pip install pylate`."
        ) from exc
    return models, rank


def get_colbert_model(
    model_name: str = COLBERT_MODEL_NAME_DEFAULT,
    device: str = "cpu",
    query_length: int = 128,
    document_length: int = 220,
):
    models, _ = _load_pylate()
    cache_key = (str(model_name), str(device), int(query_length), int(document_length))
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = models.ColBERT(
            model_name_or_path=model_name,
            query_prefix=COLBERT_QUERY_PREFIX_DEFAULT,
            document_prefix=COLBERT_DOCUMENT_PREFIX_DEFAULT,
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
            device=device,
            query_length=query_length,
            document_length=document_length,
        )
    return _MODEL_CACHE[cache_key]


def build_colbert_rerank_run(
    corpus: pd.DataFrame,
    topics: pd.DataFrame,
    primary_candidates: pd.DataFrame,
    final_k: int,
    model_name: str = COLBERT_MODEL_NAME_DEFAULT,
    device: str = "cpu",
    query_length: int = 128,
    document_length: int = 220,
    query_batch_size: int = 4,
    document_batch_size: int = 2,
    show_progress_bar: bool = False,
) -> pd.DataFrame:
    _, rank = _load_pylate()
    model = get_colbert_model(
        model_name=model_name,
        device=device,
        query_length=query_length,
        document_length=document_length,
    )

    corpus_text = {
        str(row.docno): str(row.text)
        for row in corpus[["docno", "text"]].itertuples(index=False)
    }
    topic_text = {
        str(row.qid): str(row.query)
        for row in topics[["qid", "query"]].itertuples(index=False)
    }

    qids: List[str] = []
    query_texts: List[str] = []
    documents_ids: List[List[str]] = []
    documents_texts: List[List[str]] = []

    for qid, group in primary_candidates.groupby("qid", sort=False):
        qid = str(qid)
        ranked_group = group.sort_values("rank")
        doc_ids = ranked_group["docno"].astype(str).tolist()
        doc_texts = [corpus_text[docno] for docno in doc_ids]
        qids.append(qid)
        query_texts.append(topic_text[qid])
        documents_ids.append(doc_ids)
        documents_texts.append(doc_texts)

    query_embeddings = model.encode(
        query_texts,
        is_query=True,
        batch_size=query_batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=False,
    )
    document_embeddings = model.encode(
        documents_texts,
        is_query=False,
        batch_size=document_batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=False,
    )

    reranked = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=query_embeddings,
        documents_embeddings=document_embeddings,
        device=device,
    )

    rows = []
    for qid, query_results in zip(qids, reranked):
        for rank_idx, item in enumerate(query_results[:final_k], start=1):
            rows.append(
                {
                    "qid": str(qid),
                    "docno": str(item["id"]),
                    "score": float(item["score"]),
                    "rank": int(rank_idx),
                }
            )

    return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])


def chunk_document_text(
    text: str,
    chunk_size_words: int,
    chunk_stride_words: int,
    max_chunks_per_doc: Optional[int] = None,
) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]

    chunk_size_words = max(int(chunk_size_words), 1)
    chunk_stride_words = max(int(chunk_stride_words), 1)
    max_chunks = None if max_chunks_per_doc is None else max(int(max_chunks_per_doc), 1)

    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if len(chunk_words) < chunk_size_words:
            break
        start += chunk_stride_words
        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    return chunks or [" ".join(words)]


def build_colbert_chunked_rerank_run(
    corpus: pd.DataFrame,
    topics: pd.DataFrame,
    primary_candidates: pd.DataFrame,
    final_k: int,
    model_name: str = COLBERT_MODEL_NAME_DEFAULT,
    device: str = "cpu",
    query_length: int = 128,
    document_length: int = 220,
    query_batch_size: int = 4,
    document_batch_size: int = 2,
    chunk_size_words: int = 160,
    chunk_stride_words: int = 120,
    max_chunks_per_doc: Optional[int] = 5,
    show_progress_bar: bool = False,
) -> pd.DataFrame:
    _, rank = _load_pylate()
    model = get_colbert_model(
        model_name=model_name,
        device=device,
        query_length=query_length,
        document_length=document_length,
    )

    corpus_chunks = {
        str(row.docno): chunk_document_text(
            text=str(row.text),
            chunk_size_words=chunk_size_words,
            chunk_stride_words=chunk_stride_words,
            max_chunks_per_doc=max_chunks_per_doc,
        )
        for row in corpus[["docno", "text"]].itertuples(index=False)
    }
    topic_text = {
        str(row.qid): str(row.query)
        for row in topics[["qid", "query"]].itertuples(index=False)
    }

    qids: List[str] = []
    query_texts: List[str] = []
    documents_ids: List[List[str]] = []
    documents_texts: List[List[str]] = []
    chunk_to_docno_per_qid: List[Dict[str, str]] = []

    for qid, group in primary_candidates.groupby("qid", sort=False):
        qid = str(qid)
        ranked_group = group.sort_values("rank")
        chunk_ids: List[str] = []
        chunk_texts: List[str] = []
        chunk_to_docno: Dict[str, str] = {}

        for docno in ranked_group["docno"].astype(str).tolist():
            chunks = corpus_chunks.get(docno, [""])
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{docno}__chunk_{chunk_idx}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_to_docno[chunk_id] = docno

        qids.append(qid)
        query_texts.append(topic_text[qid])
        documents_ids.append(chunk_ids)
        documents_texts.append(chunk_texts)
        chunk_to_docno_per_qid.append(chunk_to_docno)

    query_embeddings = model.encode(
        query_texts,
        is_query=True,
        batch_size=query_batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=False,
    )
    document_embeddings = model.encode(
        documents_texts,
        is_query=False,
        batch_size=document_batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=False,
    )

    reranked = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=query_embeddings,
        documents_embeddings=document_embeddings,
        device=device,
    )

    rows = []
    for qid, query_results, chunk_map in zip(qids, reranked, chunk_to_docno_per_qid):
        best_doc_scores: Dict[str, float] = {}
        for item in query_results:
            chunk_id = str(item["id"])
            docno = chunk_map.get(chunk_id)
            if docno is None:
                continue
            score = float(item["score"])
            if docno not in best_doc_scores or score > best_doc_scores[docno]:
                best_doc_scores[docno] = score

        ranked_docs = sorted(best_doc_scores.items(), key=lambda item: item[1], reverse=True)[:final_k]
        for rank_idx, (docno, score) in enumerate(ranked_docs, start=1):
            rows.append(
                {
                    "qid": str(qid),
                    "docno": str(docno),
                    "score": float(score),
                    "rank": int(rank_idx),
                }
            )

    return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])


def splice_reranked_candidates(
    primary_full_run: pd.DataFrame,
    reranked_candidates: pd.DataFrame,
    final_k: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    primary_full_run = primary_full_run.copy()
    primary_full_run["qid"] = primary_full_run["qid"].astype(str)
    primary_full_run["docno"] = primary_full_run["docno"].astype(str)
    reranked_candidates = reranked_candidates.copy()
    reranked_candidates["qid"] = reranked_candidates["qid"].astype(str)
    reranked_candidates["docno"] = reranked_candidates["docno"].astype(str)

    for qid, primary_group in primary_full_run.groupby("qid", sort=False):
        primary_group = primary_group.sort_values("rank")
        reranked_group = reranked_candidates[
            reranked_candidates["qid"] == str(qid)
        ].sort_values("rank")

        seen = set()
        merged_items = []

        for row in reranked_group.itertuples(index=False):
            docno = str(row.docno)
            merged_items.append((docno, float(row.score)))
            seen.add(docno)

        for row in primary_group.itertuples(index=False):
            docno = str(row.docno)
            if docno in seen:
                continue
            merged_items.append((docno, float(row.score)))

        if final_k is not None:
            merged_items = merged_items[: int(final_k)]

        for rank_idx, (docno, score) in enumerate(merged_items, start=1):
            rows.append(
                {
                    "qid": str(qid),
                    "docno": docno,
                    "score": float(score),
                    "rank": int(rank_idx),
                }
            )

    return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])
