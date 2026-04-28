from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import os
import re
import unicodedata
import warnings
import zipfile

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    import torch
    from sentence_transformers.util import batch_to_device
except Exception as exc:  # pragma: no cover - import-time guard for notebook use
    raise RuntimeError("Torch and sentence-transformers utilities are required for JobBERT.") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "var" / "artifacts" / "jobbert_dense"
SUBMISSION_ROOT = PROJECT_ROOT / "var" / "artifacts" / "jobbert_submission"

MODEL_NAME = "TechWolf/JobBERT-v3"
DEFAULT_TEAM_NAME = os.environ.get("TALENTCLEF_TEAM_NAME", "yourteam")
DEFAULT_APPROACH_ID = os.environ.get("TALENTCLEF_APPROACH_ID", "jobbert_dense_at50")
DEFAULT_K = 50

LANG_CONFIGS = {
    "en": {
        "development_dir": DATA_ROOT / "development" / "en",
        "test_dir": DATA_ROOT / "test" / "en",
        "submission_lang_pair": "en-en",
        "batch_size_doc": 16,
        "batch_size_query": 16,
    },
    "es": {
        "development_dir": DATA_ROOT / "development" / "es",
        "test_dir": DATA_ROOT / "test" / "es",
        "submission_lang_pair": "es-es",
        "batch_size_doc": 16,
        "batch_size_query": 16,
    },
}

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_tags_re = re.compile(r"<[^>]+>")
_ws_re = re.compile(r"\s+")

SECTION_PATTERNS = {
    "skills": [
        r"\bskills\b",
        r"\btechnical skills\b",
        r"\bcore competencies\b",
        r"\bcompetencies\b",
        r"\btechnologies\b",
        r"\btools\b",
        r"\bhabilidades\b",
        r"\bcompetencias\b",
        r"\baptitudes\b",
        r"\bhabilidades técnicas\b",
        r"\bconocimientos\b",
    ],
    "experience": [
        r"\bexperience\b",
        r"\bwork experience\b",
        r"\bprofessional experience\b",
        r"\bemployment history\b",
        r"\bcareer history\b",
        r"\bexperiencia\b",
        r"\bexperiencia laboral\b",
        r"\btrayectoria profesional\b",
        r"\bhistorial laboral\b",
    ],
    "education": [
        r"\beducation\b",
        r"\bacademic background\b",
        r"\bqualifications\b",
        r"\beducación\b",
        r"\bformación\b",
        r"\bformación académica\b",
        r"\bestudios\b",
    ],
    "certifications": [
        r"\bcertifications\b",
        r"\blicenses\b",
        r"\bcertificates\b",
        r"\bcertificaciones\b",
        r"\blicencias\b",
        r"\bcertificados\b",
    ],
    "languages": [
        r"\blanguages\b",
        r"\blanguage proficiency\b",
        r"\bidiomas\b",
        r"\blenguas\b",
    ],
    "profile": [
        r"\bprofile\b",
        r"\bsummary\b",
        r"\bprofessional summary\b",
        r"\babout me\b",
        r"\bobjective\b",
        r"\bperfil\b",
        r"\bresumen\b",
        r"\bperfil profesional\b",
        r"\bobjetivo\b",
        r"\bacerca de mí\b",
    ],
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _tags_re.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _ws_re.sub(" ", text).strip()
    return text


def safe_concat(parts: List[Optional[str]]) -> str:
    out: List[str] = []
    for part in parts:
        if part is None:
            continue
        part = str(part).strip()
        if not part or part in {"N/A", "nan"}:
            continue
        out.append(part)
    return "\n".join(out)


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_section(line: str) -> Optional[str]:
    line_norm = normalize_text(line)
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.fullmatch(pattern, line_norm):
                return section
    return None


def parse_cv_sections(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines()]
    sections: Dict[str, List[str]] = defaultdict(list)
    current = "other"

    for line in lines:
        if not line:
            continue
        section = detect_section(line)
        if section is not None:
            current = section
            continue
        sections[current].append(line)

    result = {key: "\n".join(values).strip() for key, values in sections.items()}
    for key in ["profile", "skills", "experience", "education", "certifications", "languages", "other"]:
        result.setdefault(key, "")
    result["tools"] = result["skills"]
    return result


def build_cv_retrieval_text(sections: Dict[str, str]) -> str:
    return safe_concat(
        [
            sections.get("profile", ""),
            sections.get("experience", ""),
            sections.get("skills", ""),
            sections.get("tools", ""),
            sections.get("education", ""),
            sections.get("certifications", ""),
            sections.get("languages", ""),
            sections.get("other", ""),
        ]
    )


def load_task_a_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corpus_dir = data_dir / "corpus"
    if not corpus_dir.exists():
        corpus_dir = data_dir / "corpus_elements"
    queries_dir = data_dir / "queries"

    corpus_rows = []
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file():
            continue
        docno = path.name
        raw = read_text_file(path)
        sections = parse_cv_sections(raw)
        corpus_rows.append(
            {
                "docno": str(docno),
                "raw_text": raw,
                "text": normalize_text(build_cv_retrieval_text(sections)),
            }
        )
    corpus = pd.DataFrame(corpus_rows)

    query_rows = []
    for path in sorted(queries_dir.iterdir()):
        if not path.is_file():
            continue
        qid = path.name
        raw = read_text_file(path)
        query_rows.append(
            {
                "qid": str(qid),
                "raw_query": raw,
                "query": normalize_text(raw),
            }
        )
    topics = pd.DataFrame(query_rows)

    return corpus, topics


def load_task_a_split(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus, topics = load_task_a_inputs(data_dir)
    qrels_path = data_dir / "qrels.tsv"
    qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "Q0", "docno", "score"])
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["score"] = qrels["score"].astype(float)
    return corpus, topics, qrels


def get_st_model(name: str) -> SentenceTransformer:
    if name not in _MODEL_CACHE:
        print("Loading model:", name)
        _MODEL_CACHE[name] = SentenceTransformer(name, tokenizer_kwargs={"use_fast": False})
    return _MODEL_CACHE[name]


def encode_jobbert_batch(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    features = model.tokenize(texts)
    features = batch_to_device(features, model.device)
    features["text_keys"] = ["anchor"]
    with torch.no_grad():
        out_features = model.forward(features)
    embeddings = out_features["sentence_embedding"].detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / norms


def encode_texts_jobbert(model_name: str, texts: List[str], batch_size: int = 16) -> np.ndarray:
    model = get_st_model(model_name)
    texts = [normalize_text(text) for text in texts]
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)

    order = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in order]
    parts: List[np.ndarray] = []

    for start in tqdm(range(0, len(sorted_texts), batch_size), desc=f"Encoding with {model_name}"):
        batch = sorted_texts[start : start + batch_size]
        parts.append(encode_jobbert_batch(model, batch))

    sorted_embeddings = np.concatenate(parts, axis=0)
    inverse_order = np.argsort(order)
    return sorted_embeddings[inverse_order].astype(np.float32)


def build_dense_run(
    corpus: pd.DataFrame,
    topics: pd.DataFrame,
    model_name: str,
    batch_size_doc: int,
    batch_size_query: int,
    k: int,
) -> pd.DataFrame:
    doc_embeddings = encode_texts_jobbert(model_name, corpus["text"].tolist(), batch_size=batch_size_doc)
    query_embeddings = encode_texts_jobbert(model_name, topics["query"].tolist(), batch_size=batch_size_query)

    docnos = corpus["docno"].astype(str).tolist()
    qids = topics["qid"].astype(str).tolist()
    rows = []

    for index, qid in enumerate(qids):
        scores = query_embeddings[index] @ doc_embeddings.T
        topk = min(k, len(scores))
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        for rank, doc_idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "qid": str(qid),
                    "docno": str(docnos[doc_idx]),
                    "score": float(scores[doc_idx]),
                    "rank": int(rank),
                }
            )

    run_df = pd.DataFrame(rows)
    return run_df.sort_values(["qid", "rank"], ascending=[True, True]).reset_index(drop=True)


def sanitize_token(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "", value)
    if not value:
        raise ValueError("A required identifier became empty after sanitization.")
    return value


def save_trec_run(run_df: pd.DataFrame, out_path: Path, tag: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trec = run_df.copy()
    trec["Q0"] = "Q0"
    trec["tag"] = str(tag)
    trec = trec[["qid", "Q0", "docno", "rank", "score", "tag"]]
    trec.to_csv(out_path, sep=" ", header=False, index=False)


def evaluate_run(run_df: pd.DataFrame, qrels_df: pd.DataFrame, k: int) -> Dict[str, float]:
    metrics_per_query: List[Tuple[float, float, float, float, float, float]] = []

    qrels_df = qrels_df.copy()
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["docno"] = qrels_df["docno"].astype(str)

    for qid, rel_group in qrels_df.groupby("qid", sort=False):
        qrels_map = {str(docno): float(score) for docno, score in zip(rel_group["docno"], rel_group["score"])}
        relevant = {docno for docno, score in qrels_map.items() if score > 0}
        if not relevant:
            continue

        ranked = (
            run_df.loc[run_df["qid"].astype(str) == str(qid)]
            .sort_values("rank")["docno"]
            .astype(str)
            .tolist()[:k]
        )

        hits = [1 if docno in relevant else 0 for docno in ranked]
        gains = [qrels_map.get(docno, 0.0) for docno in ranked]

        precision = float(sum(hits) / k) if k > 0 else 0.0
        recall = float(sum(hits) / len(relevant)) if relevant else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rr = 0.0
        for index, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / index
                break

        ap_hits = 0
        ap_sum = 0.0
        for index, hit in enumerate(hits, start=1):
            if hit:
                ap_hits += 1
                ap_sum += ap_hits / index
        ap = ap_sum / len(relevant) if relevant else 0.0

        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_gains = sorted(qrels_map.values(), reverse=True)[:k]
        idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics_per_query.append((ndcg, rr, precision, recall, f1, ap))

    if not metrics_per_query:
        return {
            f"nDCG@{k}": 0.0,
            f"MRR@{k}": 0.0,
            f"Prec@{k}": 0.0,
            f"Recall@{k}": 0.0,
            f"F1@{k}": 0.0,
            f"MAP@{k}": 0.0,
        }

    arr = np.asarray(metrics_per_query, dtype=np.float32)
    return {
        f"nDCG@{k}": float(arr[:, 0].mean()),
        f"MRR@{k}": float(arr[:, 1].mean()),
        f"Prec@{k}": float(arr[:, 2].mean()),
        f"Recall@{k}": float(arr[:, 3].mean()),
        f"F1@{k}": float(arr[:, 4].mean()),
        f"MAP@{k}": float(arr[:, 5].mean()),
    }


def zip_submission(run_paths: List[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_paths:
            archive.write(path, arcname=path.name)
    return out_path


def run_development_split(lang: str, cfg: dict, team_name: str, approach_id: str, k: int) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"Running JobBERT development pipeline for language: {lang}")
    print("=" * 100)

    corpus, topics, qrels = load_task_a_split(cfg["development_dir"])
    print("development corpus:", corpus.shape)
    print("development topics:", topics.shape)
    print("development qrels:", qrels.shape)

    run_df = build_dense_run(
        corpus=corpus,
        topics=topics,
        model_name=MODEL_NAME,
        batch_size_doc=cfg["batch_size_doc"],
        batch_size_query=cfg["batch_size_query"],
        k=k,
    )
    metrics = evaluate_run(run_df, qrels, k=k)

    dev_run_path = RUNS_ROOT / f"run_dev_{lang}_jobbert_dense_at{k}.trec"
    save_trec_run(run_df, dev_run_path, tag=f"{lang}_jobbert_dense_at{k}")

    row: Dict[str, object] = {
        "lang": lang,
        "split": "development",
        "submission_lang_pair": cfg["submission_lang_pair"],
        "pipeline": f"dense(JobBERT)@{k}",
        "model": MODEL_NAME,
        "queries": int(len(topics)),
        "docs": int(len(corpus)),
        "run_path": str(dev_run_path),
    }
    row.update(metrics)
    print(
        pd.DataFrame([row])[
            ["lang", "split", f"nDCG@{k}", f"MRR@{k}", f"Prec@{k}", f"Recall@{k}", f"F1@{k}", f"MAP@{k}"]
        ].to_string(index=False)
    )
    return row


def run_test_split(lang: str, cfg: dict, team_name: str, approach_id: str, k: int) -> Dict[str, str]:
    print("\n" + "=" * 100)
    print(f"Running JobBERT test pipeline for language: {lang}")
    print("=" * 100)

    corpus, topics = load_task_a_inputs(cfg["test_dir"])
    print("test corpus:", corpus.shape)
    print("test topics:", topics.shape)
    submission_k = max(1, len(corpus))

    run_df = build_dense_run(
        corpus=corpus,
        topics=topics,
        model_name=MODEL_NAME,
        batch_size_doc=cfg["batch_size_doc"],
        batch_size_query=cfg["batch_size_query"],
        k=submission_k,
    )

    raw_run_path = RUNS_ROOT / f"run_test_{lang}_jobbert_dense_full.trec"
    save_trec_run(run_df, raw_run_path, tag=f"{lang}_jobbert_dense_full")

    submission_run_path = SUBMISSION_ROOT / f"run_{cfg['submission_lang_pair']}_{team_name}.trec"
    save_trec_run(run_df, submission_run_path, tag=f"{team_name}_{approach_id}")

    info = {
        "lang": lang,
        "split": "test",
        "submission_lang_pair": cfg["submission_lang_pair"],
        "queries": str(len(topics)),
        "docs": str(len(corpus)),
        "ranked_docs_per_query": str(submission_k),
        "raw_run_path": str(raw_run_path),
        "submission_run_path": str(submission_run_path),
    }
    print(
        pd.DataFrame([info])[
            ["lang", "split", "submission_lang_pair", "queries", "docs", "ranked_docs_per_query", "submission_run_path"]
        ].to_string(index=False)
    )
    return info


def main(
    team_name: str = DEFAULT_TEAM_NAME,
    approach_id: str = DEFAULT_APPROACH_ID,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    team_name = sanitize_token(team_name)
    approach_id = sanitize_token(approach_id)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []
    submission_items: List[Dict[str, str]] = []

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL_NAME:", MODEL_NAME)
    print("TEAM_NAME:", team_name)
    print("APPROACH_ID:", approach_id)
    print("K:", k)

    for lang, cfg in LANG_CONFIGS.items():
        metrics_rows.append(run_development_split(lang, cfg, team_name=team_name, approach_id=approach_id, k=k))
        submission_items.append(run_test_split(lang, cfg, team_name=team_name, approach_id=approach_id, k=k))

    metrics_df = pd.DataFrame(metrics_rows).sort_values("lang").reset_index(drop=True)
    metrics_path = RUNS_ROOT / f"jobbert_dense_metrics_at{k}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    submission_manifest = pd.DataFrame(submission_items).sort_values("lang").reset_index(drop=True)
    submission_manifest_path = SUBMISSION_ROOT / f"{team_name}_{approach_id}_test_manifest.csv"
    submission_manifest.to_csv(submission_manifest_path, index=False)

    submission_paths = [Path(item["submission_run_path"]) for item in submission_items]
    zip_path = SUBMISSION_ROOT / f"{team_name}_{approach_id}_test_submission.zip"
    zip_submission(submission_paths, zip_path)

    print("\nSaved metrics:", metrics_path)
    print("Saved test manifest:", submission_manifest_path)
    print("Saved submission zip:", zip_path)
    print("Saved test submission runs:")
    for path in submission_paths:
        print(" ", path)

    return metrics_df


if __name__ == "__main__":
    results_df = main()
    pd.set_option("display.max_colwidth", 140)
    print("\nResults:")
    print(results_df.to_string(index=False))
