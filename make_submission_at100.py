from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyterrier as pt
import pyt_splade

from task_a_colbert_rerank import COLBERT_MODEL_NAME_DEFAULT, build_colbert_rerank_run
from task_a_jobbert_dense_en_es import (
    MODEL_NAME,
    build_dense_run,
    load_task_a_inputs,
    sanitize_token,
    save_trec_run,
)
from task_a_query_augmentation import augment_topics, build_query_view


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "test"
ART_ROOT = PROJECT_ROOT / "var" / "artifacts" / "pyterrier_experiments"
SUBMISSION_ROOT = PROJECT_ROOT / "var" / "artifacts" / "pyterrier_submission"
AUGMENTATION_CACHE_ROOT = PROJECT_ROOT / "var" / "cache" / "query_augmentation"

JOBBERT_BATCH_SIZE_DOC = 16
JOBBERT_BATCH_SIZE_QUERY = 16
SPLADE_RRF_K = 60
SPLADE_PRIMARY_WEIGHT = 2.0
SPLADE_SECONDARY_WEIGHT = 1.0
COLBERT_MODEL_NAME = COLBERT_MODEL_NAME_DEFAULT
COLBERT_DEVICE = "cpu"
COLBERT_QUERY_LENGTH = 128
COLBERT_DOCUMENT_LENGTH = 220
COLBERT_QUERY_BATCH_SIZE = 4
COLBERT_DOCUMENT_BATCH_SIZE = 2


def ensure_pt() -> None:
    if not pt.started():
        pt.init()


def build_pt_corpus_iter(corpus_df: pd.DataFrame) -> Iterable[dict]:
    for _, row in corpus_df.iterrows():
        yield {"docno": str(row["docno"]), "text": str(row["text"])}


def build_or_load_splade_index(corpus_df: pd.DataFrame, index_dir: Path):
    data_properties = index_dir / "data.properties"
    if data_properties.exists():
        return pt.IndexFactory.of(str(index_dir))

    splade = pyt_splade.Splade()
    try:
        splade_indexer = pt.IterDictIndexer(
            str(index_dir),
            pretokenised=True,
            meta={"docno": 64},
            overwrite=True,
            stemmer="none",
            stopwords=None,
            tokeniser="identity",
        )
    except Exception:
        splade_indexer = pt.IterDictIndexer(
            str(index_dir),
            pretokenised=True,
            meta={"docno": 64},
            overwrite=True,
            stemmer="none",
            stopwords=None,
        )

    index_pipe = splade.doc_encoder() >> splade_indexer
    return index_pipe.index(build_pt_corpus_iter(corpus_df), batch_size=32)


def pt_run_to_standard_run(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out["score"] = out["score"].astype(float)
    out = out.sort_values(["qid", "score"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = out.groupby("qid").cumcount() + 1
    return out.groupby("qid", sort=False).head(k)[["qid", "docno", "score", "rank"]].copy()


def cut_run_to_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out = out.sort_values(["qid", "rank"], ascending=[True, True]).reset_index(drop=True)
    return out.groupby("qid", sort=False).head(k)[["qid", "docno", "score", "rank"]].copy()


def rerank_candidates_with_secondary_rrf(
    primary_candidates: pd.DataFrame,
    secondary_full_run: pd.DataFrame,
    final_k: int,
    rrf_k: int = 60,
    primary_weight: float = 2.0,
    secondary_weight: float = 1.0,
) -> pd.DataFrame:
    secondary_rank = {
        (str(row.qid), str(row.docno)): int(row.rank)
        for row in secondary_full_run[["qid", "docno", "rank"]].itertuples(index=False)
    }

    rows = []
    for qid, group in primary_candidates.groupby("qid", sort=False):
        scored = []
        group = group.sort_values("rank")
        for row in group.itertuples(index=False):
            docno = str(row.docno)
            p_rank = int(row.rank)
            s_rank = secondary_rank.get((str(qid), docno))
            score = primary_weight / (rrf_k + p_rank)
            if s_rank is not None:
                score += secondary_weight / (rrf_k + s_rank)
            scored.append((docno, score))

        scored = sorted(scored, key=lambda item: item[1], reverse=True)[:final_k]
        for rank, (docno, score) in enumerate(scored, start=1):
            rows.append({"qid": str(qid), "docno": docno, "score": float(score), "rank": int(rank)})

    return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])


def build_jobbert_run_for_topics(corpus: pd.DataFrame, topics_df: pd.DataFrame, k: int) -> pd.DataFrame:
    return build_dense_run(
        corpus=corpus,
        topics=topics_df,
        model_name=MODEL_NAME,
        batch_size_doc=JOBBERT_BATCH_SIZE_DOC,
        batch_size_query=JOBBERT_BATCH_SIZE_QUERY,
        k=k,
    )


def load_topics_for_lang(lang: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus, topics = load_task_a_inputs(DATA_ROOT / lang)
    topics_aug = augment_topics(
        topics=topics,
        lang=lang,
        split_name="test",
        cache_root=AUGMENTATION_CACHE_ROOT,
        overwrite=False,
    )
    return corpus, build_query_view(topics_aug, "ORIGINAL")


def build_jobbert_submission(lang: str, k: int) -> pd.DataFrame:
    corpus, topics_pt = load_topics_for_lang(lang)
    return build_jobbert_run_for_topics(corpus, topics_pt, k)


def build_jobbert_splade_rerank_submission(lang: str, k: int) -> pd.DataFrame:
    corpus, topics_pt = load_topics_for_lang(lang)
    art_dir = ART_ROOT / lang
    art_dir.mkdir(parents=True, exist_ok=True)

    jobbert_run = build_jobbert_run_for_topics(corpus, topics_pt, len(corpus))
    primary_candidates = cut_run_to_k(jobbert_run, k)

    splade_index = build_or_load_splade_index(corpus, art_dir / "pt_splade_index_test")
    splade = pyt_splade.Splade()
    retriever = splade.query_encoder() >> pt.terrier.Retriever(splade_index, wmodel="Tf")
    splade_run = pt_run_to_standard_run(retriever.transform(topics_pt), len(corpus))

    return rerank_candidates_with_secondary_rrf(
        primary_candidates=primary_candidates,
        secondary_full_run=splade_run,
        final_k=k,
        rrf_k=SPLADE_RRF_K,
        primary_weight=SPLADE_PRIMARY_WEIGHT,
        secondary_weight=SPLADE_SECONDARY_WEIGHT,
    )


def build_jobbert_colbert_submission(lang: str, k: int) -> pd.DataFrame:
    corpus, topics_pt = load_topics_for_lang(lang)
    return build_colbert_rerank_run(
        corpus=corpus,
        topics=topics_pt,
        primary_candidates=build_jobbert_run_for_topics(corpus, topics_pt, k),
        final_k=k,
        model_name=COLBERT_MODEL_NAME,
        device=COLBERT_DEVICE,
        query_length=COLBERT_QUERY_LENGTH,
        document_length=COLBERT_DOCUMENT_LENGTH,
        query_batch_size=COLBERT_QUERY_BATCH_SIZE,
        document_batch_size=COLBERT_DOCUMENT_BATCH_SIZE,
        show_progress_bar=False,
    )


def build_run_for_lang(lang: str, pipeline: str, k: int) -> pd.DataFrame:
    key = pipeline.upper().strip()
    if key == "JOBBERT":
        return build_jobbert_submission(lang=lang, k=k)
    if key in {"JOBBERT_SPLADE_RRF", "JOBBERT>>SPLADE_RRF"}:
        return build_jobbert_splade_rerank_submission(lang=lang, k=k)
    if key in {"JOBBERT_COLBERT", "JOBBERT>>COLBERT"}:
        return build_jobbert_colbert_submission(lang=lang, k=k)
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def default_label_for_pipelines(en_pipeline: str, es_pipeline: str) -> str:
    def normalize(value: str) -> str:
        return value.lower().replace(">>", "_rerank_").replace("+", "_").replace(" ", "_")

    return f"en_{normalize(en_pipeline)}__es_{normalize(es_pipeline)}"


def zip_flat(files: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="alonso")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--en-pipeline", default="JOBBERT_SPLADE_RRF")
    parser.add_argument("--es-pipeline", default="JOBBERT_COLBERT")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    ensure_pt()
    k = int(args.k)
    team_name = sanitize_token(args.team_name)
    en_pipeline = args.en_pipeline.strip()
    es_pipeline = args.es_pipeline.strip()
    label = sanitize_token(args.label) if args.label else sanitize_token(
        default_label_for_pipelines(en_pipeline, es_pipeline)
    )

    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    variant_dir = SUBMISSION_ROOT / label
    variant_dir.mkdir(parents=True, exist_ok=True)

    en_run = build_run_for_lang(lang="en", pipeline=en_pipeline, k=k)
    es_run = build_run_for_lang(lang="es", pipeline=es_pipeline, k=k)

    en_path = variant_dir / f"run_en-en_{team_name}.trec"
    es_path = variant_dir / f"run_es-es_{team_name}.trec"
    save_trec_run(en_run, en_path, tag=f"{team_name}_{sanitize_token(en_pipeline.lower())}_at{k}")
    save_trec_run(es_run, es_path, tag=f"{team_name}_{sanitize_token(es_pipeline.lower())}_at{k}")

    zip_path = SUBMISSION_ROOT / f"{team_name}_{label}_at{k}.zip"
    zip_flat([en_path, es_path], zip_path)

    manifest = SUBMISSION_ROOT / f"{team_name}_{label}_at{k}.txt"
    manifest.write_text(
        "\n".join(
            [
                f"team_name={team_name}",
                f"k={k}",
                f"label={label}",
                f"en_pipeline={en_pipeline}",
                f"es_pipeline={es_pipeline}",
                f"en_run={en_path}",
                f"es_run={es_path}",
                f"zip={zip_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Saved:")
    print(en_path)
    print(es_path)
    print(zip_path)
    print(manifest)


if __name__ == "__main__":
    main()
