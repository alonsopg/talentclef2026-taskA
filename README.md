# TalentCLEF 2026 Task A

Code, notebooks, development results, and official search runs for the TalentCLEF 2026 Task A submission:

**Hybrid Dense-Sparse Retrieval and Re-Ranking for Multilingual Resume Retrieval from Job Descriptions**

This repository contains the reproducible code used to build and compare three main Task A systems:

- `JobBERT`: dense first-stage retrieval with `TechWolf/JobBERT-v3`
- `JobBERT -> SPLADE`: dense retrieval followed by sparse reranking with reciprocal-rank fusion
- `JobBERT -> ColBERT`: dense retrieval followed by multilingual late-interaction reranking with `jinaai/jina-colbert-v2`

The public repository includes:

- the core Python scripts
- the main experiment notebooks
- the official TalentCLEF evaluation script
- the development results table used for model selection
- the final official run files submitted to Codabench

The TalentCLEF data itself is **not** included here.

## Repository Layout

```text
.
├── data/
│   └── README.md
├── evaluation/
│   ├── README.md
│   ├── requirements.txt
│   └── talentclef_evaluate.py
├── notebooks/
│   ├── TASK-A-Augmentation-EN-ES-V1.ipynb
│   ├── TASK-A-JobBERT-Dense-EN-ES-V1.ipynb
│   └── TASK-A-PyTerrier-EN-ES-V1.ipynb
├── results/
│   └── dev/
│       └── reconstructed_dev_metrics_at50.csv
├── runs/
│   ├── SUBMISSION_NOTES_2026-04-27.md
│   └── official_submissions/
│       ├── best_jobbert_splade_rrf/
│       ├── jobbert_colbert/
│       └── mixed_en_splade_es_colbert/
├── make_submission_at100.py
├── requirements.txt
├── task_a_colbert_rerank.py
├── task_a_jobbert_dense_en_es.py
└── task_a_query_augmentation.py
```

## Data

Download the official TalentCLEF 2026 Task A data and place it under `data/` with this structure:

```text
data/
├── development/
│   ├── en/
│   └── es/
└── test/
    ├── en/
    └── es/
```

See [data/README.md](data/README.md) for details.

## Environment

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

The official evaluator has its own minimal dependency file in `evaluation/requirements.txt`.

## Main Files

- `task_a_jobbert_dense_en_es.py`
  - JobBERT-only dense retrieval baseline
  - evaluates on the development split
  - exports official test-set run files and a flat submission ZIP
- `task_a_query_augmentation.py`
  - deterministic query augmentation utilities
  - compact rewrite and structured query views
- `task_a_colbert_rerank.py`
  - multilingual ColBERT reranking utilities using PyLate
- `make_submission_at100.py`
  - builds official `@100` Task A submissions for different EN/ES pipeline combinations

## Reproducing the Main Runs

Run the JobBERT baseline:

```bash
export TALENTCLEF_TEAM_NAME=alonso
export TALENTCLEF_APPROACH_ID=jobbert_dense
python task_a_jobbert_dense_en_es.py
```

Build the best-performing official submission:

```bash
python make_submission_at100.py \
  --team-name alonso \
  --k 100 \
  --en-pipeline JOBBERT_SPLADE_RRF \
  --es-pipeline JOBBERT_SPLADE_RRF \
  --label best_jobbert_splade_rrf
```

Build the mixed submission:

```bash
python make_submission_at100.py \
  --team-name alonso \
  --k 100 \
  --en-pipeline JOBBERT_SPLADE_RRF \
  --es-pipeline JOBBERT_COLBERT \
  --label mixed_en_splade_es_colbert
```

Build the ColBERT submission:

```bash
python make_submission_at100.py \
  --team-name alonso \
  --k 100 \
  --en-pipeline JOBBERT_COLBERT \
  --es-pipeline JOBBERT_COLBERT \
  --label jobbert_colbert
```

## Development Results

The main development comparison table used for model selection is in:

- `results/dev/reconstructed_dev_metrics_at50.csv`

Best development performance:

- `en`: `JobBERT -> SPLADE_RRF`, `MAP@50 = 0.7808`
- `es`: `JobBERT -> SPLADE_RRF`, `MAP@50 = 0.7350`

## Official Search Runs

The submitted official run files are included under `runs/official_submissions/`:

- `best_jobbert_splade_rrf`
- `mixed_en_splade_es_colbert`
- `jobbert_colbert`

Each submission folder contains:

- `run_en-en_alonso.trec`
- `run_es-es_alonso.trec`
- `manifest.txt`

## Notes

- The notebooks are kept for transparency and experiment tracking.
- The scripts are the cleaner entry points for reproducing the final runs.
- Public Task A data does not expose gender labels, so no direct public fairness reranking pipeline is included here.
