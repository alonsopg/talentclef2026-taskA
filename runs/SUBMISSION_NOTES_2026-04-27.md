# TalentCLEF 2026 Task A Submission Notes

## Recommended Upload Order

1. **Best leaderboard run**
   - ZIP: `runs/official_submissions/best_jobbert_splade_rrf/alonso_best_jobbert_splade_rrf_at100.zip`
   - Pipelines:
     - `en-en`: `JobBERT>>SPLADE_RRF`
     - `es-es`: `JobBERT>>SPLADE_RRF`
   - Rationale: strongest dev performance overall.

2. **Mixed strong + innovative run**
   - ZIP: `runs/official_submissions/mixed_en_splade_es_colbert/alonso_mixed_en_splade_es_colbert_at100.zip`
   - Pipelines:
     - `en-en`: `JobBERT>>SPLADE_RRF`
     - `es-es`: `JobBERT>>ColBERT`
   - Rationale: keeps the best English setup while using a late-interaction reranker for Spanish.

3. **Pure ColBERT variant**
   - ZIP: `runs/official_submissions/jobbert_colbert/alonso_jobbert_colbert_at100.zip`
   - Pipelines:
     - `en-en`: `JobBERT>>ColBERT`
     - `es-es`: `JobBERT>>ColBERT`
   - Rationale: most architecturally distinctive late-interaction submission.

## Short Model Descriptions

### 1. Hybrid Dense-Sparse Reranking

This system uses `TechWolf/JobBERT-v3` as a first-stage dense retriever to obtain the top 100 candidate resumes for each job description. The candidate set is then reranked with a sparse lexical signal from SPLADE using reciprocal-rank fusion over the candidate pool. The final submission includes the top 100 ranked resumes per query.

Suggested label:
`Hybrid dense-sparse reranking with JobBERT first-stage retrieval and SPLADE candidate reranking.`

### 2. Mixed Dense-Sparse / Late-Interaction Reranking

This submission combines two reranking strategies. For English, it uses the same `JobBERT>>SPLADE_RRF` hybrid dense-sparse reranking pipeline. For Spanish, it uses `TechWolf/JobBERT-v3` for first-stage retrieval and a pretrained multilingual ColBERT reranker (`jinaai/jina-colbert-v2`) for late-interaction reranking over the top 100 candidates.

Suggested label:
`Language-specific reranking with SPLADE for English and multilingual ColBERT for Spanish on top of JobBERT retrieval.`

### 3. Multilingual Late-Interaction Reranking

This system uses `TechWolf/JobBERT-v3` as a first-stage dense retriever and a pretrained multilingual ColBERT reranker (`jinaai/jina-colbert-v2`) as the second stage for both English and Spanish. For each job description, the model reranks the top 100 resumes produced by the first-stage retriever and outputs the top 100 results.

Suggested label:
`Multilingual late-interaction reranking with JobBERT retrieval and pretrained ColBERT candidate reranking.`

## Dev Reference Metrics

- `JobBERT>>SPLADE_RRF`
  - `en MAP@50 = 0.7808`
  - `es MAP@50 = 0.7350`

- `JobBERT>>ColBERT`
  - `en MAP@50 = 0.7728`
  - `es MAP@50 = 0.7348`

- `Mixed`
  - English uses the best SPLADE reranker.
  - Spanish uses the ColBERT reranker, which is essentially tied with SPLADE on dev.
