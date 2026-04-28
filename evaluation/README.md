# TalentCLEF 2026 Evaluation Script

This repository contains the official evaluation script for the [**TalentCLEF 2026** shared task](https://talentclef.github.io/talentclef/docs/). The script loads qrels and run files (both in TREC format), evaluates the run against the qrels using the **Ranx** evaluation framework, and computes standard information retrieval metrics.

## Overview

The TalentCLEF 2026 evaluation script supports:
- [**Task A**: Contextualized Job-Person Matching](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/#task-a--contextualized-job-person-matching)
  - Monolingual evaluation (English, Spanish)
  - Cross-lingual evaluation (English-Spanish)
  - Gender-aware evaluation
- [**Task B**: Job-Skill Matching with Skill Type Classification](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/#task-b-job-skill-matching-with-skill-type-classification)
  - General relevance evaluation (binary: relevant vs. not relevant)
  - Graded relevance evaluation (core skills vs. contextual skills vs. not relevant)


## Setting Up the Evaluation Environment

### Clone TalentCLEF Evaluation Script Repository

Open your terminal and run:

```bash
git clone https://github.com/TalentCLEF/talentclef26_evaluation_script.git
cd talentclef26_evaluation_script
```

### Create Python Environment

1. **Create the Virtual Environment:**
   ```bash
   python3 -m venv .env
   ```

2. **Activate the Virtual Environment:**
   ```bash
   source .env/bin/activate
   ```

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Required packages:**
- pandas
- ranx

## Toy Data

The repository includes toy data for testing in the `toy-data/` directory. 

## Usage

### Basic Command

```bash
python talentclef_evaluate.py --task <A|B> --qrels <qrels_file> --run <run_file> [OPTIONS]
```

### Arguments

- `--task`: **(Required)** Task to evaluate: `A` or `B` (case-insensitive)
- `--qrels`: **(Required)** Path to the qrels file in TREC format
- `--run`: **(Required)** Path to the run file in TREC format
- `--lang-mode`: **(Required for Task A)** Language mode: `en`, `es`, `en-es`
- `--mappings`: **(Optional)** Directory containing ID mapping files
- `--gender`: **(Optional)** Gender filter for candidates only: `m` (male), `f` (female). **Note: Only for demonstration; not available in public dev and test set.**

### Examples

Below are examples for each evaluation scenario using the toy data.

<details>
<summary><b>1. Monolingual Evaluation - English</b></summary>

Evaluate English job postings against English CVs.

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode en \
  --qrels toy-data/taskA/en/qrels.tsv \
  --run toy-data/taskA/en/sample_run.txt
```

</details>

<details>
<summary><b>2. Monolingual Evaluation - Spanish</b></summary>

Evaluate Spanish job postings against Spanish CVs.

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode es \
  --qrels toy-data/taskA/es/qrels.tsv \
  --run toy-data/taskA/es/sample_run.txt
```


</details>

<details>
<summary><b>3. Cross-lingual Evaluation (English-Spanish)</b></summary>

Evaluate English job postings against Spanish CVs (cross-lingual matching).

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode en-es \
  --qrels toy-data/taskA/en-es/qrels.tsv \
  --run toy-data/taskA/en-es/sample_run.txt
```


</details>

<details>
<summary><b>4. Gender-Aware Evaluation - English (Male Only)</b></summary>

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode en \
  --qrels toy-data/taskA/mappings_gender/en/qrels.tsv \
  --run toy-data/taskA/mappings_gender/en/sample_run.txt \
  --mappings toy-data/taskA/mappings_gender/en \
  --gender m
```

> ⚠️ **Note:** The --gender flag and gender-disaggregated evaluation are provided here as an EXAMPLE of how the internal evaluation system will work. Gender information and mappings will NOT be available in the public test set. This toy data demonstrates the evaluation methodology for transparency purposes only.

</details>

<details>
<summary><b>5. Gender-Aware Evaluation - English (Female Only)</b></summary>

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode en \
  --qrels toy-data/taskA/mappings_gender/en/qrels.tsv \
  --run toy-data/taskA/mappings_gender/en/sample_run.txt \
  --mappings toy-data/taskA/mappings_gender/en \
  --gender f
```

> ⚠️ **Note:** The --gender flag and gender-disaggregated evaluation are provided here as an EXAMPLE of how the internal evaluation system will work. Gender information and mappings will NOT be available in the public test set. This toy data demonstrates the evaluation methodology for transparency purposes only.

</details>

<details>
<summary><b>7. Gender-Aware Evaluation - Spanish (Male Only)</b></summary>

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode es \
  --qrels toy-data/taskA/mappings_gender/es/qrels.tsv \
  --run toy-data/taskA/mappings_gender/es/sample_run.txt \
  --mappings toy-data/taskA/mappings_gender/es \
  --gender m
```

> ⚠️ **Note:** The --gender flag and gender-disaggregated evaluation are provided here as an EXAMPLE of how the internal evaluation system will work. Gender information and mappings will NOT be available in the public test set. This toy data demonstrates the evaluation methodology for transparency purposes only.

</details>

<details>
<summary><b>8. Gender-Aware Evaluation - Spanish (Female Only)</b></summary>

```bash
python talentclef_evaluate.py \
  --task A \
  --lang-mode es \
  --qrels toy-data/taskA/mappings_gender/es/qrels.tsv \
  --run toy-data/taskA/mappings_gender/es/sample_run.txt \
  --mappings toy-data/taskA/mappings_gender/es \
  --gender f
```

> ⚠️ **Note:** The --gender flag and gender-disaggregated evaluation are provided here as an EXAMPLE of how the internal evaluation system will work. Gender information and mappings will NOT be available in the public test set. This toy data demonstrates the evaluation methodology for transparency purposes only.

</details>

<details>
<summary><b>9. Task B Evaluation - Job-to-Skill Matching</b></summary>

Evaluate job titles against skill corpus. Task B provides two evaluation scenarios:
- **General Relevance**: All relevant skills (levels 1 or 2) are treated equally
- **Graded Relevance**: Distinguishes between core skills (level 2) and contextual skills (level 1). 

```bash
python talentclef_evaluate.py \
  --task B \
  --qrels toy-data/taskB/qrels.tsv \
  --run toy-data/taskB/sample_run.txt
```

**Note:** Task B does not require `--lang-mode` as it focuses on skill matching rather than multilingual document matching.

</details>

## Multilingual and Gender-Aware Evaluation

### Language Modes

The evaluation script supports different language configurations:

- **Monolingual**: `en` (English jobs → English candidates), `es` (Spanish jobs → Spanish candidates)
- **Cross-lingual**: `en-es` (English jobs → Spanish candidates)

## File Formats

### Qrels File (Ground Truth)

Tab-separated format with 4 columns:
```
query_id    iter    doc_id    relevance
```

Example:
```
1001    0    101    1
1001    0    102    0
1002    0    102    1
```

- `query_id`: Query/job posting identifier
- `iter`: Iteration (always 0 for standard TREC format)
- `doc_id`: Document/CV identifier
- `relevance`: Relevance judgment (1 = relevant, 0 = not relevant)

**Task B Qrels Format:**

For Task B (Job-to-Skill matching), the qrels file uses graded relevance:

```
toy_qb_jt_1    0    toy_cb_sk_1    2
toy_qb_jt_1    0    toy_cb_sk_2    2
toy_qb_jt_1    0    toy_cb_sk_5    1
```

- `query_id`: Job title identifier
- `iter`: Iteration (always 0)
- `doc_id`: Skill identifier
- `relevance`: Relevance level (2 = core skill, 1 = contextual skill, 0 = not relevant)

### Run File (System Output)

Space or tab-separated format with 5 or 6 columns:
```
query_id    Q0    doc_id    rank    score    [tag]
```

Example:
```
1001    Q0    101    1    0.95    my_system
1001    Q0    102    2    0.85    my_system
1002    Q0    102    1    0.92    my_system
```

- `query_id`: Query/job posting identifier
- `Q0`: Literal string "Q0" (TREC standard)
- `doc_id`: Document/CV identifier
- `rank`: Rank position (1-based)
- `score`: Relevance score (higher = more relevant)
- `tag`: Optional system identifier

## Evaluation Metrics

### Task A Metrics

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **Precision@5**: Precision at rank 5
- **Precision@10**: Precision at rank 10
- **Precision@100**: Precision at rank 100

### Task B Metrics

Task B evaluation computes metrics for two scenarios:

#### General Relevance (Binary)
All skills marked as relevant (1 or 2) are treated equally:
- **NDCG** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Precision@5**: Precision at rank 5
- **Precision@10**: Precision at rank 10
- **Precision@100**: Precision at rank 100

#### Graded Relevance
Distinguishes between skill types (2 = core skill, 1 = contextual skill):
- **NDCG** (Normalized Discounted Cumulative Gain) - gives higher weight to core skills
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Precision@5**: Precision at rank 5
- **Precision@10**: Precision at rank 10
- **Precision@100**: Precision at rank 100

**Relevance Levels:**
- **2**: Core skill
- **1**: Contextual skill
- **0**: Not relevant 




## Task Context

This evaluation script is developed specifically for the [**TalentCLEF 2026** shared task](https://talentclef.github.io/talentclef/docs/talentclef-2026/task-summary/),

