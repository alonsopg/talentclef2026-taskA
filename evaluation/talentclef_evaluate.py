import argparse
import pandas as pd
import os
from ranx import Qrels, Run, evaluate

def load_mappings(mappings_dir):
    """
    Loads ID mapping files from the mappings directory.
    Returns dictionaries for corpus and query mappings.
    Format: external_id -> internal_id (with language and gender info)
    """
    corpus_mapping = {}
    query_mapping = {}
    
    if not os.path.isdir(mappings_dir):
        return corpus_mapping, query_mapping
    
    mapping_files = os.listdir(mappings_dir)
    
    # Load corpus mapping
    corpus_files = [f for f in mapping_files if f.startswith("corpus_mapping")]
    if corpus_files:
        corpus_path = os.path.join(mappings_dir, corpus_files[0])
        corpus_df = pd.read_csv(corpus_path, sep="\t", header=None, dtype=str)
        # Format: external_id, internal_id
        if len(corpus_df.columns) >= 2:
            corpus_mapping = dict(zip(corpus_df.iloc[:, 0], corpus_df.iloc[:, 1]))
    
    # Load query mapping
    query_files = [f for f in mapping_files if f.startswith("query_mapping")]
    if query_files:
        query_path = os.path.join(mappings_dir, query_files[0])
        query_df = pd.read_csv(query_path, sep="\t", header=None, dtype=str)
        if len(query_df.columns) >= 2:
            query_mapping = dict(zip(query_df.iloc[:, 0], query_df.iloc[:, 1]))
    
    return corpus_mapping, query_mapping

def load_qrels(qrels_path, lang_mode=None, mappings_dir=None, gender_filter=None):
    """
    Loads the qrels file (TREC format: q_id, iter, doc_id, rel)
    and converts it to a Qrels object.
    
    Args:
        qrels_path: Path to qrels file
        lang_mode: Language mode filter (e.g., 'en', 'es', 'en-es')
        mappings_dir: Directory containing ID mapping files
        gender_filter: Gender filter for candidates ('m' or 'f')
    """
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=None,
                           names=["q_id", "iter", "doc_id", "rel"],
                           dtype={"q_id": str, "doc_id": str, "rel":int})
    
    # Ensure IDs are strings before applying mappings
    qrels_df["q_id"] = qrels_df["q_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    
    # Apply ID mappings if provided
    if mappings_dir:
        corpus_mapping, query_mapping = load_mappings(mappings_dir)
        if query_mapping:
            qrels_df["q_id"] = qrels_df["q_id"].map(query_mapping).fillna(qrels_df["q_id"])
        if corpus_mapping:
            qrels_df["doc_id"] = qrels_df["doc_id"].map(corpus_mapping).fillna(qrels_df["doc_id"])
    
    # Apply language mode filter if specified (only when IDs contain language patterns)
    if lang_mode and mappings_dir:
        if "-" in lang_mode:
            # Cross-lingual mode (e.g., "en-es")
            query_lang, doc_lang = lang_mode.split("-")
            qrels_df = qrels_df[
                qrels_df["q_id"].str.contains(f"_{query_lang}_") & 
                qrels_df["doc_id"].str.contains(f"_{doc_lang}_")
            ]
        else:
            # Monolingual mode (e.g., "en" or "es")
            qrels_df = qrels_df[
                qrels_df["q_id"].str.contains(f"_{lang_mode}_") & 
                qrels_df["doc_id"].str.contains(f"_{lang_mode}_")
            ]
    
    # Apply gender filter to documents only (not queries)
    if gender_filter:
        qrels_df = qrels_df[qrels_df["doc_id"].str.contains(f"_{gender_filter}")]
    
    # Ensure IDs are object dtype (strings) before creating Qrels object
    qrels_df["q_id"] = qrels_df["q_id"].astype("object")
    qrels_df["doc_id"] = qrels_df["doc_id"].astype("object")
    
    return Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="rel")

def load_run(run_path, lang_mode=None, mappings_dir=None, gender_filter=None):
    """
    Loads the run file (TREC format: q_id, Q0, doc_id, rank, score, [tag])
    and converts it to a Run object.
    
    Args:
        run_path: Path to run file
        lang_mode: Language mode filter (e.g., 'en', 'es', 'en-es')
        mappings_dir: Directory containing ID mapping files
        gender_filter: Gender filter for candidates ('m' or 'f')
    """
    run_df = pd.read_csv(run_path, sep=r"\s+", header=None)
    
    # Assign column names based on the number of columns
    if run_df.shape[1] == 5:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score"]
    elif run_df.shape[1] >= 6:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score", "tag"]
    else:
        raise ValueError("The run file does not have the expected format.")
    
    # Convert IDs to strings first
    run_df["q_id"] = run_df["q_id"].astype(str)
    run_df["doc_id"] = run_df["doc_id"].astype(str)
    
    # Apply ID mappings if provided
    if mappings_dir:
        corpus_mapping, query_mapping = load_mappings(mappings_dir)
        if query_mapping:
            run_df["q_id"] = run_df["q_id"].map(query_mapping).fillna(run_df["q_id"])
        if corpus_mapping:
            run_df["doc_id"] = run_df["doc_id"].map(corpus_mapping).fillna(run_df["doc_id"])
    
    # Apply language mode filter if specified (only when IDs contain language patterns)
    if lang_mode and mappings_dir:
        if "-" in lang_mode:
            # Cross-lingual mode (e.g., "en-es")
            query_lang, doc_lang = lang_mode.split("-")
            run_df = run_df[
                run_df["q_id"].str.contains(f"_{query_lang}_") & 
                run_df["doc_id"].str.contains(f"_{doc_lang}_")
            ]
        else:
            # Monolingual mode (e.g., "en" or "es")
            run_df = run_df[
                run_df["q_id"].str.contains(f"_{lang_mode}_") & 
                run_df["doc_id"].str.contains(f"_{lang_mode}_")
            ]
    
    # Apply gender filter to documents only (not queries)
    if gender_filter:
        run_df = run_df[run_df["doc_id"].str.contains(f"_{gender_filter}")]
    
    # Ensure IDs are object dtype (strings) before creating Run object
    run_df["q_id"] = run_df["q_id"].astype("object")
    run_df["doc_id"] = run_df["doc_id"].astype("object")
    
    return Run.from_df(run_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

def evaluate_task_a(qrels, run, gender_filter=None):
    """
    Evaluate Task A: Job-to-CV matching.
    Uses standard IR metrics.
    
    Args:
        qrels: Qrels object
        run: Run object
        gender_filter: Optional gender filter ('m', 'f', 'n' for male/female/neutral)
    """
    metrics = ["map", "mrr", "ndcg", "precision@5", "precision@10", "precision@100"]
    
    if gender_filter:
        print(f"Running Task A evaluation (gender: {gender_filter})...")
    else:
        print("Running Task A evaluation...")
    
    results = evaluate(qrels, run, metrics)
    return results

def evaluate_task_b(qrels, run):
    """
    Evaluate Task B: Job-to-Skill matching.
    Computes metrics for both binary and graded relevance scenarios.
    
    Args:
        qrels: Qrels object with relevance judgments (1 or 2)
        run: Run object with system predictions
    
    Returns:
        dict: Dictionary with results for both scenarios
    """
    metrics = ["ndcg", "map", "mrr", "precision@5", "precision@10", "precision@100"]
    
    print("Running Task B evaluation...")
    
    # Scenario 1: Binary relevance (all relevant skills treated equally)
    # Convert all relevance scores to 1 (relevant) vs 0 (not relevant)
    qrels_binary_dict = {}
    for q_id in qrels.qrels:
        qrels_binary_dict[q_id] = {}
        for doc_id in qrels.qrels[q_id]:
            # Convert any positive relevance to 1
            qrels_binary_dict[q_id][doc_id] = 1 if qrels.qrels[q_id][doc_id] > 0 else 0
    
    qrels_binary = Qrels(qrels_binary_dict)
    results_binary = evaluate(qrels_binary, run, metrics)
    
    # Scenario 2: Graded relevance (distinguish between levels 2 and 1)
    results_graded = evaluate(qrels, run, metrics)
    
    return {
        "binary": results_binary,
        "graded": results_graded
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for TalentCLEF2026. "
                    "Supports both Task A and Task B evaluations with multilingual and gender-aware options."
    )
    parser.add_argument("--task", required=True, choices=["A", "B", "a", "b"],
                        help="Task to evaluate: A (job-to-CV matching) or B (TBD)")
    parser.add_argument("--qrels", required=True, help="Path to the qrels file (TREC format)")
    parser.add_argument("--run", required=True, help="Path to the run file (TREC format)")
    parser.add_argument("--lang-mode", dest="lang_mode", 
                        choices=["en", "es", "en-es"],
                        help="Language mode: monolingual (en, es) or cross-lingual (en-es)")
    parser.add_argument("--mappings", dest="mappings_dir",
                        help="Directory containing ID mapping files (corpus_mapping.tsv, query_mapping.tsv)")
    parser.add_argument("--gender", choices=["m", "f"],
                        help="Gender filter for candidates: m (male), f (female)")
    args = parser.parse_args()

    # Normalize task to uppercase
    task = args.task.upper()
    lang_mode = args.lang_mode if args.lang_mode else None
    gender_filter = args.gender if args.gender else None

    # Validate that lang-mode is required for Task A
    if task == "A" and not lang_mode:
        parser.error("--lang-mode is required for Task A. Please specify: en, es, or en-es")

    # Display received parameters
    print("=" * 60)
    print(f"TalentCLEF 2026 - Task {task} Evaluation")
    print("=" * 60)
    print("Received parameters:")
    print(f"  Task: {task}")
    print(f"  Qrels: {args.qrels}")
    print(f"  Run: {args.run}")
    if lang_mode:
        print(f"  Language Mode: {lang_mode}")
    if args.mappings_dir:
        print(f"  Mappings Directory: {args.mappings_dir}")
    if gender_filter:
        print(f"  Gender Filter: {gender_filter}")
    print()

    print("Loading qrels...")
    qrels = load_qrels(args.qrels, lang_mode=lang_mode, mappings_dir=args.mappings_dir, gender_filter=gender_filter)
    print("Loading run...")
    run = load_run(args.run, lang_mode=lang_mode, mappings_dir=args.mappings_dir, gender_filter=gender_filter)
    print()

    # Execute task-specific evaluation
    if task == "A":
        results = evaluate_task_a(qrels, run, gender_filter=gender_filter)
    elif task == "B":
        results = evaluate_task_b(qrels, run)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Display common evaluation results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if task == "A":
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
    elif task == "B":
        print("\n--- General Relevance (1 or 2 → relevant) ---")
        for metric, score in results["binary"].items():
            print(f"{metric}: {score:.4f}")
        
        print("\n--- Graded Relevance (2 = core skill, 1 = contextual skill) ---")
        for metric, score in results["graded"].items():
            print(f"{metric}: {score:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()