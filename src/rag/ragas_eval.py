"""
src/rag/ragas_eval.py — RAGAS Evaluation using Groq API
=========================================================
Updated to use Groq (llama-3.3-70b-versatile) for:
  - Answer generation
  - RAGAS LLM metrics (faithfulness, answer_relevancy, context_precision)

Run:
    cd "C:\\Users\\kashy\\OneDrive\\Desktop\\Masters Project"
    conda activate masters-project
    pip install langchain-groq
    python -m src.rag.ragas_eval

Output:
    output/reports/ragas_results.csv
    output/reports/ragas_summary.txt
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

from src.rag.tools import vector_search, pandas_analytics
from src.rag.graph_tool import graph_query
from src.rag.router import QueryRouter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR    = Path("output/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROQ_MODEL    = "llama-3.3-70b-versatile"
SLEEP_BETWEEN = 6    # Groq free tier: 30 req/min, so 6s between calls is safe
RETRY_SLEEP   = 30

# ---------------------------------------------------------------------------
# Eval questions — 10 total, covering all route types
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    "How many awards are in each category?",
    "Which subcategory has the most recognition records?",
    "What percentage of awards are in Organizational and Team Enablement?",
    "What kind of messages do engineering teams write?",
    "What themes appear in Creative and Brand Impact recognition?",
    "What does leadership recognition typically look like?",
    "Which departments recognize each other most?",
    "Which role group gives the most recognition?",
    "Compare the top categories and give examples of each",
    "Which subcategory has the most awards and what do those messages say?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def retrieve_context(question: str) -> list[str]:
    router   = QueryRouter()
    decision = router.route(question)
    contexts = []
    for tool_name in decision.tools:
        if tool_name == "vector_search":
            raw = vector_search(question, k=5)
        elif tool_name == "pandas_analytics":
            raw = pandas_analytics(question)
        elif tool_name == "graph_query":
            raw = graph_query(question)
        else:
            raw = ""
        if raw:
            chunks = [raw[i:i+500] for i in range(0, min(len(raw), 2000), 500)]
            contexts.extend(chunks)
    return contexts if contexts else ["No context retrieved."]


def safe_generate(client: Groq, question: str, contexts: list[str], attempt: int = 1) -> str:
    try:
        context_text = "\n\n".join(contexts)
        prompt = (
            f"Context from HR recognition database:\n\n{context_text}\n\n"
            f"---\nQuestion: {question}\n\n"
            f"Answer concisely using the context above."
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        if ("429" in str(e) or "rate_limit" in str(e).lower()) and attempt <= 3:
            wait = RETRY_SLEEP * attempt
            print(f"\n      Rate limit — waiting {wait}s (retry {attempt}/3)...")
            time.sleep(wait)
            return safe_generate(client, question, contexts, attempt + 1)
        raise


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(client: Groq) -> Dataset:
    print(f"\n[2/4] Building dataset ({len(EVAL_QUESTIONS)} questions, {SLEEP_BETWEEN}s sleep between calls)...\n")

    user_inputs        = []
    responses          = []
    retrieved_contexts = []
    reference          = []

    for i, question in enumerate(EVAL_QUESTIONS):
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {question[:65]}")
        ctx = retrieve_context(question)
        ans = safe_generate(client, question, ctx)
        user_inputs.append(question)
        responses.append(ans)
        retrieved_contexts.append(ctx)
        reference.append(question)
        print(f"        ✓ ({len(ans)} chars)")
        if i < len(EVAL_QUESTIONS) - 1:
            print(f"        sleeping {SLEEP_BETWEEN}s...")
            time.sleep(SLEEP_BETWEEN)

    print(f"\n  Dataset ready ✓")

    return Dataset.from_dict({
        "user_input":         user_inputs,
        "response":           responses,
        "retrieved_contexts": retrieved_contexts,
        "reference":          reference,
    })


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    print("=" * 60)
    print("  RAGAS Evaluation — Groq API (llama-3.3-70b-versatile)")
    print(f"  Questions: {len(EVAL_QUESTIONS)} | Sleep: {SLEEP_BETWEEN}s between calls")
    print("=" * 60)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    # Groq client for answer generation
    groq_client = Groq(api_key=api_key)

    # RAGAS evaluator — Groq LLM + local embeddings (no extra API key needed)
    print("\n[1/4] Setting up RAGAS evaluator...")
    langchain_llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0,
    )
    # Use local sentence-transformers for embeddings — no API key needed
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    ragas_llm        = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    for metric in [faithfulness, answer_relevancy, context_precision]:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embeddings

    print("  Ready ✓")

    # Build dataset
    dataset = build_dataset(groq_client)

    # Run metrics
    print("\n[3/4] Running RAGAS metrics (sleeping 15s to reset rate limits)...")
    time.sleep(15)

    results = evaluate(
    dataset          = dataset,
    metrics          = [faithfulness, answer_relevancy, context_precision],
    raise_exceptions = False,
    ) 

    results_df = results.to_pandas()
    print("\nDataFrame columns:", results_df.columns.tolist())

    # Detect column names
    q_col = next((c for c in results_df.columns if "input" in c or "question" in c), results_df.columns[0])
    f_col = next((c for c in results_df.columns if "faith" in c), None)
    r_col = next((c for c in results_df.columns if "relev" in c or "relevancy" in c), None)
    p_col = next((c for c in results_df.columns if "precision" in c), None)

    f_score = results_df[f_col].mean() if f_col else float("nan")
    r_score = results_df[r_col].mean() if r_col else float("nan")
    p_score = results_df[p_col].mean() if p_col else float("nan")

    valid   = [s for s in [f_score, r_score, p_score] if str(s) != "nan"]
    overall = sum(valid) / len(valid) if valid else float("nan")

    grade = (
        "Excellent ✓"       if overall >= 0.8 else
        "Good"               if overall >= 0.6 else
        "Fair"               if overall >= 0.4 else
        "Needs improvement"  if str(overall) != "nan" else
        "Could not compute — rate limits affected scoring"
    )

    # Save CSV
    csv_path = OUTPUT_DIR / "ragas_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Print results
    print("\n" + "=" * 60)
    print("  RAGAS RESULTS")
    print("=" * 60)
    print(f"\n  Faithfulness:      {f_score:.3f} / 1.000" if str(f_score) != "nan" else "  Faithfulness:      N/A")
    print(f"  Answer Relevancy:  {r_score:.3f} / 1.000" if str(r_score) != "nan" else "  Answer Relevancy:  N/A")
    print(f"  Context Precision: {p_score:.3f} / 1.000" if str(p_score) != "nan" else "  Context Precision: N/A")
    print(f"\n  Overall:           {overall:.3f} / 1.000" if str(overall) != "nan" else "\n  Overall:           N/A")
    print(f"  Grade:             {grade}")

    # Per-question table
    print(f"\n  Per-question breakdown:")
    print(f"  {'Question':<55}", end="")
    if f_col: print(f" {'Faith':>6}", end="")
    if r_col: print(f" {'Relev':>6}", end="")
    if p_col: print(f" {'Prec':>6}", end="")
    print()
    print("  " + "-" * 75)
    for _, row in results_df.iterrows():
        q = str(row[q_col])[:52] + "..." if len(str(row[q_col])) > 52 else str(row[q_col])
        print(f"  {q:<55}", end="")
        if f_col: print(f" {row[f_col]:>6.3f}", end="")
        if r_col: print(f" {row[r_col]:>6.3f}", end="")
        if p_col: print(f" {row[p_col]:>6.3f}", end="")
        print()

    # Save summary
    summary_path = OUTPUT_DIR / "ragas_summary.txt"
    with open(summary_path, "w") as f:
        f.write("RAGAS EVALUATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model               : {GROQ_MODEL}\n")
        f.write(f"Questions evaluated : {len(EVAL_QUESTIONS)}\n")
        f.write(f"Route types tested  : Analytical, Semantic, Graph, Hybrid\n\n")
        f.write(f"SCORES:\n")
        f.write(f"  Faithfulness      : {f_score:.3f}\n" if str(f_score) != "nan" else "  Faithfulness      : N/A\n")
        f.write(f"  Answer Relevancy  : {r_score:.3f}\n" if str(r_score) != "nan" else "  Answer Relevancy  : N/A\n")
        f.write(f"  Context Precision : {p_score:.3f}\n" if str(p_score) != "nan" else "  Context Precision : N/A\n")
        f.write(f"  Overall           : {overall:.3f}\n\n" if str(overall) != "nan" else "  Overall           : N/A\n\n")
        f.write(f"Grade: {grade}\n")

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {summary_path}")
    print("\n" + "=" * 60)
    print("  Done! Use ragas_summary.txt for your slides.")
    print("=" * 60)

    return {
        "faithfulness":     f_score,
        "answer_relevancy": r_score,
        "context_precision": p_score,
        "overall":          overall,
        "grade":            grade,
    }


if __name__ == "__main__":
    run_evaluation()
