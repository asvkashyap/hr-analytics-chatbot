"""
src/rag/tools.py — Agent Tools for HR Recognition Agentic RAG
=============================================================
Three tools the agent can call:

  1. vector_search      — semantic search over ChromaDB
  2. pandas_analytics   — counts, rankings, distributions
  3. substitution       — find roles similar to a target role

Columns: message, award_title, recipient_title, nominator_title,
         batch_id, category_id, category_name, subcategory_id, subcategory_name
"""

import re
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH       = Path("output/reports/annotated_results.csv")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "recognition_records"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 20

# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across all tool calls
# ---------------------------------------------------------------------------

_df: pd.DataFrame | None = None
_embedder: SentenceTransformer | None = None
_collection = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        for col in _df.select_dtypes(include="object").columns:
            _df[col] = _df[col].fillna("").astype(str)
    return _df


def _load_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _load_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Tool 1 — Vector Search
# ---------------------------------------------------------------------------

def vector_search(query: str, k: int = TOP_K) -> str:
    """
    Semantic search over ChromaDB recognition records.
    Best for: qualitative questions, themes, examples, tone, patterns.

    Args:
        query: Natural language question
        k:     Number of results to retrieve

    Returns:
        Formatted string with top matching records
    """
    embedder   = _load_embedder()
    collection = _load_collection()

    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant records found for this query."

    lines = [f"=== Top {len(docs)} Semantic Matches for: '{query}' ===\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        relevance = round((1 - dist) * 100, 1)
        lines.append(f"[{i}] Relevance: {relevance}%")
        lines.append(f"    Category:    {meta.get('category_name', 'N/A')}")
        lines.append(f"    Subcategory: {meta.get('subcategory_name', 'N/A')}")
        lines.append(f"    Recipient:   {meta.get('recipient_title', 'N/A')}")
        lines.append(f"    Award:       {meta.get('award_title', 'N/A')}")
        # Extract just the message part from the chunk
        msg_start = doc.find("Message: ")
        if msg_start != -1:
            msg = doc[msg_start + 9: msg_start + 9 + 250].strip()
            lines.append(f"    Message:     {msg}...")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — Pandas Analytics
# ---------------------------------------------------------------------------

def pandas_analytics(query: str) -> str:
    """
    Answer analytical/statistical questions using pandas directly on the CSV.
    Best for: counts, rankings, distributions, percentages, top N lists.

    Args:
        query: Natural language question

    Returns:
        Formatted string with analytics results
    """
    df = _load_df().copy()
    q  = query.lower()
    results = []

    # --- Detect what the user is asking about ---

    # Theme/topic questions: "what are people recognized for", "recognition themes",
    # "most common topics", "what do people get recognized for"
    is_theme_query = any(w in q for w in [
        "recognized for", "recognised for", "recognition theme", "recognition topic",
        "recognition pattern", "most recognized", "most recognised",
        "what do people", "what are people", "themes", "topics",
        "behaviors", "behaviours", "what kind of recognition",
    ])

    is_category_query = any(w in q for w in [
        "category", "categories", "distribution", "breakdown", "overview",
    ])

    is_subcategory_query = any(w in q for w in [
        "subcategory", "subcategories", "sub category", "sub-category",
    ])

    is_recipient_query = any(w in q for w in [
        "who received", "top performer", "most awarded",
        "who gets", "who is recognized", "who is recognised",
    ])

    is_nominator_query = any(w in q for w in [
        "nominator", "who gave", "who recognized", "giver",
        "who nominated", "who gives", "who nominates",
    ])

    is_award_query = any(w in q for w in [
        "award title", "award type", "award name",
    ])

    is_gap_query = any(w in q for w in [
        "gap", "gaps", "under-recognized", "underrecognized", "least",
        "lowest", "missing", "weak", "neglected", "blind spot",
    ])

    is_give_receive_query = any(w in q for w in [
        "give vs receive", "give and receive", "give versus receive",
        "giving vs receiving", "give more", "receive more",
    ])

    # --- Theme/topic questions → show category + subcategory breakdown ---
    if is_theme_query or is_category_query:
        counts = df["category_name"].value_counts()
        pcts   = (counts / len(df) * 100).round(1)
        results.append("=== Category Distribution (What People Are Recognized For) ===")
        for cat, cnt in counts.items():
            results.append(f"  {cat}: {cnt} records ({pcts[cat]}%)")
        results.append(f"  Total: {len(df)} records\n")

        # Also show top subcategories for theme queries
        if is_theme_query:
            sub_counts = df["subcategory_name"].value_counts().head(10)
            results.append("=== Top 10 Specific Behaviors (Subcategories) ===")
            for sub, cnt in sub_counts.items():
                pct = round(cnt / len(df) * 100, 1)
                # Find parent category
                parent = df[df["subcategory_name"] == sub]["category_name"].iloc[0]
                results.append(f"  {sub} [{parent}]: {cnt} ({pct}%)")
            results.append("")

    # --- Subcategory distribution ---
    if is_subcategory_query and not is_theme_query:
        counts = df["subcategory_name"].value_counts().head(15)
        results.append("=== Top 15 Subcategories ===")
        for sub, cnt in counts.items():
            pct = round(cnt / len(df) * 100, 1)
            parent = df[df["subcategory_name"] == sub]["category_name"].iloc[0]
            results.append(f"  {sub} [{parent}]: {cnt} ({pct}%)")
        results.append("")

    # --- Recognition gaps ---
    if is_gap_query:
        # Show least-represented categories
        cat_counts = df["category_name"].value_counts()
        results.append("=== Recognition Gaps — Categories by Volume ===")
        for cat in cat_counts.index[::-1]:  # reverse = smallest first
            cnt = cat_counts[cat]
            pct = round(cnt / len(df) * 100, 1)
            results.append(f"  {cat}: {cnt} records ({pct}%)")
        results.append("")

        # Show bottom subcategories
        sub_counts = df["subcategory_name"].value_counts()
        bottom_10 = sub_counts.tail(10)
        results.append("=== Bottom 10 Subcategories (Potential Blind Spots) ===")
        for sub in reversed(bottom_10.index):
            cnt = sub_counts[sub]
            pct = round(cnt / len(df) * 100, 1)
            parent = df[df["subcategory_name"] == sub]["category_name"].iloc[0]
            results.append(f"  {sub} [{parent}]: {cnt} ({pct}%)")
        results.append("")

    # --- Give vs receive ---
    if is_give_receive_query:
        give_counts = df["nominator_title"].value_counts().head(10)
        recv_counts = df["recipient_title"].value_counts().head(10)
        results.append("=== Top 10 Givers (by title) ===")
        for title, cnt in give_counts.items():
            results.append(f"  {title}: {cnt} given")
        results.append("")
        results.append("=== Top 10 Receivers (by title) ===")
        for title, cnt in recv_counts.items():
            results.append(f"  {title}: {cnt} received")
        results.append("")

    # --- Top recipients ---
    if is_recipient_query and not is_theme_query:
        n = _extract_n(q, default=5)
        counts = df["recipient_title"].value_counts().head(n)
        results.append(f"=== Top {n} Recipients by Title ===")
        for title, cnt in counts.items():
            results.append(f"  {title}: {cnt} awards")
        results.append("")

    # --- Top nominators ---
    if is_nominator_query:
        n = _extract_n(q, default=5)
        counts = df["nominator_title"].value_counts().head(n)
        results.append(f"=== Top {n} Nominators by Title ===")
        for title, cnt in counts.items():
            results.append(f"  {title}: {cnt} nominations")
        results.append("")

    # --- Award types ---
    if is_award_query:
        n = _extract_n(q, default=5)
        counts = df["award_title"].value_counts().head(n)
        results.append(f"=== Top {n} Award Titles ===")
        for title, cnt in counts.items():
            results.append(f"  {title}: {cnt}")
        results.append("")

    # --- Role specific lookup ---
    role_match = _detect_role(q, df)
    if role_match:
        role_df = df[
            df["recipient_title"].str.lower().str.contains(role_match, na=False) |
            df["nominator_title"].str.lower().str.contains(role_match, na=False)
        ]
        if not role_df.empty:
            results.append(f"=== Stats for '{role_match}' ===")
            results.append(f"  Total records: {len(role_df)}")
            rec_count = df[df["recipient_title"].str.lower().str.contains(role_match, na=False)].shape[0]
            nom_count = df[df["nominator_title"].str.lower().str.contains(role_match, na=False)].shape[0]
            results.append(f"  As recipient: {rec_count} awards")
            results.append(f"  As nominator: {nom_count} nominations")
            results.append("  Category breakdown:")
            for cat, cnt in role_df["category_name"].value_counts().items():
                pct = round(cnt / len(role_df) * 100, 1)
                results.append(f"    {cat}: {cnt} ({pct}%)")
            results.append("  Top 5 subcategories:")
            for sub, cnt in role_df["subcategory_name"].value_counts().head(5).items():
                results.append(f"    {sub}: {cnt}")
            results.append("")

    # --- Fallback: show category + subcategory overview ---
    if not results:
        counts = df["category_name"].value_counts()
        pcts = (counts / len(df) * 100).round(1)
        results.append("=== Recognition Overview ===")
        results.append(f"  Total records: {len(df)}")
        results.append(f"  Categories: {df['category_name'].nunique()}")
        results.append(f"  Subcategories: {df['subcategory_name'].nunique()}")
        results.append(f"  Unique recipient titles: {df['recipient_title'].nunique()}")
        results.append(f"  Unique nominator titles: {df['nominator_title'].nunique()}")
        results.append("\n  Category breakdown:")
        for cat, cnt in counts.items():
            results.append(f"    {cat}: {cnt} records ({pcts[cat]}%)")
        results.append("\n  Top 10 subcategories:")
        for sub, cnt in df["subcategory_name"].value_counts().head(10).items():
            pct = round(cnt / len(df) * 100, 1)
            results.append(f"    {sub}: {cnt} ({pct}%)")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Tool 3 — Substitution
# ---------------------------------------------------------------------------

def substitution(query: str) -> str:
    """
    Find roles with similar contribution patterns to a target role.
    Best for: 'who could replace X', 'similar to X', 'if X left'.

    Args:
        query: Natural language query mentioning a role

    Returns:
        Ranked list of similar roles with similarity scores
    """
    df         = _load_df()
    embedder   = _load_embedder()
    collection = _load_collection()

    target_role = _extract_target_role(query, df)

    if not target_role:
        return (
            "Could not identify a specific role in your query.\n"
            "Try: 'Who could replace a Director of Engineering?'"
        )

    # Get all records for the target role
    target_df = df[
        df["recipient_title"].str.lower().str.contains(target_role.lower(), na=False)
    ]

    if target_df.empty:
        return f"No records found for role matching '{target_role}'."

    # Build a profile text combining all their recognition
    profile_parts = []
    for _, row in target_df.iterrows():
        profile_parts.append(
            f"{row['category_name']} - {row['subcategory_name']}: {row['message'][:100]}"
        )
    profile_text = " | ".join(profile_parts[:10])

    # Embed the profile
    profile_embedding = embedder.encode([profile_text])[0].tolist()

    # Find similar records
    results = collection.query(
        query_embeddings=[profile_embedding],
        n_results=50,
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Aggregate by role, excluding the target
    role_scores: dict[str, list[float]] = {}
    for meta, dist in zip(metadatas, distances):
        role = meta.get("recipient_title", "")
        if role and target_role.lower() not in role.lower():
            role_scores.setdefault(role, []).append(1 - dist)

    if not role_scores:
        return f"No similar roles found for '{target_role}'."

    # Rank by average similarity
    ranked = sorted(
        [(role, round(float(np.mean(scores)) * 100, 1)) for role, scores in role_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    lines = [
        f"=== Roles Most Similar to '{target_role}' ===",
        f"Based on {len(target_df)} recognition records\n",
        "Target role category breakdown:",
    ]
    for cat, cnt in target_df["category_name"].value_counts().items():
        lines.append(f"  {cat}: {cnt} awards")

    lines.append("\nTop 10 most similar roles:")
    for rank, (role, score) in enumerate(ranked, 1):
        lines.append(f"  {rank}. {role} — {score}% similarity")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_n(query: str, default: int = 5) -> int:
    """Extract a number from query like 'top 10' or 'top five'."""
    word_nums = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    match = re.search(r"top\s+(\d+)", query)
    if match:
        return int(match.group(1))
    for word, num in word_nums.items():
        if word in query:
            return num
    return default


def _detect_role(query: str, df: pd.DataFrame) -> str | None:
    """Check if query mentions a known role fragment."""
    all_roles = pd.concat([
        df["recipient_title"], df["nominator_title"]
    ]).str.lower().unique()

    q = query.lower()
    best, best_len = None, 0
    for role in all_roles:
        if role and len(role) > 3 and role in q and len(role) > best_len:
            best = role
            best_len = len(role)
    return best


def _extract_target_role(query: str, df: pd.DataFrame) -> str | None:
    """Extract a role name from substitution queries."""
    patterns = [
        r"replace\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.|,)",
        r"if\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)\s+(?:left|quit|resigned|leaving)",
        r"similar to\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.|,)",
        r"substitute\s+(?:for\s+)?(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.|,)",
        r"backup\s+(?:for\s+)?(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.|,)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: match any known role title in the query
    all_roles = df["recipient_title"].str.lower().unique()
    q = query.lower()
    for role in sorted(all_roles, key=len, reverse=True):
        if role and len(role) > 5 and role in q:
            return role

    return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: What are people recognized for?")
    print("=" * 60)
    print(pandas_analytics("What are people most recognized for?"))

    print("\n" + "=" * 60)
    print("Test 2: Recognition gaps")
    print("=" * 60)
    print(pandas_analytics("Where are the recognition gaps?"))

    print("\n" + "=" * 60)
    print("Test 3: Category distribution")
    print("=" * 60)
    print(pandas_analytics("show me category distribution"))

    print("\n" + "=" * 60)
    print("Test 4: Vector search")
    print("=" * 60)
    print(vector_search("teamwork and collaboration", k=3))

    print("\n" + "=" * 60)
    print("Test 5: Substitution")
    print("=" * 60)
    print(substitution("who could replace a Director of Engineering?"))
