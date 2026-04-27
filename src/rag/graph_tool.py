"""
src/rag/graph_tool.py — GraphRAG Tool for Relationship Queries
==============================================================
Builds a NetworkX directed graph from recognition data.
Nodes = role groups, Edges = recognition events between them.

Answers questions like:
  - Which departments recognize each other most?
  - Who are the most connected recognizers?
  - What is the recognition flow between Leadership and Engineering?
  - Which roles are most central in the recognition network?

Usage:
    from src.rag.graph_tool import graph_query
    result = graph_query("which departments recognize each other most?")
"""

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("output/reports/annotated_results.csv")

# Role group classifier — same logic as dashboard.py for consistency
def classify_role(title: str) -> str:
    if not title or pd.isna(title):
        return "Other"
    t = title.lower()
    if any(k in t for k in ["customer service", "customer success", "helpdesk"]):
        return "Customer Service"
    if any(k in t for k in ["engineer", "software", "qa", "technical", "devops", "architect"]):
        return "Engineering"
    if any(k in t for k in ["design", "creative", "ux", "ui", "copywriter", "content"]):
        return "Design / Creative"
    if any(k in t for k in ["account", "finance", "payroll", "buyer", "procurement", "financial"]):
        return "Finance / Admin"
    if any(k in t for k in ["director", "vp", "head of", "chief", "president", "senior director"]):
        return "Leadership"
    if any(k in t for k in ["hr", "hx", "people", "talent", "recruit"]):
        return "People / HR"
    if any(k in t for k in ["market", "brand", "media", "pr ", "communications"]):
        return "Marketing"
    if any(k in t for k in ["product", "program", "project"]):
        return "Product / Program"
    if any(k in t for k in ["data", "analy", "insight", "research"]):
        return "Data / Analytics"
    if any(k in t for k in ["sales", "business develop", "revenue"]):
        return "Sales / BD"
    return "Other"


# ---------------------------------------------------------------------------
# Graph builder — singleton
# ---------------------------------------------------------------------------

_graph: nx.DiGraph | None = None
_df: pd.DataFrame | None  = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        for col in _df.select_dtypes(include="object").columns:
            _df[col] = _df[col].fillna("").astype(str)
        _df["nom_group"] = _df["nominator_title"].apply(classify_role)
        _df["rec_group"]  = _df["recipient_title"].apply(classify_role)
    return _df


def _build_graph() -> nx.DiGraph:
    """
    Build directed graph where:
      - Nodes = role groups
      - Edges = recognition events (nominator_group → recipient_group)
      - Edge weight = number of recognitions
    """
    global _graph
    if _graph is not None:
        return _graph

    df = _load_df()

    G = nx.DiGraph()

    # Count edges
    edge_counts: dict[tuple, int] = defaultdict(int)
    edge_categories: dict[tuple, list] = defaultdict(list)

    for _, row in df.iterrows():
        nom_group = row["nom_group"]
        rec_group  = row["rec_group"]
        category   = row["category_name"]
        edge_counts[(nom_group, rec_group)] += 1
        edge_categories[(nom_group, rec_group)].append(category)

    # Add nodes with attributes
    all_groups = set()
    for (src, tgt) in edge_counts:
        all_groups.add(src)
        all_groups.add(tgt)

    for group in all_groups:
        group_df = df[df["rec_group"] == group]
        G.add_node(
            group,
            total_received=len(group_df),
            top_category=group_df["category_name"].value_counts().index[0] if len(group_df) > 0 else "N/A",
        )

    # Add edges with weights
    for (src, tgt), weight in edge_counts.items():
        cats = edge_categories[(src, tgt)]
        top_cat = max(set(cats), key=cats.count)
        G.add_edge(src, tgt, weight=weight, top_category=top_cat)

    _graph = G
    return _graph


# ---------------------------------------------------------------------------
# Graph query tool
# ---------------------------------------------------------------------------

def graph_query(query: str) -> str:
    """
    Answer relationship/network questions about the recognition graph.

    Args:
        query: Natural language question about relationships

    Returns:
        Formatted string with graph analysis results
    """
    G  = _build_graph()
    df = _load_df()
    q  = query.lower()
    results = []

    # --- Most connected pairs (strongest relationships) ---
    if any(w in q for w in ["most", "strongest", "top", "highest", "between", "each other", "connected"]):
        edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
        n = _extract_n(q, default=10)
        results.append(f"=== Top {n} Strongest Recognition Relationships ===\n")
        for src, tgt, data in edges[:n]:
            results.append(
                f"  {src} → {tgt}: {data['weight']} recognitions "
                f"(top category: {data['top_category']})"
            )
        results.append("")

    # --- Most giving groups (out-degree) ---
    if any(w in q for w in ["give", "gave", "nominate", "most recogni", "who recogni"]):
        out_weights = {
            node: sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
            for node in G.nodes()
        }
        sorted_givers = sorted(out_weights.items(), key=lambda x: x[1], reverse=True)
        n = _extract_n(q, default=5)
        results.append(f"=== Top {n} Recognition Givers (by volume) ===\n")
        for group, total in sorted_givers[:n]:
            results.append(f"  {group}: {total} recognitions given")
        results.append("")

    # --- Most receiving groups (in-degree) ---
    if any(w in q for w in ["receive", "received", "most awarded", "who received", "most recognized"]):
        in_weights = {
            node: sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
            for node in G.nodes()
        }
        sorted_receivers = sorted(in_weights.items(), key=lambda x: x[1], reverse=True)
        n = _extract_n(q, default=5)
        results.append(f"=== Top {n} Recognition Receivers (by volume) ===\n")
        for group, total in sorted_receivers[:n]:
            results.append(f"  {group}: {total} recognitions received")
        results.append("")

    # --- Centrality — most influential nodes ---
    if any(w in q for w in ["central", "influential", "important", "hub", "key"]):
        centrality = nx.betweenness_centrality(G, weight="weight")
        sorted_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        results.append("=== Most Central Role Groups (Network Influence) ===\n")
        for group, score in sorted_central[:5]:
            results.append(f"  {group}: centrality score = {score:.3f}")
        results.append("")

    # --- Self-recognition (same group recognizing itself) ---
    if any(w in q for w in ["self", "same", "within", "internal"]):
        results.append("=== Within-Group Recognition (Self-loops) ===\n")
        for node in G.nodes():
            if G.has_edge(node, node):
                weight = G[node][node]["weight"]
                results.append(f"  {node} → {node}: {weight} internal recognitions")
        results.append("")

    # --- Specific group lookup ---
    group_match = _detect_group(q)
    if group_match:
        results.append(f"=== Recognition Network for '{group_match}' ===\n")

        # What they give
        out_edges = sorted(
            [(tgt, d["weight"], d["top_category"]) for _, tgt, d in G.out_edges(group_match, data=True)],
            key=lambda x: x[1], reverse=True
        )
        if out_edges:
            results.append(f"  Recognizes others:")
            for tgt, weight, cat in out_edges[:5]:
                results.append(f"    → {tgt}: {weight} times (mainly: {cat})")

        # What they receive
        in_edges = sorted(
            [(src, d["weight"], d["top_category"]) for src, _, d in G.in_edges(group_match, data=True)],
            key=lambda x: x[1], reverse=True
        )
        if in_edges:
            results.append(f"\n  Recognized by:")
            for src, weight, cat in in_edges[:5]:
                results.append(f"    ← {src}: {weight} times (mainly: {cat})")
        results.append("")

    # --- Network overview fallback ---
    if not results:
        results.append("=== Recognition Network Overview ===\n")
        results.append(f"  Nodes (role groups): {G.number_of_nodes()}")
        results.append(f"  Edges (relationships): {G.number_of_edges()}")
        results.append(f"  Total recognitions: {sum(d['weight'] for _, _, d in G.edges(data=True))}")
        results.append("")

        # Top 5 relationships
        edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
        results.append("  Top 5 strongest relationships:")
        for src, tgt, data in edges[:5]:
            results.append(f"    {src} → {tgt}: {data['weight']} recognitions")
        results.append("")

        # Degree summary
        results.append("  Role groups by recognition received:")
        in_weights = {
            node: sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
            for node in G.nodes()
        }
        for group, total in sorted(in_weights.items(), key=lambda x: x[1], reverse=True):
            results.append(f"    {group}: {total}")

    return "\n".join(results)


def get_graph_stats() -> dict:
    """
    Return key graph statistics for display in dashboard.
    """
    G  = _build_graph()
    df = _load_df()

    edges      = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    centrality = nx.betweenness_centrality(G, weight="weight")

    return {
        "num_nodes":        G.number_of_nodes(),
        "num_edges":        G.number_of_edges(),
        "total_recognitions": sum(d["weight"] for _, _, d in G.edges(data=True)),
        "top_relationship": f"{edges[0][0]} → {edges[0][1]}" if edges else "N/A",
        "top_relationship_count": edges[0][2]["weight"] if edges else 0,
        "most_central":     max(centrality, key=centrality.get) if centrality else "N/A",
        "all_edges":        [(s, t, d["weight"], d["top_category"]) for s, t, d in edges],
        "centrality":       dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_n(query: str, default: int = 5) -> int:
    word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    match = re.search(r"top\s+(\d+)", query)
    if match:
        return int(match.group(1))
    for word, num in word_nums.items():
        if word in query:
            return num
    return default


def _detect_group(query: str) -> str | None:
    """Check if query mentions a known role group."""
    groups = [
        "Leadership", "Engineering", "Design / Creative", "Customer Service",
        "Finance / Admin", "People / HR", "Marketing",
        "Product / Program", "Data / Analytics", "Sales / BD", "Other"
    ]
    q = query.lower()
    for group in groups:
        if group.lower() in q or group.lower().split(" /")[0] in q:
            return group
    # Common aliases
    aliases = {
        "engineer": "Engineering", "designer": "Design / Creative",
        "leader": "Leadership", "executive": "Leadership",
        "hr": "People / HR", "finance": "Finance / Admin",
        "marketing": "Marketing", "sales": "Sales / BD",
        "data": "Data / Analytics", "product": "Product / Program",
        "customer": "Customer Service",
    }
    for alias, group in aliases.items():
        if alias in q:
            return group
    return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building recognition graph...")
    stats = get_graph_stats()
    print(f"\nGraph built:")
    print(f"  Nodes: {stats['num_nodes']} role groups")
    print(f"  Edges: {stats['num_edges']} relationships")
    print(f"  Total recognitions: {stats['total_recognitions']}")
    print(f"  Strongest relationship: {stats['top_relationship']} ({stats['top_relationship_count']} times)")
    print(f"  Most central group: {stats['most_central']}")

    print("\n" + "="*60)
    print("Test 1: Which departments recognize each other most?")
    print("="*60)
    print(graph_query("which departments recognize each other most?"))

    print("="*60)
    print("Test 2: How connected is Leadership?")
    print("="*60)
    print(graph_query("how does leadership recognize others?"))

    print("="*60)
    print("Test 3: Network overview")
    print("="*60)
    print(graph_query("give me a network overview"))
