"""
src/rag/neo4j_tool.py — Neo4j Cypher Query Tool
=================================================
Answers rich relationship questions using Cypher queries
that are impossible with standard RAG or NetworkX.

Queries available:
  - Top recognition relationships between groups
  - Most connected / central role groups
  - Recognition paths between two departments
  - Mutual recognition pairs (bidirectional)
  - Bridge roles (span multiple departments)
  - Recognition chains (multi-hop paths)
  - Group-specific network analysis
  - Overall graph statistics

Run standalone:
    python -m src.rag.neo4j_tool
"""

import sys
import os
import re
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Singleton driver
# ---------------------------------------------------------------------------

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        uri      = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        _driver  = GraphDatabase.driver(uri, auth=(username, password))
    return _driver


def _run_query(cypher: str, params: dict = {}) -> list:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [dict(record) for record in result]


# ---------------------------------------------------------------------------
# Neo4j query tool
# ---------------------------------------------------------------------------

def neo4j_query(query: str) -> str:
    """
    Answer relationship/network questions using Cypher queries on Neo4j.
    Handles: cross-department flows, centrality, paths, mutual recognition,
             bridge roles, recognition chains, group analysis.

    Args:
        query: Natural language question about relationships

    Returns:
        Formatted string with graph analysis results
    """
    q       = query.lower()
    results = []

    # --- Strongest relationships between groups ---
    if any(w in q for w in ["strongest", "most", "top", "highest", "between", "each other", "flow"]):
        n = _extract_n(q, default=10)
        records = _run_query(
            """
            MATCH (a:RoleGroup)-[r:RECOGNIZED]->(b:RoleGroup)
            RETURN a.name AS from_group, b.name AS to_group,
                   r.count AS count, r.top_category AS top_category
            ORDER BY r.count DESC
            LIMIT $n
            """,
            {"n": n}
        )
        results.append(f"=== Top {n} Strongest Recognition Relationships ===\n")
        for rec in records:
            arrow = "↔" if rec["from_group"] == rec["to_group"] else "→"
            results.append(
                f"  {rec['from_group']} {arrow} {rec['to_group']}: "
                f"{rec['count']} recognitions (mainly: {rec['top_category']})"
            )
        results.append("")

    # --- Mutual recognition (bidirectional pairs) ---
    if any(w in q for w in ["mutual", "bidirectional", "each other", "both ways", "reciprocal"]):
        records = _run_query(
            """
            MATCH (a:RoleGroup)-[r1:RECOGNIZED]->(b:RoleGroup)-[r2:RECOGNIZED]->(a)
            WHERE id(a) < id(b)
            RETURN a.name AS group1, b.name AS group2,
                   r1.count AS a_to_b, r2.count AS b_to_a,
                   r1.count + r2.count AS total
            ORDER BY total DESC
            """
        )
        results.append("=== Mutual Recognition Pairs (Both Directions) ===\n")
        for rec in records:
            results.append(
                f"  {rec['group1']} ↔ {rec['group2']}: "
                f"{rec['total']} total "
                f"({rec['a_to_b']} + {rec['b_to_a']})"
            )
        results.append("")

    # --- Most central groups (PageRank-style degree) ---
    if any(w in q for w in ["central", "influential", "hub", "important", "key player"]):
        records = _run_query(
            """
            MATCH (g:RoleGroup)
            OPTIONAL MATCH (g)-[out:RECOGNIZED]->()
            OPTIONAL MATCH ()-[in:RECOGNIZED]->(g)
            WITH g,
                 coalesce(sum(out.count), 0) AS given,
                 coalesce(sum(in.count), 0)  AS received
            RETURN g.name AS group,
                   given, received,
                   given + received AS total_connections
            ORDER BY total_connections DESC
            """
        )
        results.append("=== Most Central Role Groups ===\n")
        for rec in records:
            results.append(
                f"  {rec['group']}: {rec['total_connections']} total "
                f"(gave {rec['given']}, received {rec['received']})"
            )
        results.append("")

    # --- Cross-team relationships (excluding self-loops) ---
    if any(w in q for w in ["cross", "between department", "cross team", "cross department", "different"]):
        records = _run_query(
            """
            MATCH (a:RoleGroup)-[r:RECOGNIZED]->(b:RoleGroup)
            WHERE a.name <> b.name
            RETURN a.name AS from_group, b.name AS to_group,
                   r.count AS count, r.top_category AS top_category
            ORDER BY r.count DESC
            LIMIT 15
            """
        )
        results.append("=== Cross-Department Recognition (No Self-Loops) ===\n")
        for rec in records:
            results.append(
                f"  {rec['from_group']} → {rec['to_group']}: "
                f"{rec['count']} (mainly: {rec['top_category']})"
            )
        results.append("")

    # --- Recognition paths between two specific groups ---
    group_pair = _extract_group_pair(q)
    if group_pair:
        src, tgt = group_pair
        records = _run_query(
            """
            MATCH path = shortestPath(
                (a:RoleGroup {name: $src})-[:RECOGNIZED*..5]->(b:RoleGroup {name: $tgt})
            )
            RETURN [node IN nodes(path) | node.name] AS path_nodes,
                   length(path) AS hops
            LIMIT 3
            """,
            {"src": src, "tgt": tgt}
        )
        if records:
            results.append(f"=== Recognition Path: {src} → {tgt} ===\n")
            for rec in records:
                path_str = " → ".join(rec["path_nodes"])
                results.append(f"  {path_str} ({rec['hops']} hops)")
        else:
            results.append(f"  No path found between {src} and {tgt}")
        results.append("")

    # --- Top individual role bridges (span multiple groups) ---
    if any(w in q for w in ["bridge", "span", "connect", "link", "individual"]):
        records = _run_query(
            """
            MATCH (r:Role)-[:RECOGNIZED]->(other:Role)
            WITH r, count(distinct other.group) AS groups_reached,
                 count(other) AS total_given
            WHERE groups_reached > 1
            RETURN r.title AS role, r.group AS own_group,
                   groups_reached, total_given
            ORDER BY groups_reached DESC, total_given DESC
            LIMIT 10
            """
        )
        results.append("=== Top Bridge Roles (Span Multiple Departments) ===\n")
        for rec in records:
            results.append(
                f"  {rec['role']} ({rec['own_group']}): "
                f"recognized across {rec['groups_reached']} groups "
                f"({rec['total_given']} total given)"
            )
        results.append("")

    # --- Specific group deep dive ---
    group_match = _detect_group(q)
    if group_match and not group_pair:
        records = _run_query(
            """
            MATCH (g:RoleGroup {name: $group})
            OPTIONAL MATCH (g)-[out:RECOGNIZED]->(other:RoleGroup)
            OPTIONAL MATCH (src:RoleGroup)-[in:RECOGNIZED]->(g)
            RETURN g.name AS group,
                   g.total_received AS total_received,
                   g.top_category AS top_category,
                   collect(distinct {to: other.name, count: out.count}) AS gives_to,
                   collect(distinct {from: src.name, count: in.count}) AS receives_from
            """,
            {"group": group_match}
        )
        if records:
            rec = records[0]
            results.append(f"=== Neo4j Analysis: {group_match} ===\n")
            results.append(f"  Total received: {rec['total_received']}")
            results.append(f"  Top category:   {rec['top_category']}")

            gives = sorted(
                [x for x in rec["gives_to"] if x["to"] and x["count"]],
                key=lambda x: x["count"] or 0, reverse=True
            )[:5]
            if gives:
                results.append(f"\n  Recognizes:")
                for g in gives:
                    results.append(f"    → {g['to']}: {g['count']} times")

            receives = sorted(
                [x for x in rec["receives_from"] if x["from"] and x["count"]],
                key=lambda x: x["count"] or 0, reverse=True
            )[:5]
            if receives:
                results.append(f"\n  Recognized by:")
                for r in receives:
                    results.append(f"    ← {r['from']}: {r['count']} times")
        results.append("")

    # --- Overall graph stats fallback ---
    if not results:
        stats = _run_query(
            """
            MATCH (g:RoleGroup)-[r:RECOGNIZED]->(h:RoleGroup)
            RETURN count(distinct g) AS num_groups,
                   count(r) AS num_relationships,
                   sum(r.count) AS total_recognitions
            """
        )
        top_edges = _run_query(
            """
            MATCH (a:RoleGroup)-[r:RECOGNIZED]->(b:RoleGroup)
            RETURN a.name AS from_group, b.name AS to_group, r.count AS count
            ORDER BY r.count DESC LIMIT 5
            """
        )
        if stats:
            s = stats[0]
            results.append("=== Neo4j Recognition Network Overview ===\n")
            results.append(f"  Role groups:       {s['num_groups']}")
            results.append(f"  Relationships:     {s['num_relationships']}")
            results.append(f"  Total recognitions:{s['total_recognitions']}")
            results.append("\n  Top 5 relationships:")
            for rec in top_edges:
                arrow = "↔" if rec["from_group"] == rec["to_group"] else "→"
                results.append(f"    {rec['from_group']} {arrow} {rec['to_group']}: {rec['count']}")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GROUPS = [
    "Leadership", "Engineering", "Design / Creative", "Customer Service",
    "Finance / Admin", "People / HR", "Marketing",
    "Product / Program", "Data / Analytics", "Sales / BD", "Other"
]

GROUP_ALIASES = {
    "engineer": "Engineering", "designer": "Design / Creative",
    "leader": "Leadership", "executive": "Leadership",
    "hr": "People / HR", "finance": "Finance / Admin",
    "marketing": "Marketing", "sales": "Sales / BD",
    "data": "Data / Analytics", "product": "Product / Program",
    "customer": "Customer Service",
}


def _detect_group(query: str) -> str | None:
    q = query.lower()
    for group in GROUPS:
        if group.lower() in q or group.lower().split(" /")[0] in q:
            return group
    for alias, group in GROUP_ALIASES.items():
        if alias in q:
            return group
    return None


def _extract_group_pair(query: str) -> tuple | None:
    """Extract two group names from a query like 'between engineering and marketing'."""
    found = []
    q = query.lower()
    for group in GROUPS:
        if group.lower() in q or group.lower().split(" /")[0] in q:
            found.append(group)
    if len(found) >= 2:
        return (found[0], found[1])
    return None


def _extract_n(query: str, default: int = 10) -> int:
    word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    match = re.search(r"top\s+(\d+)", query)
    if match:
        return int(match.group(1))
    for word, num in word_nums.items():
        if word in query:
            return num
    return default


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "Which departments recognize each other most?",
        "Show me mutual recognition pairs",
        "Who are the most central hubs?",
        "Show me cross-department recognition only",
        "How does Engineering connect to the network?",
        "Who are the bridge roles that span multiple departments?",
        "What is the path between Leadership and Engineering?",
    ]

    print("=" * 60)
    print("  Neo4j Tool — Cypher Query Tests")
    print("=" * 60)

    for q in tests:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)
        print(neo4j_query(q))
