"""
src/rag/graph_tool.py - Recognition network analysis with NetworkX
==================================================================
Two graphs, both built from the standard schema (see schema.py):
  - Department graph: departments are nodes, edge weight = recognitions
    given from one department to another.
  - Role graph: job titles are nodes, same idea at title level.
    Used for bridge roles that span multiple departments.

Built with pandas groupby (fast on large datasets) and cached to disk.
The cache is tied to the data file and the schema settings, so editing
either one triggers a rebuild.

The structured functions (top_relationships, bridge_roles, ...) are the
building blocks. graph_query() keeps the old keyword interface so the
router and agent keep working unchanged.

Usage:
    from src.rag.graph_tool import graph_query
    print(graph_query("which departments recognize each other most?"))
"""

import pickle
import re
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd

from src.rag.schema import load_dataset, load_schema, dataset_name, schema_fingerprint

warnings.filterwarnings("ignore")

CACHE_DIR = Path("output/graphs")
EXACT_CENTRALITY_MAX_NODES = 500   # above this, use sampled betweenness
CENTRALITY_SAMPLE_K = 200

_graphs: dict | None = None


# ---------------------------------------------------------------------------
# Build (groupby, not row loops) and cache
# ---------------------------------------------------------------------------

def _most_common(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Most frequent `value` per group, vectorized (no per-group Python calls)."""
    counts = df.groupby(keys + [value], observed=True).size().reset_index(name="_n")
    return (
        counts.sort_values("_n", ascending=False, kind="stable")
        .drop_duplicates(keys)
        .drop(columns="_n")
    )


def _edge_table(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """One row per (src, dst) pair: recognition count and most common category."""
    weights = df.groupby([src, dst], observed=True).size().reset_index(name="weight")
    top_cat = _most_common(df, [src, dst], "category").rename(columns={"category": "top_category"})
    return weights.merge(top_cat, on=[src, dst])


def _graph_from_edges(edges: pd.DataFrame, src: str, dst: str) -> nx.DiGraph:
    return nx.from_pandas_edgelist(
        edges, source=src, target=dst,
        edge_attr=["weight", "top_category"], create_using=nx.DiGraph,
    )


def _build() -> dict:
    df = load_dataset()
    dept = _graph_from_edges(
        _edge_table(df, "giver_department", "receiver_department"),
        "giver_department", "receiver_department",
    )
    role = _graph_from_edges(
        _edge_table(df, "giver_title", "receiver_title"),
        "giver_title", "receiver_title",
    )
    # Department each title belongs to (most common), for bridge-role output
    both_sides = pd.concat([
        df[["giver_title", "giver_department"]].set_axis(["title", "department"], axis=1),
        df[["receiver_title", "receiver_department"]].set_axis(["title", "department"], axis=1),
    ])
    title_dept = _most_common(both_sides, ["title"], "department").set_index("title")["department"].to_dict()
    nx.set_node_attributes(role, title_dept, "department")
    return {"department": dept, "role": role}


def _load_graphs() -> dict:
    global _graphs
    if _graphs is not None:
        return _graphs

    source = Path(load_schema()["dataset"]["path"])
    # Tied to the data file AND the schema settings: editing department
    # rules or column mappings produces a new fingerprint and a rebuild
    cache = CACHE_DIR / f"{dataset_name()}_{schema_fingerprint()}_graphs.pkl"

    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        with open(cache, "rb") as f:
            _graphs = pickle.load(f)
    else:
        _graphs = _build()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(_graphs, f)
    return _graphs


def _dept_graph() -> nx.DiGraph:
    return _load_graphs()["department"]


def _role_graph() -> nx.DiGraph:
    return _load_graphs()["role"]


# ---------------------------------------------------------------------------
# Structured queries (ready for structured tool calling later)
# ---------------------------------------------------------------------------

def top_relationships(n: int = 10, cross_department_only: bool = False) -> list[dict]:
    edges = [
        {"from": s, "to": t, "count": d["weight"], "top_category": d["top_category"]}
        for s, t, d in _dept_graph().edges(data=True)
        if not (cross_department_only and s == t)
    ]
    return sorted(edges, key=lambda e: e["count"], reverse=True)[:n]


def department_volumes(direction: str = "given") -> list[tuple[str, int]]:
    G = _dept_graph()
    deg = G.out_degree(weight="weight") if direction == "given" else G.in_degree(weight="weight")
    return sorted(deg, key=lambda x: x[1], reverse=True)


def central_departments(n: int = 5) -> list[tuple[str, float]]:
    """Betweenness centrality. NetworkX treats edge weight as DISTANCE, but our
    weights are recognition counts (closeness). Convert to distance = 1 / count,
    so strong connections count as short routes, not long ones."""
    G = _dept_graph().copy()
    G.remove_edges_from(nx.selfloop_edges(G))   # within-department loops are not routes
    for _, _, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]
    if G.number_of_nodes() > EXACT_CENTRALITY_MAX_NODES:
        scores = nx.betweenness_centrality(G, k=CENTRALITY_SAMPLE_K, weight="distance", seed=42)
    else:
        scores = nx.betweenness_centrality(G, weight="distance")
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def within_department() -> list[tuple[str, int]]:
    G = _dept_graph()
    pairs = [(n, G[n][n]["weight"]) for n in G.nodes() if G.has_edge(n, n)]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def mutual_pairs(n: int = 10) -> list[dict]:
    G = _dept_graph()
    out = []
    for a, b, d in G.edges(data=True):
        if a < b and G.has_edge(b, a):
            back = G[b][a]["weight"]
            out.append({"a": a, "b": b, "a_to_b": d["weight"], "b_to_a": back,
                        "total": d["weight"] + back})
    return sorted(out, key=lambda x: x["total"], reverse=True)[:n]


def bridge_roles(n: int = 10) -> list[dict]:
    """Titles that give recognition into the most distinct departments."""
    R = _role_graph()
    rows = []
    for title in R.nodes():
        reached = {R.nodes[t].get("department") for t in R.successors(title)}
        reached.discard(None)
        if len(reached) > 1:
            rows.append({
                "role": title,
                "department": R.nodes[title].get("department", ""),
                "departments_reached": len(reached),
                "total_given": int(R.out_degree(title, weight="weight")),
            })
    rows.sort(key=lambda r: (r["departments_reached"], r["total_given"]), reverse=True)
    return rows[:n]


def department_profile(department: str) -> dict:
    G = _dept_graph()
    if department not in G:
        return {}
    gives = sorted(((t, d["weight"], d["top_category"]) for _, t, d in G.out_edges(department, data=True)),
                   key=lambda x: x[1], reverse=True)
    gets = sorted(((s, d["weight"], d["top_category"]) for s, _, d in G.in_edges(department, data=True)),
                  key=lambda x: x[1], reverse=True)
    return {"gives_to": gives, "receives_from": gets}


def path_between(a: str, b: str) -> list[str] | None:
    try:
        return nx.shortest_path(_dept_graph(), a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ---------------------------------------------------------------------------
# Department detection, driven by the data and schema.yaml
# ---------------------------------------------------------------------------

def _department_names(dept: str) -> set[str]:
    """Ways a department can be named in a question: the full name and each
    part of a compound name ("IT & Security" -> "security", "Supply Chain &
    E-Commerce" -> "supply chain", "e-commerce"). Parts under 3 letters are
    skipped because they collide with ordinary words."""
    d = dept.lower()
    parts = {p.strip() for p in re.split(r"\s*[/&]\s*", d)}
    return {n for n in parts | {d} if len(n) > 2}


def _departments_in(query: str) -> list[str]:
    """Departments named in the query. Only department NAMES count, not the
    title keywords from schema.yaml: keywords like "network" or "test" are
    ordinary words in analytics questions ("network overview")."""
    q = query.lower()
    default = load_schema().get("default_department", "Other")
    hits = {}
    for dept in _dept_graph().nodes():
        if dept == default:   # "other" appears in normal phrases like "each other"
            continue
        positions = [m.start() for n in _department_names(dept)
                     for m in [re.search(rf"\b{re.escape(n)}", q)] if m]
        if positions:
            hits[dept] = min(positions)
    # Keep the order they appear in the query ("path from A to B")
    return sorted(hits, key=hits.get)


def _extract_n(query: str, default: int = 5) -> int:
    match = re.search(r"top\s+(\d+)", query)
    if match:
        return int(match.group(1))
    words = {"three": 3, "five": 5, "ten": 10}
    for w, n in words.items():
        if re.search(rf"\btop {w}\b", query):
            return n
    return default


# ---------------------------------------------------------------------------
# Keyword interface (used by router.py and agent_graph.py)
# ---------------------------------------------------------------------------

def _has(q: str, words: list[str]) -> bool:
    return any(w in q for w in words)


def graph_query(query: str) -> str:
    """Decide the question's intent first, then answer only that.
    Generic tables appear only when no specific intent matches."""
    # "give me ..." is a request, not about giving recognition
    q = re.sub(r"\bgive me\b", "show me", query.lower())
    depts = _departments_in(q)

    intents = {
        "path":    len(depts) >= 2 and _has(q, ["path", "route", "chain", "reach"]),
        "bridge":  _has(q, ["bridge", "span", "multiple departments", "across departments"]),
        "mutual":  _has(q, ["mutual", "bidirectional", "both ways", "reciprocal"]),
        "central": _has(q, ["central", "influential", "hub", "key player", "important"]),
        "within":  _has(q, ["within", "internal", "same department", "self-recogni"]),
        "givers":  _has(q, ["give", "gave", "giver", "nominat", "who recogni"]) and not depts,
        "receivers": _has(q, ["receive", "received", "receiver", "most awarded"]) and not depts,
        # "Cross" only when no department is named; "how does X connect to
        # other departments" is a question about X, answered by its profile
        "cross":   _has(q, ["cross"]) or (_has(q, ["other department", "different department"]) and not depts),
    }
    intents["profile"] = bool(depts) and not intents["path"]
    # "each other" with no department named means mutual + strongest flows
    if "each other" in q and not depts:
        intents["mutual"] = True
    specific = any(intents.values())
    intents["strongest"] = (not specific and _has(q, ["most", "top", "highest", "strongest", "flow", "between"])) \
        or ("each other" in q and not depts) or _has(q, ["strongest"])

    out: list[str] = []
    n = _extract_n(q, default=10)

    if intents["path"]:
        path = path_between(depts[0], depts[1])
        out.append(f"=== Recognition Path: {depts[0]} -> {depts[1]} ===\n")
        out.append(f"  {' -> '.join(path)}" if path else "  No path found")
        out.append("")

    if intents["bridge"]:
        out.append(f"=== Top {n} Bridge Roles (recognize across the most departments) ===\n")
        for r in bridge_roles(n):
            out.append(f"  {r['role']} ({r['department']}): reaches {r['departments_reached']} "
                       f"departments, {r['total_given']} recognitions given")
        out.append("")

    if intents["mutual"]:
        out.append("=== Mutual Recognition Pairs (both directions) ===\n")
        for p in mutual_pairs(10):
            out.append(f"  {p['a']} <-> {p['b']}: {p['total']} total ({p['a_to_b']} + {p['b_to_a']})")
        out.append("")

    if intents["cross"] or intents["strongest"]:
        cross = intents["cross"]
        title = "Cross-Department Relationships" if cross else "Strongest Recognition Relationships"
        out.append(f"=== Top {n} {title} ===\n")
        for e in top_relationships(n, cross_department_only=cross):
            out.append(f"  {e['from']} -> {e['to']}: {e['count']} recognitions (mainly: {e['top_category']})")
        out.append("")

    if intents["givers"]:
        k = _extract_n(q)
        out.append(f"=== Top {k} Departments by Recognition Given ===\n")
        out += [f"  {d}: {total} given" for d, total in department_volumes("given")[:k]]
        out.append("")

    if intents["receivers"]:
        k = _extract_n(q)
        out.append(f"=== Top {k} Departments by Recognition Received ===\n")
        out += [f"  {d}: {total} received" for d, total in department_volumes("received")[:k]]
        out.append("")

    if intents["central"]:
        out.append("=== Most Central Departments (betweenness centrality) ===\n")
        out += [f"  {d}: {score:.3f}" for d, score in central_departments(5)]
        out.append("")

    if intents["within"]:
        out.append("=== Within-Department Recognition ===\n")
        out += [f"  {d} -> {d}: {w}" for d, w in within_department()]
        out.append("")

    if intents["profile"]:
        for d in depts[:2]:
            prof = department_profile(d)
            if not prof:
                continue
            out.append(f"=== Recognition Network for '{d}' ===\n")
            out.append("  Recognizes:")
            out += [f"    -> {t}: {w} (mainly: {c})" for t, w, c in prof["gives_to"][:5]]
            out.append("  Recognized by:")
            out += [f"    <- {s_}: {w} (mainly: {c})" for s_, w, c in prof["receives_from"][:5]]
            out.append("")

    if not out:
        G = _dept_graph()
        out.append("=== Recognition Network Overview ===\n")
        out.append(f"  Departments: {G.number_of_nodes()}")
        out.append(f"  Department relationships: {G.number_of_edges()}")
        out.append(f"  Job titles: {_role_graph().number_of_nodes()}")
        out.append(f"  Total recognitions: {int(G.size(weight='weight'))}")
        out.append("\n  Top 5 relationships:")
        out += [f"    {e['from']} -> {e['to']}: {e['count']}" for e in top_relationships(5)]

    return "\n".join(out)


def get_graph_stats() -> dict:
    """Summary stats for dashboards (same keys as the previous version)."""
    G = _dept_graph()
    edges = top_relationships(n=G.number_of_edges())
    central = dict(central_departments(n=G.number_of_nodes()))
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_recognitions": int(G.size(weight="weight")),
        "top_relationship": f"{edges[0]['from']} → {edges[0]['to']}" if edges else "N/A",
        "top_relationship_count": edges[0]["count"] if edges else 0,
        "most_central": max(central, key=central.get) if central else "N/A",
        "all_edges": [(e["from"], e["to"], e["count"], e["top_category"]) for e in edges],
        "centrality": central,
    }


if __name__ == "__main__":
    tests = [
        "Which departments recognize each other most?",
        "Show me mutual recognition pairs",
        "Who are the most central hubs?",
        "Show me cross-department recognition only",
        "How does Engineering connect to the network?",
        "Who are the bridge roles that span multiple departments?",
        "What is the path between Leadership and Engineering?",
        "Give me a network overview",
        "How does Leadership connect to other departments?",
        "Who gives the most recognition?",
    ]
    for t in tests:
        print("=" * 60)
        print(f"Q: {t}")
        print("-" * 60)
        print(graph_query(t))
