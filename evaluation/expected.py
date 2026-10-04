"""
evaluation/expected.py - Correct answers for the evaluation, computed from the data
==================================================================================
Each check returns what a correct answer must contain:
  numbers    every value must appear in the answer (at the answer's precision)
  names_all  every name must appear (case-insensitive)
  names_any  for each group, at least one name must appear (used for ties)

This is deliberately separate code from the agent's tools: plain pandas and
networkx written fresh here. If expected values came from the tools
themselves, a tool bug would make its own wrong answer look correct.
The one shared piece is load_dataset(), which applies the column mapping and
department rules: those are definitions, not calculations.

Print every expected answer for review:
    python -m evaluation.expected
"""

import networkx as nx
import pandas as pd

from src.rag.schema import load_dataset


def _df() -> pd.DataFrame:
    return load_dataset()


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 1)


def _tied_top(counts: pd.Series) -> list[str]:
    """All names tied for the highest count."""
    return sorted(counts[counts == counts.max()].index.tolist())


def _tied_bottom(counts: pd.Series) -> list[str]:
    return sorted(counts[counts == counts.min()].index.tolist())


# ---------------------------------------------------------------------------
# Group A: analytics
# ---------------------------------------------------------------------------

def category_counts():
    c = _df()["category"].value_counts()
    return {"numbers": [int(v) for v in c.values]}


def least_category():
    c = _df()["category"].value_counts()
    return {"names_any": [_tied_bottom(c)], "numbers": [int(c.min())]}


def share_operational_excellence():
    df = _df()
    n = int((df["category"] == "Operational Excellence & Delivery").sum())
    return {"numbers": [_pct(n, len(df))]}


def top5_subcategories():
    c = _df()["subcategory"].value_counts()
    fifth = c.iloc[4]
    clear = c[c > fifth].index.tolist()               # definitely in the top 5
    tied = sorted(c[c == fifth].index.tolist())       # compete for the last slots
    slots = 5 - len(clear)
    if len(tied) <= slots:          # every tied name fits in the top 5
        return {"names_all": clear + tied}
    return {"names_all": clear, "names_any": [tied]}   # ties compete for the last slots


def least_subcategory():
    c = _df()["subcategory"].value_counts()
    return {"names_any": [_tied_bottom(c)]}


def top_giver_title():
    return {"names_any": [_tied_top(_df()["giver_title"].value_counts())]}


def top_receiver_title():
    return {"names_any": [_tied_top(_df()["receiver_title"].value_counts())]}


def top_award_title():
    awards = _df()["award"]
    return {"names_any": [_tied_top(awards[awards != ""].value_counts())]}


def senior_software_engineer_received():
    df = _df()
    n = int(df["receiver_title"].str.lower().str.contains("senior software engineer").sum())
    return {"numbers": [n]}


def top_giving_department():
    c = _df()["giver_department"].value_counts()
    return {"names_any": [_tied_top(c)], "numbers": [int(c.max())]}


def _dept_edges() -> pd.Series:
    return _df().groupby(["giver_department", "receiver_department"]).size()


def top_mutual_pair():
    e = _dept_edges()
    pairs = {}
    for (a, b), w in e.items():
        if a < b and (b, a) in e.index:
            pairs[(a, b)] = int(w + e[(b, a)])
    best = max(pairs.values())
    winners = [p for p, w in pairs.items() if w == best]
    a, b = winners[0]
    out = {"numbers": [best]}
    if len(winners) == 1:
        out["names_all"] = [a, b]
    return out


def most_central_department():
    # Betweenness with distance = 1 / count: strong links are short routes.
    # Within-department loops are not routes between departments.
    G = nx.DiGraph()
    for (a, b), w in _dept_edges().items():
        if a != b:
            G.add_edge(a, b, distance=1.0 / w)
    scores = pd.Series(nx.betweenness_centrality(G, weight="distance"))
    return {"names_any": [_tied_top(scores.round(6))]}


def strongest_cross_flow():
    e = _dept_edges()
    cross = e[[a != b for a, b in e.index]]
    best = int(cross.max())
    winners = cross[cross == best].index.tolist()
    out = {"numbers": [best]}
    if len(winners) == 1:
        out["names_all"] = list(winners[0])
    return out


def finance_top_outgoing():
    e = _dept_edges()
    out_edges = e.loc["Finance / Admin"]
    out_edges = out_edges[out_edges.index != "Finance / Admin"]
    return {"names_any": [_tied_top(out_edges)], "numbers": [int(out_edges.max())]}


def top_bridge_role():
    df = _df()
    # A title's department is the one it appears under most often
    both = pd.concat([
        df[["giver_title", "giver_department"]].set_axis(["title", "dept"], axis=1),
        df[["receiver_title", "receiver_department"]].set_axis(["title", "dept"], axis=1),
    ])
    title_dept = both.groupby("title")["dept"].agg(lambda s: s.value_counts().index[0])
    df = df.assign(target_dept=df["receiver_title"].map(title_dept))
    stats = df.groupby("giver_title").agg(
        reached=("target_dept", "nunique"), given=("target_dept", "size"),
    )
    stats = stats[stats["reached"] > 1]
    top = stats.sort_values(["reached", "given"], ascending=False)
    best = top.iloc[0]
    tied = top[(top["reached"] == best["reached"]) & (top["given"] == best["given"])]
    return {"names_any": [sorted(tied.index.tolist())], "numbers": [int(best["reached"])]}


# ---------------------------------------------------------------------------
# Group C reuses some of the above
# ---------------------------------------------------------------------------

def top_subcategory():
    return {"names_any": [_tied_top(_df()["subcategory"].value_counts())]}


CHECKS = {
    name: fn for name, fn in globals().items()
    if callable(fn) and not name.startswith("_") and fn.__module__ == __name__
    and name not in ("load_dataset",)
}


def expected_for(check: str) -> dict:
    return CHECKS[check]()


if __name__ == "__main__":
    for name, fn in CHECKS.items():
        print(f"{name}: {fn()}")
