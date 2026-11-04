"""
src/rag/schema.py - Load any recognition dataset into the standard schema
=========================================================================
Every tool reads data through load_dataset(), never from the CSV directly.
The column mapping and department rules live in config/schema.yaml, so a new
company's data only needs a new config file, not code changes.

Standard columns after loading:
    message, award, giver_title, receiver_title,
    giver_department, receiver_department, date (if mapped),
    category, subcategory
"""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "config/schema.yaml"))

STANDARD_COLUMNS = [
    "message", "award", "giver_title", "receiver_title",
    "giver_department", "receiver_department", "date",
    "category", "subcategory",
]
REQUIRED_COLUMNS = ["message", "giver_title", "receiver_title", "category", "subcategory"]


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    unknown = set(cfg.get("columns", {})) - set(STANDARD_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown standard columns in {SCHEMA_PATH}: {sorted(unknown)}")
    missing = [c for c in REQUIRED_COLUMNS if not cfg["columns"].get(c)]
    if missing:
        raise ValueError(f"{SCHEMA_PATH} must map these required columns: {missing}")
    return cfg


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Short keywords (3 letters or fewer, like 'ui', 'hr', 'vp') must match a
    whole word. Longer ones match at the start of a word, so stems like
    'analy' still catch 'Analyst' and 'Analytics', but never mid-word."""
    k = re.escape(keyword.strip().lower())
    return re.compile(rf"\b{k}\b" if len(keyword.strip()) <= 3 else rf"\b{k}")


@lru_cache(maxsize=1)
def _compiled_rules() -> tuple:
    cfg = load_schema()
    rules = tuple(
        (dept, tuple(_keyword_pattern(k) for k in keywords))
        for dept, keywords in cfg.get("department_rules", {}).items()
    )
    overrides = {t.strip().lower(): d for t, d in (cfg.get("department_overrides") or {}).items()}
    return rules, overrides


def classify_title(title: str) -> str:
    """Assign a department from a job title.
    Exact-title overrides win; otherwise the first matching rule in schema.yaml."""
    cfg = load_schema()
    default = cfg.get("default_department", "Other")
    if not title:
        return default
    rules, overrides = _compiled_rules()
    t = title.strip().lower()
    if t in overrides:
        return overrides[t]
    for department, patterns in rules:
        if any(p.search(t) for p in patterns):
            return department
    return default


def schema_fingerprint() -> str:
    """Short hash of everything in schema.yaml that affects the data.
    Saved graphs are tied to it, so editing the rules forces a rebuild."""
    cfg = load_schema()
    relevant = {k: cfg.get(k) for k in
                ("dataset", "columns", "department_rules", "department_overrides", "default_department")}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Read the CSV named in schema.yaml and return it in the standard schema.
    Cached: callers must .copy() before modifying the returned frame."""
    cfg = load_schema()
    path = Path(cfg["dataset"]["path"])
    raw = pd.read_csv(path)

    mapping = {std: src for std, src in cfg["columns"].items() if src}
    absent = [src for src in mapping.values() if src not in raw.columns]
    if absent:
        raise ValueError(f"Columns named in {SCHEMA_PATH} are missing from {path}: {absent}")

    df = pd.DataFrame({std: raw[src] for std, src in mapping.items()})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in df.columns:
        if col != "date" and df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)
    if "award" not in df.columns:
        df["award"] = ""

    # Departments: use real columns when mapped, otherwise derive from titles
    for side in ("giver", "receiver"):
        dept_col = f"{side}_department"
        if dept_col not in df.columns:
            df[dept_col] = df[f"{side}_title"].map(classify_title)
        else:
            df[dept_col] = df[dept_col].replace("", cfg.get("default_department", "Other"))

    return df


def dataset_name() -> str:
    return load_schema()["dataset"].get("name", "dataset")


def describe_dataset() -> str:
    """Plain-text facts about the loaded data, for system prompts.
    Computed at load time, so prompts never hardcode counts."""
    df = load_dataset()
    lines = [
        f"- {len(df):,} recognition records",
        f"- {df['category'].nunique()} main categories: {', '.join(sorted(df['category'].unique()))}",
        f"- {df['subcategory'].nunique()} subcategories",
        f"- {pd.concat([df['giver_department'], df['receiver_department']]).nunique()} departments",
        f"- {pd.concat([df['giver_title'], df['receiver_title']]).nunique():,} distinct job titles",
    ]
    if "date" in df.columns and df["date"].notna().any():
        lines.append(f"- Dates from {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Department report: review classification quality after editing the rules
#   python -m src.rag.schema
# ---------------------------------------------------------------------------

SNAPSHOT_PATH = Path("output/reports/department_snapshot.csv")


def _classify_substring(title: str) -> str:
    """The original matching method (keyword anywhere in the title, with the
    old "pr " workaround). Used only for the first report, before a snapshot
    exists, so you can see what the word-boundary change moved."""
    cfg = load_schema()
    t = (title or "").lower()
    for department, keywords in cfg.get("department_rules", {}).items():
        if any((k + " " if k == "pr" else k) in t for k in keywords):
            return department
    return cfg.get("default_department", "Other")


def department_report() -> str:
    """Departments per title, the biggest unclassified titles, and every title
    whose department changed since the last report. Run after editing rules."""
    df = load_dataset()
    default = load_schema().get("default_department", "Other")
    titles = pd.concat([df["giver_title"], df["receiver_title"]])
    counts = titles.value_counts()
    depts = pd.Series({t: classify_title(t) for t in counts.index}, name="department")

    lines = ["=== Titles per department (records in brackets) ===", ""]
    for dept, group in depts.groupby(depts):
        lines.append(f"  {dept}: {len(group)} titles [{int(counts[group.index].sum()):,} records]")

    unclassified = counts[depts[depts == default].index]
    lines += ["", f"=== Most common titles in '{default}' (add rules or overrides) ===", ""]
    lines += [f"  {t} [{n}]" for t, n in unclassified.head(20).items()]

    if SNAPSHOT_PATH.exists():
        previous = pd.read_csv(SNAPSHOT_PATH, index_col=0)["department"]
        source = "since the last report"
    else:
        previous = pd.Series({t: _classify_substring(t) for t in counts.index})
        source = "vs. the original substring matching"
    changed = [(t, previous[t], d) for t, d in depts.items() if t in previous and previous[t] != d]
    lines += ["", f"=== {len(changed)} titles changed {source} ===", ""]
    lines += [f"  {t}: {old} -> {new}" for t, old, new in sorted(changed)]

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    depts.to_csv(SNAPSHOT_PATH)
    return "\n".join(lines)


if __name__ == "__main__":
    print(department_report())
