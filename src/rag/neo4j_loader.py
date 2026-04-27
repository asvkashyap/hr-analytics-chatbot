"""
src/rag/neo4j_loader.py — Load Recognition Graph into Neo4j
=============================================================
Creates nodes and relationships in Neo4j AuraDB from annotated_results.csv.

Graph schema:
  Nodes:
    (:RoleGroup {name, total_received, top_category})
    (:Role {title, group})
    (:Category {name})
    (:Subcategory {name, category})

  Relationships:
    (:RoleGroup)-[:RECOGNIZED {count, top_category}]->(:RoleGroup)
    (:Role)-[:RECEIVED {award_title, category, subcategory}]->(:Role)

Run once:
    cd "C:\\Users\\kashy\\OneDrive\\Desktop\\Masters Project"
    conda activate masters-project
    python -m src.rag.neo4j_loader

Re-running wipes and rebuilds cleanly.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from collections import defaultdict
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("output/reports/annotated_results.csv")


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
# Loader
# ---------------------------------------------------------------------------

def load_graph():
    print("=" * 60)
    print("  Neo4j Loader — HR Recognition Graph")
    print("=" * 60)

    # Connect
    uri      = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        raise ValueError("NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD must be set in .env")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print("\n[0] Connected to Neo4j AuraDB ✓")

    # Load data
    print(f"\n[1/5] Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str)

    df["nom_group"] = df["nominator_title"].apply(classify_role)
    df["rec_group"]  = df["recipient_title"].apply(classify_role)
    print(f"      Loaded {len(df)} records ✓")

    with driver.session() as session:

        # Wipe existing graph
        print("\n[2/5] Wiping existing graph...")
        session.run("MATCH (n) DETACH DELETE n")
        print("      Wiped ✓")

        # Create constraints
        print("\n[3/5] Creating constraints and indexes...")
        session.run("CREATE CONSTRAINT role_group_name IF NOT EXISTS FOR (r:RoleGroup) REQUIRE r.name IS UNIQUE")
        session.run("CREATE CONSTRAINT role_title IF NOT EXISTS FOR (r:Role) REQUIRE r.title IS UNIQUE")
        session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
        session.run("CREATE CONSTRAINT subcategory_name IF NOT EXISTS FOR (s:Subcategory) REQUIRE s.name IS UNIQUE")
        print("      Constraints created ✓")

        # Create Category nodes
        print("\n[4/5] Creating nodes...")
        categories = df["category_name"].unique()
        for cat in categories:
            session.run(
                "MERGE (c:Category {name: $name})",
                name=cat
            )
        print(f"      Created {len(categories)} Category nodes")

        # Create Subcategory nodes
        subcats = df[["subcategory_name", "category_name"]].drop_duplicates()
        for _, row in subcats.iterrows():
            session.run(
                """
                MERGE (s:Subcategory {name: $name})
                SET s.category = $category
                WITH s
                MATCH (c:Category {name: $category})
                MERGE (s)-[:BELONGS_TO]->(c)
                """,
                name=row["subcategory_name"],
                category=row["category_name"]
            )
        print(f"      Created {len(subcats)} Subcategory nodes")

        # Create RoleGroup nodes
        group_stats = df.groupby("rec_group").agg(
            total_received=("message", "count"),
            top_category=("category_name", lambda x: x.value_counts().index[0])
        ).reset_index()

        for _, row in group_stats.iterrows():
            session.run(
                """
                MERGE (r:RoleGroup {name: $name})
                SET r.total_received = $total_received,
                    r.top_category   = $top_category
                """,
                name=row["rec_group"],
                total_received=int(row["total_received"]),
                top_category=row["top_category"]
            )
        print(f"      Created {len(group_stats)} RoleGroup nodes")

        # Create Role nodes (individual job titles)
        all_roles = pd.concat([
            df[["recipient_title", "rec_group"]].rename(columns={"recipient_title": "title", "rec_group": "group"}),
            df[["nominator_title", "nom_group"]].rename(columns={"nominator_title": "title", "nom_group": "group"}),
        ]).drop_duplicates("title")

        batch_size = 100
        for i in range(0, len(all_roles), batch_size):
            batch = all_roles.iloc[i:i+batch_size]
            session.run(
                """
                UNWIND $roles AS role
                MERGE (r:Role {title: role.title})
                SET r.group = role.group
                WITH r, role
                MATCH (g:RoleGroup {name: role.group})
                MERGE (r)-[:BELONGS_TO]->(g)
                """,
                roles=batch.to_dict("records")
            )
        print(f"      Created {len(all_roles)} Role nodes")

        # Create RoleGroup → RoleGroup RECOGNIZED relationships
        print("\n[5/5] Creating relationships...")
        edge_data = defaultdict(lambda: {"count": 0, "categories": []})
        for _, row in df.iterrows():
            key = (row["nom_group"], row["rec_group"])
            edge_data[key]["count"] += 1
            edge_data[key]["categories"].append(row["category_name"])

        for (src, tgt), data in edge_data.items():
            top_cat = max(set(data["categories"]), key=data["categories"].count)
            session.run(
                """
                MATCH (a:RoleGroup {name: $src})
                MATCH (b:RoleGroup {name: $tgt})
                MERGE (a)-[r:RECOGNIZED]->(b)
                SET r.count        = $count,
                    r.top_category = $top_category
                """,
                src=src, tgt=tgt,
                count=data["count"],
                top_category=top_cat
            )
        print(f"      Created {len(edge_data)} RoleGroup RECOGNIZED relationships")

        # Create Role → Role RECOGNIZED relationships (individual level)
        role_edges = defaultdict(lambda: {"count": 0, "awards": [], "categories": [], "subcategories": []})
        for _, row in df.iterrows():
            key = (row["nominator_title"], row["recipient_title"])
            role_edges[key]["count"] += 1
            role_edges[key]["awards"].append(row["award_title"])
            role_edges[key]["categories"].append(row["category_name"])
            role_edges[key]["subcategories"].append(row["subcategory_name"])

        # Batch role relationships
        role_edge_list = []
        for (src, tgt), data in role_edges.items():
            top_cat  = max(set(data["categories"]), key=data["categories"].count)
            top_sub  = max(set(data["subcategories"]), key=data["subcategories"].count)
            top_award = max(set(data["awards"]), key=data["awards"].count)
            role_edge_list.append({
                "src": src, "tgt": tgt,
                "count": data["count"],
                "top_category": top_cat,
                "top_subcategory": top_sub,
                "top_award": top_award,
            })

        for i in range(0, len(role_edge_list), batch_size):
            batch = role_edge_list[i:i+batch_size]
            session.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Role {title: edge.src})
                MATCH (b:Role {title: edge.tgt})
                MERGE (a)-[r:RECOGNIZED]->(b)
                SET r.count            = edge.count,
                    r.top_category     = edge.top_category,
                    r.top_subcategory  = edge.top_subcategory,
                    r.top_award        = edge.top_award
                """,
                edges=batch
            )
        print(f"      Created {len(role_edge_list)} Role RECOGNIZED relationships")

    driver.close()

    print("\n" + "=" * 60)
    print("  Neo4j Load Complete!")
    print(f"  Nodes   : {len(categories)} categories + {len(subcats)} subcategories")
    print(f"            + {len(group_stats)} role groups + {len(all_roles)} roles")
    print(f"  Edges   : {len(edge_data)} group relationships + {len(role_edge_list)} role relationships")
    print(f"  Database: {uri}")
    print("=" * 60)
    print("\nNext: python -m src.rag.neo4j_tool")


if __name__ == "__main__":
    load_graph()
