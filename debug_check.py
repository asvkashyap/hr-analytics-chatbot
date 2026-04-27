import json
from pathlib import Path

path = Path("output/reports/taxonomy/compressed_taxonomy_with_ids.json")
with open(path) as f:
    tax = json.load(f)
print("Taxonomy keys:", list(tax.keys()))
print("First category:", list(tax["recognition_taxonomy"].keys())[0])