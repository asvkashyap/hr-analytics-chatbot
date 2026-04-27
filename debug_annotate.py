import json
import pandas as pd
from pathlib import Path

debug_path = Path("output/reports/debug_raw_results.json")
if not debug_path.exists():
    print("No debug file found — creating mock data with 50 rows")
    mock_tagged = [
        {
            "row_id": "row" + str(i),
            "reason": "test reason",
            "category_name": "Operational Excellence & Delivery",
            "category_id": "B",
            "sub_category_name": "Project & Process Management",
            "sub_category_id": "B1"
        }
        for i in range(1, 51)
    ]
    results = [
        {"batch_id": "1", "tagged_results": mock_tagged},
        {"batch_id": "2", "tagged_results": mock_tagged},
    ]
else:
    print("Found real debug file")
    with open(debug_path) as f:
        results = json.load(f)

# Test validate_batch_result logic
def validate_batch_result(batch_id, tagged, expected_rows):
    if isinstance(tagged, dict):
        tagged = [tagged]
    if not isinstance(tagged, list):
        print("ERROR batch " + str(batch_id) + " wrong type: " + str(type(tagged)))
        return False, tagged
    if len(tagged) < expected_rows * 0.5:
        print("ERROR batch " + str(batch_id) + " only " + str(len(tagged)) + "/" + str(expected_rows) + " rows")
        return False, tagged
    return True, tagged

# Test extract_tags logic
all_rows = []
for result in results:
    batch_id = result["batch_id"]
    tagged = result["tagged_results"]

    print("batch_id: " + str(batch_id) + " | type: " + str(type(tagged)))

    if isinstance(tagged, dict):
        tagged = [tagged]

    is_valid, tagged = validate_batch_result(batch_id, tagged, 50)
    if not is_valid:
        print("STOPPING — invalid batch result")
        break

    print("  Valid: " + str(len(tagged)) + " rows")

    for i, item in enumerate(tagged):
        if not isinstance(item, dict):
            continue
        row_id = item.get("row_id", "row" + str(i + 1))
        all_rows.append({
            "batch_id": batch_id,
            "row_id": row_id,
            "reason": item.get("reason", ""),
            "category_id": item.get("category_id", ""),
            "category_name": item.get("category_name", ""),
            "subcategory_id": item.get("sub_category_id", item.get("subcategory_id", "")),
            "subcategory_name": item.get("sub_category_name", item.get("subcategory_name", "")),
        })

print("\nTotal rows extracted: " + str(len(all_rows)))
if all_rows:
    print("Sample row 1: " + str(all_rows[0]))
    print("Sample row 2: " + str(all_rows[1]))