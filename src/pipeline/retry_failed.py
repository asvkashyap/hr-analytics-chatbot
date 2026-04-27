"""
Retry failed annotation records.
Usage: python -m src.pipeline.retry_failed
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.config_loader import get_config, get_log_level
from src.utils.logging_config import get_logger, setup_logging
from src.pipeline.annotate import (
    build_taxonomy_text,
    build_prompt,
    process_batch,
    get_all_api_keys,
)

setup_logging(get_log_level())
logger = get_logger(__name__)


def run(cfg=None):
    cfg = cfg or get_config()
    ann_cfg = cfg["annotation"]
    tax_cfg = cfg["taxonomy"]

    # Load failed records
    failed_path = Path("output/reports/failed_records.csv")
    if not failed_path.exists():
        logger.info("No failed records found — nothing to retry")
        return

    df_failed = pd.read_csv(failed_path)
    logger.info("Retrying " + str(len(df_failed)) + " failed records")

    # Load taxonomy
    taxonomy_path = Path(tax_cfg["output_file"])
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    taxonomy_text = build_taxonomy_text(taxonomy)

    # Load API keys
    api_keys = get_all_api_keys()
    if not api_keys:
        raise Exception("No API keys found.")
    logger.info("Loaded " + str(len(api_keys)) + " API key(s)")

    # Batch the failed records
    batch_size = ann_cfg["batch_size"]
    df_failed = df_failed.reset_index(drop=True)
    df_failed["batch_id"] = np.ceil((np.arange(len(df_failed)) + 1) / batch_size).astype(int)
    total_batches = int(df_failed["batch_id"].max())
    logger.info("Total retry batches: " + str(total_batches))

    all_results = []
    still_failed = []
    current_key_idx = 0

    for batch_num in tqdm(range(1, total_batches + 1), desc="Retrying failed"):
        batch_df = df_failed[df_failed["batch_id"] == batch_num].copy().reset_index()
        orig_indices = batch_df["index"].tolist()
        batch_df = batch_df.drop(columns=["batch_id", "index"])

        prompt = build_prompt(batch_df, taxonomy_text)
        results, current_key_idx = process_batch(
            batch_df, prompt, batch_num, api_keys, current_key_idx
        )

        if results is None:
            for orig_idx in orig_indices:
                still_failed.append(df_failed.loc[orig_idx].to_dict())
        else:
            for item in results:
                local_idx = item["index"]
                orig_idx = orig_indices[local_idx]
                row = df_failed.loc[orig_idx].to_dict()
                row.update({
                    "category_id": item["category_id"],
                    "category_name": item["category_name"],
                    "subcategory_id": item["subcategory_id"],
                    "subcategory_name": item["subcategory_name"],
                })
                all_results.append(row)

        if batch_num < total_batches:
            logger.info("Waiting 6s before next batch...")
            time.sleep(6)

    # Merge with existing results
    if all_results:
        existing_path = Path(ann_cfg["output_file"])
        if existing_path.exists():
            df_existing = pd.read_csv(existing_path)
            df_new = pd.DataFrame(all_results)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame(all_results)

        df_combined.to_csv(ann_cfg["output_file"], index=False, encoding="utf-8-sig")
        logger.info("Combined results saved: " + str(len(df_combined)) + " records")

    # Update failed records file
    if still_failed:
        df_still = pd.DataFrame(still_failed)
        df_still.to_csv(failed_path, index=False)
        logger.warning("Still failed: " + str(len(still_failed)) + " records")
    else:
        failed_path.unlink()
        logger.info("All failed records recovered successfully")

    print("\n=== RETRY SUMMARY ===")
    print("Retried:      " + str(len(df_failed)))
    print("Recovered:    " + str(len(all_results)))
    print("Still failed: " + str(len(still_failed)))

    if all_results:
        df_combined = pd.DataFrame(all_results)
        print("\n=== Category Distribution (recovered) ===")
        print(df_combined["category_name"].value_counts().to_string())


if __name__ == "__main__":
    run()