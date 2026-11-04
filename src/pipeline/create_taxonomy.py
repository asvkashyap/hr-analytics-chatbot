"""
Stage 1: Create Topic Taxonomy
Usage: python -m src.pipeline.create_taxonomy
"""

import json
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.llm.gemini_client import GeminiClient
from src.llm.prompts import TAXONOMY_COMPRESS_PROMPT, TAXONOMY_INCREMENTAL_PROMPT, TAXONOMY_INIT_PROMPT
from src.utils.config_loader import get_config, get_log_level
from src.utils.data_utils import clean_dataframe, format_award_info, load_csv, validate_schema
from src.utils.exceptions import TaxonomyParseError, LLMRateLimitError
from src.utils.json_utils import extract_json, safe_dumps
from src.utils.logging_config import get_logger, setup_logging
from src.utils.llm_usage import summarize

setup_logging(get_log_level())
logger = get_logger(__name__)


def is_valid_taxonomy(obj):
    return (
        isinstance(obj, dict) and
        "recognition_taxonomy" in obj and
        isinstance(obj["recognition_taxonomy"], dict)
    )


def merge_taxonomies(base, additions):
    if not is_valid_taxonomy(additions):
        logger.warning("Skipping merge — additions is not a valid taxonomy")
        return base

    result = copy.deepcopy(base)
    base_tax = result.setdefault("recognition_taxonomy", {})
    additions_tax = additions.get("recognition_taxonomy", {})

    for cat_name, cat_data in additions_tax.items():
        if not isinstance(cat_data, dict):
            continue
        if cat_name in base_tax:
            existing = base_tax[cat_name].get("subcategories", [])
            for sub in cat_data.get("subcategories", []):
                if sub not in existing:
                    existing.append(sub)
            base_tax[cat_name]["subcategories"] = existing
        else:
            base_tax[cat_name] = cat_data

    return result


def add_ids(taxonomy):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = {"recognition_taxonomy": {}}

    for i, (cat_name, cat_data) in enumerate(taxonomy["recognition_taxonomy"].items()):
        cat_id = letters[i % 26]
        subcats = [
            {"id": cat_id + str(j), "name": sub}
            for j, sub in enumerate(cat_data.get("subcategories", []), start=1)
        ]
        result["recognition_taxonomy"][cat_name] = {
            "id": cat_id,
            "description": cat_data.get("description", ""),
            "subcategories": subcats,
        }

    return result


def load_checkpoint(start_batch, output_dir, tax_cfg):
    """Load taxonomy from checkpoint or final file when resuming."""
    if start_batch <= 1:
        return None

    # Try exact previous batch checkpoint first
    last_checkpoint = output_dir / ("taxonomy_batch_" + str(start_batch - 1).zfill(4) + ".json")
    if last_checkpoint.exists():
        with open(last_checkpoint, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        logger.info("Resumed from checkpoint: " + str(last_checkpoint))
        return taxonomy

    # Try final output file
    final_file = Path(tax_cfg["output_file"])
    if final_file.exists():
        with open(final_file, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        logger.info("Resumed from final file: " + str(final_file))
        return taxonomy

    # Try any checkpoint file — find the latest one
    checkpoints = sorted(output_dir.glob("taxonomy_batch_*.json"), reverse=True)
    if checkpoints:
        with open(checkpoints[0], "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        logger.info("Resumed from latest checkpoint: " + str(checkpoints[0]))
        return taxonomy

    logger.warning("No checkpoint found — starting fresh")
    return None


def run(cfg=None):
    cfg = cfg or get_config()
    tax_cfg = cfg["taxonomy"]
    paths = cfg["paths"]

    output_dir = Path(tax_cfg["output_file"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = load_csv(paths["raw_data"])
    validate_schema(df)
    df = clean_dataframe(
        df,
        exclude_award_reasons=cfg["visualization"]["exclude_award_reasons"],
        exclude_product_values=cfg["visualization"]["exclude_product_values"],
    )

    df = df.reset_index(drop=True)
    logger.info("Processing full dataset: " + str(len(df)) + " records")

    df["formatted_info"] = df.apply(format_award_info, axis=1)

    batch_size = tax_cfg["batch_size"]
    df["batch_id"] = np.ceil((np.arange(len(df)) + 1) / batch_size).astype(int)
    df["row_in_batch"] = "row" + ((np.arange(len(df)) % batch_size) + 1).astype(str)
    df["summary_dict_batch"] = df["row_in_batch"] + ": " + df["formatted_info"]

    client = GeminiClient(component="taxonomy")

    start_batch = tax_cfg["start_batch"]
    compress_every = tax_cfg["compress_every_n_batches"]
    unique_batches = sorted(df["batch_id"].unique())

    # Load checkpoint if resuming
    taxonomy = load_checkpoint(start_batch, output_dir, tax_cfg)
    total_cost = 0.0

    for batch_id in tqdm(unique_batches, desc="Building taxonomy"):
        if batch_id < start_batch:
            continue

        batch_content = "\n".join(
            df[df["batch_id"] == batch_id]["summary_dict_batch"].values
        )

        if taxonomy is None:
            prompt = TAXONOMY_INIT_PROMPT.format(batch_content=batch_content)
        else:
            prompt = TAXONOMY_INCREMENTAL_PROMPT.format(
                batch_content=batch_content,
                existing_taxonomy=safe_dumps(taxonomy),
            )

        try:
            response = client.complete(prompt)
            total_cost += response.total_cost_usd
            parsed = extract_json(response.text, batch_id=batch_id)

            if not is_valid_taxonomy(parsed):
                logger.warning("Batch " + str(batch_id) + " returned invalid structure — skipping")
                time.sleep(8)
                continue

            if taxonomy is None:
                taxonomy = parsed
            else:
                taxonomy = merge_taxonomies(taxonomy, parsed)

        except LLMRateLimitError:
            logger.error("Rate limit hit — stopping pipeline.")
            logger.error("Resume by setting start_batch: " + str(batch_id) + " in config.yaml")
            break

        except Exception as exc:
            logger.error("Batch " + str(batch_id) + " failed: " + str(exc))
            time.sleep(8)
            continue

        # Periodic compression
        if batch_id % compress_every == 0 and taxonomy:
            logger.info("Compressing taxonomy at batch " + str(batch_id))
            try:
                time.sleep(8)
                compress_prompt = TAXONOMY_COMPRESS_PROMPT.format(taxonomy=safe_dumps(taxonomy))
                compress_resp = client.complete(compress_prompt)
                total_cost += compress_resp.total_cost_usd
                compressed = extract_json(compress_resp.text)
                if is_valid_taxonomy(compressed):
                    taxonomy = compressed
                    logger.info("Compression successful")
                else:
                    logger.warning("Compression returned invalid structure — keeping current taxonomy")
            except LLMRateLimitError:
                logger.error("Rate limit hit during compression — stopping.")
                break
            except Exception as exc:
                logger.warning("Compression failed, continuing: " + str(exc))

        # Checkpoint save
        checkpoint = output_dir / ("taxonomy_batch_" + str(batch_id).zfill(4) + ".json")
        checkpoint.write_text(safe_dumps(taxonomy or {}), encoding="utf-8")
        logger.info("Checkpoint saved: " + str(checkpoint))

        # Wait between batches to avoid rate limits
        logger.info("Waiting 8 seconds before next batch...")
        time.sleep(8)

    if not taxonomy:
        raise TaxonomyParseError("Taxonomy creation produced no results.")

    logger.info("Adding IDs to final taxonomy...")
    taxonomy_with_ids = add_ids(taxonomy)

    final_path = Path(tax_cfg["output_file"])
    final_path.write_text(safe_dumps(taxonomy_with_ids), encoding="utf-8")

    logger.info("Taxonomy saved to " + str(final_path))
    logger.info("Total cost: $" + str(round(total_cost, 4)))
    print("\n" + summarize())

    print("\n=== TAXONOMY SUMMARY ===")
    for cat_name, cat_data in taxonomy_with_ids["recognition_taxonomy"].items():
        print(cat_data["id"] + ". " + cat_name + " - " + str(len(cat_data["subcategories"])) + " subcategories")

    return taxonomy_with_ids


if __name__ == "__main__":
    run()