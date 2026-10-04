"""
Stage 2: Annotate records with taxonomy tags.
Uses urllib directly, multiple API keys with rotation, strict count validation.
Usage: python -m src.pipeline.annotate
"""

import json
import json5
import time
import numpy as np
import pandas as pd
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

from src.utils.config_loader import get_config, get_gemini_api_key, get_log_level
from src.utils.data_utils import clean_dataframe, load_csv, validate_schema
from src.utils.logging_config import get_logger, setup_logging
from src.utils.llm_usage import RateLimiter, record_usage, extract_usage, estimate_tokens, summarize

setup_logging(get_log_level())
logger = get_logger(__name__)

MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    + MODEL
    + ":generateContent"
)
RETRIES_PER_KEY = 2
SLEEP_BETWEEN = 6   # fixed pause kept as a safety margin; the limiter below adapts to real usage
_limiter = RateLimiter("gemini")   # limits come from config/pricing.yaml

SYSTEM_PROMPT = """You are an HR researcher annotating employee recognition messages.
For each message, assign the single best matching category and subcategory from the taxonomy.

Return ONLY a valid JSON array — no explanation, no markdown, no code fences.
Each element must have exactly these fields:
  "index"           : integer (the index provided with the message)
  "category_id"     : string (e.g. "A")
  "category_name"   : string
  "subcategory_id"  : string (e.g. "A1")
  "subcategory_name": string

Example:
[
  {"index": 0, "category_id": "A", "category_name": "Strategic Business Outcomes", "subcategory_id": "A1", "subcategory_name": "Business Growth & Client Success"},
  {"index": 1, "category_id": "D", "category_name": "Organizational & Team Enablement", "subcategory_id": "D1", "subcategory_name": "Team Collaboration & Integration"}
]"""


def build_taxonomy_text(taxonomy):
    lines = []
    for cat_name, cat_data in taxonomy["recognition_taxonomy"].items():
        lines.append("[" + cat_data["id"] + "] " + cat_name)
        for sub in cat_data["subcategories"]:
            lines.append("    [" + sub["id"] + "] " + sub["name"])
    return "\n".join(lines)


def build_prompt(batch_df, taxonomy_text):
    messages_block = ""
    for local_idx, (_, row) in enumerate(batch_df.iterrows()):
        messages_block += "[" + str(local_idx) + "] " + str(row["message"]) + "\n"

    return (
        "TAXONOMY:\n" + taxonomy_text +
        "\n\nRECOGNITION MESSAGES TO ANNOTATE:\n" + messages_block +
        "\nReturn a JSON array with one object per message."
    )


def call_gemini(prompt, api_key):
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384,
        }
    }

    data = json.dumps(payload).encode("utf-8")
    url = GEMINI_URL + "?key=" + api_key
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    _limiter.wait(estimate_tokens(SYSTEM_PROMPT + prompt, expected_output=4000))
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # Log tokens and cost (thinking tokens count as output)
    in_tok, out_tok = extract_usage(result)
    _limiter.correct(in_tok + out_tok)
    record_usage("annotation", MODEL, in_tok, out_tok, time.perf_counter() - start)

    return result["candidates"][0]["content"]["parts"][0]["text"].strip()


def parse_response(text, expected_count):
    clean = text.strip()

    # Strip markdown fences
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    clean = clean.strip()

    # Save raw response for debugging
    debug_path = Path("output/reports/debug_raw_response.txt")
    debug_path.write_text(clean, encoding="utf-8")

    # Try standard json first then json5 as fallback
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        try:
            parsed = json5.loads(clean)
        except Exception as e:
            raise ValueError("JSON parse failed: " + str(e))

    if not isinstance(parsed, list):
        raise ValueError("Response is not a JSON array")

    if len(parsed) != expected_count:
        raise ValueError(
            "Expected " + str(expected_count) + " items, got " + str(len(parsed))
        )

    required = {"index", "category_id", "category_name", "subcategory_id", "subcategory_name"}
    for item in parsed:
        missing = required - set(item.keys())
        if missing:
            raise ValueError("Missing fields " + str(missing))

    return parsed


def get_all_api_keys():
    cfg = get_config()
    keys = cfg.get("llm", {}).get("api_keys", [])

    if not keys:
        try:
            single_key = get_gemini_api_key()
            keys = [single_key]
        except Exception:
            keys = []

    return [k for k in keys if k and not k.startswith("your_key")]


def process_batch(batch_df, prompt, batch_num, api_keys, start_key_idx=0):
    for key_idx in range(start_key_idx, len(api_keys)):
        api_key = api_keys[key_idx]
        for attempt in range(1, RETRIES_PER_KEY + 1):
            try:
                logger.info(
                    "Batch " + str(batch_num) +
                    " | Key " + str(key_idx + 1) +
                    " | Attempt " + str(attempt)
                )
                raw = call_gemini(prompt, api_key)
                parsed = parse_response(raw, len(batch_df))
                logger.info(
                    "Batch " + str(batch_num) + " OK — " + str(len(parsed)) + " records"
                )
                return parsed, key_idx

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                logger.error("HTTP " + str(e.code) + ": " + body[:200])
                if e.code == 429:
                    logger.warning(
                        "Key " + str(key_idx + 1) +
                        " rate limited — rotating to next key immediately"
                    )
                    break
                elif e.code in (500, 503):
                    wait = 10 if attempt == 1 else 20
                    logger.info("Waiting " + str(wait) + "s before retry...")
                    time.sleep(wait)
                else:
                    break

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                logger.error("Parse error: " + str(e))

            except Exception as e:
                logger.error("Unexpected error: " + str(e))
                time.sleep(5)

        logger.warning("Key " + str(key_idx + 1) + " exhausted — rotating to next key...")
        time.sleep(2)

    # All keys exhausted — wrap around after wait
    logger.warning("All keys exhausted — waiting 60s then wrapping to Key 1...")
    time.sleep(60)
    return None, 0


def run(cfg=None):
    cfg = cfg or get_config()
    paths = cfg["paths"]
    tax_cfg = cfg["taxonomy"]
    ann_cfg = cfg["annotation"]

    # Load API keys
    api_keys = get_all_api_keys()
    if not api_keys:
        raise Exception("No API keys found. Check config.yaml llm.api_keys or .env GEMINI_API_KEY.")
    logger.info("Loaded " + str(len(api_keys)) + " API key(s)")

    # Load taxonomy
    taxonomy_path = Path(tax_cfg["output_file"])
    if not taxonomy_path.exists():
        raise FileNotFoundError("Taxonomy not found. Run create_taxonomy.py first.")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    logger.info("Taxonomy loaded")
    taxonomy_text = build_taxonomy_text(taxonomy)

    # Load and clean data
    df = load_csv(paths["raw_data"])
    validate_schema(df)
    df = clean_dataframe(
        df,
        exclude_award_reasons=cfg["visualization"]["exclude_award_reasons"],
        exclude_product_values=cfg["visualization"]["exclude_product_values"],
    )
    df = df.reset_index(drop=True)
    logger.info("Annotating full dataset: " + str(len(df)) + " records")

    batch_size = ann_cfg["batch_size"]
    df["batch_id"] = np.ceil((np.arange(len(df)) + 1) / batch_size).astype(int)
    total_batches = int(df["batch_id"].max())
    logger.info("Total batches: " + str(total_batches))


    all_results = []
    failed_rows = []
    current_key_idx = 0

    for batch_num in tqdm(range(1, total_batches + 1), desc="Annotating"):
        batch_df = df[df["batch_id"] == batch_num].copy().reset_index()
        orig_indices = batch_df["index"].tolist()
        batch_df = batch_df.drop(columns=["batch_id", "index"])

        prompt = build_prompt(batch_df, taxonomy_text)
        results, current_key_idx = process_batch(batch_df, prompt, batch_num, api_keys, current_key_idx)

        if results is None:
            for orig_idx in orig_indices:
                failed_rows.append(df.loc[orig_idx].to_dict())
        else:
            for item in results:
                local_idx = item["index"]
                orig_idx = orig_indices[local_idx]
                row = df.loc[orig_idx].to_dict()
                row.update({
                    "category_id": item["category_id"],
                    "category_name": item["category_name"],
                    "subcategory_id": item["subcategory_id"],
                    "subcategory_name": item["subcategory_name"],
                })
                all_results.append(row)

        if batch_num < total_batches:
            logger.info("Waiting " + str(SLEEP_BETWEEN) + "s before next batch...")
            time.sleep(SLEEP_BETWEEN)

    # Save results
    output_dir = Path(ann_cfg["output_file"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(ann_cfg["output_file"], index=False, encoding="utf-8-sig")
        logger.info("Results saved: " + str(len(results_df)) + " records")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_path = output_dir / "failed_records.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.warning("Failed records saved: " + str(len(failed_rows)))

    print("\n=== ANNOTATION SUMMARY ===")
    print("Total:     " + str(len(df)))
    print("Annotated: " + str(len(all_results)))
    print("Failed:    " + str(len(failed_rows)))
    print("\n" + summarize())

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n=== Category Distribution ===")
        print(results_df["category_name"].value_counts().to_string())

    return pd.DataFrame(all_results) if all_results else None


if __name__ == "__main__":
    run()