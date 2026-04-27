from pathlib import Path
import pandas as pd
from src.utils.exceptions import DataLoadError, DataSchemaError, DataValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
REQUIRED_COLUMNS = {"message"}


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise DataLoadError("Data file not found: " + str(path))
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded " + str(len(df)) + " rows from " + str(path))
        return df
    except Exception as exc:
        raise DataLoadError("Failed to read CSV: " + str(path)) from exc


def validate_schema(df, source=""):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataSchemaError(missing_columns=list(missing), source=source)
    logger.debug("Schema validation passed")


def validate_data_quality(df):
    report = {}
    report["null_rates"] = df.isnull().mean().to_dict()

    if "message" in df.columns:
        empty_messages = df["message"].isna() | (df["message"].str.strip() == "")
        report["empty_message_pct"] = round(empty_messages.mean() * 100, 2)
        report["avg_message_length"] = df["message"].dropna().str.len().mean()
        dup_count = df["message"].duplicated().sum()
        report["duplicate_messages"] = int(dup_count)
        report["duplicate_pct"] = round(dup_count / len(df) * 100, 2)

    if report.get("empty_message_pct", 0) > 50:
        raise DataValidationError(
            "Over 50% of messages are empty.",
            context=report,
        )

    logger.info("Data quality check passed for " + str(len(df)) + " rows")
    return report


def clean_dataframe(df, exclude_award_reasons=None, exclude_product_values=None):
    original_count = len(df)

    df = df[df["message"].notna() & (df["message"].str.strip() != "")].copy()

    if exclude_award_reasons and "award_reason" in df.columns:
        df = df[~df["award_reason"].isin(exclude_award_reasons)]

    if exclude_product_values and "product" in df.columns:
        df = df[~df["product"].isin(exclude_product_values)]

    df = df.drop_duplicates(subset=["message"])

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    if "award_date" in df.columns:
        df["award_date"] = pd.to_datetime(df["award_date"], errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("Cleaning done. Removed " + str(original_count - len(df)) + " rows. Final: " + str(len(df)))
    return df


def format_award_info(row):
    def _val(col):
        v = row.get(col)
        return str(v).strip() if pd.notna(v) and str(v).strip() else None

    parts = []

    nom = _val("nominator_title")
    if nom:
        parts.append("<nominator>" + nom + "</nominator>")

    rec = _val("recipient_title")
    if rec:
        parts.append("<recipient>" + rec + "</recipient>")

    title = _val("award_title")
    if title:
        parts.append("<award>" + title + "</award>")

    message = _val("message") or ""
    parts.append("<message>" + message + "</message>")

    return "\n".join(parts)