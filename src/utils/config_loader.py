import os
from functools import lru_cache
from pathlib import Path
import yaml
from dotenv import load_dotenv
from src.utils.exceptions import ConfigurationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
load_dotenv()
CONFIG_PATH = Path("config/config.yaml")


@lru_cache(maxsize=1)
def get_config() -> dict:
    if not CONFIG_PATH.exists():
        raise ConfigurationError("Config file not found: " + str(CONFIG_PATH))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError("Invalid YAML: " + str(exc)) from exc
    if cfg is None:
        raise ConfigurationError("Config file is empty")
    logger.debug("Config loaded")
    return cfg


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ConfigurationError("GEMINI_API_KEY is not set in your .env file.")
    return key


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO").upper()