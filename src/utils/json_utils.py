import json
import re
from typing import Any
import json5
from src.utils.exceptions import LLMResponseError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json(raw: str, batch_id: int | None = None):
    if not raw or not raw.strip():
        raise LLMResponseError(raw or "", batch_id=batch_id)

    raw = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            return result

    obj_match = _find_outermost(raw, "{", "}")
    arr_match = _find_outermost(raw, "[", "]")

    candidates = []
    if obj_match:
        candidates.append(obj_match)
    if arr_match:
        candidates.append(arr_match)

    for candidate in candidates:
        result = _try_parse(candidate)
        if result is not None:
            return result

    result = _try_parse(raw)
    if result is not None:
        return result

    raise LLMResponseError(raw, batch_id=batch_id)


def _try_parse(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json5.loads(text)
    except Exception:
        pass
    return None


def _find_outermost(text: str, open_char: str, close_char: str):
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def safe_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)