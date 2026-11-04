import time
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from src.llm.base_client import BaseLLMClient, LLMResponse
from src.utils.config_loader import get_config, get_gemini_api_key
from src.utils.exceptions import LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMResponseError
from src.utils.logging_config import get_cost_logger, get_logger
from src.utils.llm_usage import RateLimiter, record_usage, extract_usage, price_for, estimate_tokens

logger = get_logger(__name__)
cost_logger = get_cost_logger()
_limiter = RateLimiter("gemini")   # limits come from config/pricing.yaml


class GeminiClient(BaseLLMClient):

    def __init__(self, override_model=None, component="gemini_client"):
        cfg = get_config()["llm"]
        api_key = get_gemini_api_key()

        self.client = genai.Client(api_key=api_key)
        self.model_name = override_model or cfg["model"]
        self.component = component
        self.default_temperature = cfg.get("temperature", 0.1)
        self.max_output_tokens = cfg.get("max_output_tokens", 4096)
        self.timeout = cfg.get("timeout_seconds", 60)

        # Prices now come from config/pricing.yaml (shared with the agents)
        pricing = price_for(self.model_name)
        self._input_cost_per_m = pricing["input_per_million"]
        self._output_cost_per_m = pricing["output_per_million"]

        logger.info("GeminiClient initialised | model=" + self.model_name)

    def complete(self, prompt, temperature=None):
        return self._complete_with_retry(prompt, temperature)

    def health_check(self):
        try:
            resp = self.complete("Say OK", temperature=0.0)
            logger.info("Health check passed | " + resp.text[:50])
            return True
        except Exception as exc:
            logger.error("Health check failed: " + str(exc))
            return False

    @retry(
        retry=retry_if_exception_type(LLMRateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _complete_with_retry(self, prompt, temperature=None):
        temp = temperature if temperature is not None else self.default_temperature
        _limiter.wait(estimate_tokens(prompt, expected_output=self.max_output_tokens))
        start = time.perf_counter()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "api key" in exc_str or "401" in exc_str:
                raise LLMAuthError("Gemini auth failed: " + str(exc)) from exc
            if "quota" in exc_str or "429" in exc_str or "rate" in exc_str:
                raise LLMRateLimitError("Rate limit hit: " + str(exc)) from exc
            if "timeout" in exc_str:
                raise LLMTimeoutError(self.timeout) from exc
            raise LLMResponseError(str(exc)) from exc

        elapsed = time.perf_counter() - start

        try:
            text = response.text
        except Exception as exc:
            raise LLMResponseError(str(response)) from exc

        # extract_usage counts thinking tokens as output, since Gemini bills them that way
        input_tokens, output_tokens = extract_usage(response)
        _limiter.correct(input_tokens + output_tokens)
        input_cost = (input_tokens / 1_000_000) * self._input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * self._output_cost_per_m
        record_usage(self.component, self.model_name, input_tokens, output_tokens, elapsed)

        result = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            model=self.model_name,
            raw=response,
        )

        cost_logger.info(
            "model=" + self.model_name +
            " | in=" + str(input_tokens) +
            " | out=" + str(output_tokens) +
            " | cost=$" + str(round(result.total_cost_usd, 5)) +
            " | elapsed=" + str(round(elapsed, 2)) + "s"
        )

        return result