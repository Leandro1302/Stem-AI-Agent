"""
Single helper for all LLM calls.
Switch backends by changing the BACKEND constant below.
"""
import logging
import os

logger = logging.getLogger(__name__)

BACKEND = "openai"  # "openai" or "ollama"

OLLAMA_MODEL = "gemma3"
OPENAI_MODEL = "gpt-4o-mini"


if BACKEND == "openai":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "BACKEND is 'openai' but OPENAI_API_KEY environment variable is not set."
        )
    from openai import OpenAI
    _openai_client = OpenAI()
elif BACKEND == "ollama":
    import ollama
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r}. Use 'openai' or 'ollama'.")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured LLM backend and return the raw response string."""
    try:
        if BACKEND == "openai":
            response = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        else:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.message.content
        logger.debug("LLM response received (%d chars)", len(content))
        return content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise
