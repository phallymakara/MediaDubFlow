"""
LLM-powered translation service for Khmer localization.

Supports both OpenAI and Anthropic Claude APIs to translate dialogue segments
in sequential batches with glossary support and structured JSON response parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from mediadubflow.config.settings import TranslationProvider, settings

_SYSTEM_PROMPT_TEMPLATE = """You are an expert subtitle and dubbing translator specializing in natural, culturally authentic Khmer localization for dramas and movies.

Your task is to translate spoken dialogue lines from {source_language} into natural, conversational Khmer (ភាសាខ្មែរ).

Guidelines:
1. Translate dialogue to sound natural when spoken aloud by Khmer voice actors.
2. Maintain character relationships, tone, politeness levels, and emotional subtext.
3. Preserve the exact segment IDs provided in the input.
4. Keep translations concise enough to match the original timing constraints of video dubbing.
{glossary_section}

Format Requirements:
Output ONLY a valid JSON array of objects with the exact schema:
[
  {{"id": <int>, "translated_text": "<Khmer translation>"}}
]
Do not include markdown code fences or explanatory text. Output pure JSON only."""


def _build_system_prompt(source_language: str, glossary: dict[str, str] | None = None) -> str:
    """Build the translation system prompt with optional drama glossary terms."""
    glossary_section = ""
    if glossary:
        terms_list = "\n".join(f"- {term}: {meaning}" for term, meaning in glossary.items())
        glossary_section = f"\nUse the following project glossary for character names and specific terminology:\n{terms_list}\n"

    src_lang = source_language if source_language else "the original source language"
    return _SYSTEM_PROMPT_TEMPLATE.format(
        source_language=src_lang,
        glossary_section=glossary_section,
    )


def _extract_json_array(response_text: str) -> list[dict[str, Any]]:
    """
    Extract and parse a JSON array from raw model response text.

    Handles markdown code blocks and wrapping objects gracefully.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Fallback: search for first [ and last ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse LLM translation response as JSON: {exc}"
                ) from exc
        else:
            raise ValueError(f"Failed to parse LLM translation response as JSON: {exc}") from exc

    candidates: list[Any] = []
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                candidates = val
                break
    else:
        raise ValueError(f"Expected a JSON list of translated objects, got: {type(parsed)}")

    # Strict schema validation: only retain valid dictionaries with integer id and clean string
    sanitized: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            line_id = int(item["id"])
            raw_text = str(item.get("translated_text", "")).strip()
            # Filter non-printable control characters while preserving tabs, newlines, and valid Unicode (Khmer)
            clean_text = "".join(ch for ch in raw_text if ch in ("\t", "\n") or ord(ch) >= 32)
            sanitized.append({"id": line_id, "translated_text": clean_text})
        except (ValueError, TypeError):
            continue

    if not sanitized and candidates:
        raise ValueError("Parsed JSON did not contain valid {id, translated_text} objects.")

    return sanitized


async def _translate_batch_openai(
    batch_lines: list[dict[str, Any]],
    system_prompt: str,
) -> dict[int, str]:
    """Translate a batch of lines using the OpenAI API."""
    if not settings.openai_api_key.strip():
        raise ValueError(
            "OpenAI translation provider is active, but OPENAI_API_KEY is not configured in .env."
        )

    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    user_content = json.dumps(batch_lines, ensure_ascii=False)

    logger.debug("Requesting OpenAI translation for {} lines", len(batch_lines))
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    response_text = response.choices[0].message.content or "[]"
    parsed_items = _extract_json_array(response_text)

    return {
        int(item["id"]): str(item.get("translated_text", "")).strip()
        for item in parsed_items
        if "id" in item
    }


async def _translate_batch_anthropic(
    batch_lines: list[dict[str, Any]],
    system_prompt: str,
) -> dict[int, str]:
    """Translate a batch of lines using the Anthropic Claude API."""
    if not settings.anthropic_api_key.strip():
        raise ValueError(
            "Anthropic translation provider is active, but ANTHROPIC_API_KEY is not configured in .env."
        )

    from anthropic import AsyncAnthropic  # noqa: PLC0415

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    user_content = json.dumps(batch_lines, ensure_ascii=False)

    logger.debug("Requesting Anthropic translation for {} lines", len(batch_lines))
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        temperature=0.3,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_content},
        ],
    )

    # Combine text from content blocks
    response_text = "".join(block.text for block in response.content if hasattr(block, "text"))
    parsed_items = _extract_json_array(response_text)

    return {
        int(item["id"]): str(item.get("translated_text", "")).strip()
        for item in parsed_items
        if "id" in item
    }


async def translate_transcript_segments(
    segments: list[dict[str, Any]],
    *,
    source_language: str = "",
    target_language: str = "km",
    glossary: dict[str, str] | None = None,
    provider: TranslationProvider | None = None,
    batch_size: int = 40,
) -> list[dict[str, Any]]:
    """
    Translate transcript segments into Khmer using the configured LLM provider.

    Processes segments in sequential batches to prevent context token overflow
    and maps translated dialogue lines back to their original segment timestamps.

    Args:
        segments: List of transcript segment dictionaries (containing 'id', 'start', 'end', 'text').
        source_language: Source language code (e.g. 'zh', 'ko', 'en').
        target_language: Target language code (default 'km').
        glossary: Optional mapping of names/terms to preferred Khmer translations.
        provider: Override translation provider (defaults to settings.translation_provider).
        batch_size: Number of dialogue lines sent per LLM API request.

    Returns:
        List of enriched segment dictionaries with 'translated_text' added.

    Raises:
        ValueError: If API keys are missing or LLM returns unparseable content.
    """
    if not segments:
        return []

    active_provider = provider or settings.translation_provider
    system_prompt = _build_system_prompt(source_language, glossary)

    translated_map: dict[int, str] = {}

    for i in range(0, len(segments), batch_size):
        chunk = segments[i : i + batch_size]
        batch_input = [{"id": seg["id"], "text": seg.get("text", "").strip()} for seg in chunk]

        logger.info(
            "Translating batch {}/{} ({} lines) using provider={}",
            (i // batch_size) + 1,
            (len(segments) + batch_size - 1) // batch_size,
            len(batch_input),
            active_provider,
        )

        if active_provider == TranslationProvider.OPENAI:
            batch_result = await _translate_batch_openai(batch_input, system_prompt)
        elif active_provider == TranslationProvider.ANTHROPIC:
            batch_result = await _translate_batch_anthropic(batch_input, system_prompt)
        else:
            raise ValueError(f"Unsupported translation provider: {active_provider}")

        translated_map.update(batch_result)

    # Build final enriched segments preserving original structure
    enriched_segments: list[dict[str, Any]] = []
    for seg in segments:
        seg_id = seg["id"]
        original = seg.get("text", "").strip()
        translated = translated_map.get(seg_id, original)

        enriched_segments.append(
            {
                "id": seg_id,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "original_text": original,
                "translated_text": translated,
            }
        )

    logger.info(
        "Successfully translated {} segments into {}",
        len(enriched_segments),
        target_language,
    )
    return enriched_segments
