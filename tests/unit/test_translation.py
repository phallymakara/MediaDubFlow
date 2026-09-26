"""
Unit tests for the TranslationStage and LLM translation service.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mediadubflow.config.settings import TranslationProvider, settings
from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.translation import TranslationStage
from mediadubflow.services.translation_service import (
    _build_system_prompt,
    _extract_json_array,
    translate_transcript_segments,
)


def test_build_system_prompt_without_glossary() -> None:
    """Prompt includes default source language and no glossary section."""
    prompt = _build_system_prompt("zh")
    assert "zh" in prompt
    assert "Khmer" in prompt
    assert "glossary" not in prompt.lower()


def test_build_system_prompt_with_glossary() -> None:
    """Prompt includes injected glossary terms."""
    glossary = {"Lord Ming": "ព្រះអង្គម្ចាស់មីង", "Sword": "ដាវបុរាណ"}
    prompt = _build_system_prompt("zh", glossary)
    assert 'source="Lord Ming"' in prompt
    assert 'target="ព្រះអង្គម្ចាស់មីង"' in prompt
    assert 'source="Sword"' in prompt
    assert 'target="ដាវបុរាណ"' in prompt


def test_extract_json_array_pure_json() -> None:
    """Successfully parses pure JSON array string."""
    raw = '[{"id": 0, "translated_text": "សួស្តី"}]'
    parsed = _extract_json_array(raw)
    assert len(parsed) == 1
    assert parsed[0]["translated_text"] == "សួស្តី"


def test_extract_json_array_markdown_fenced() -> None:
    """Successfully extracts JSON array enclosed in markdown code fences."""
    raw = '```json\n[{"id": 1, "translated_text": "ជំរាបសួរ"}]\n```'
    parsed = _extract_json_array(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == 1


def test_extract_json_array_wrapped_in_dict() -> None:
    """Successfully extracts list from a dictionary response."""
    raw = '{"translations": [{"id": 2, "translated_text": "អរគុណ"}]}'
    parsed = _extract_json_array(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == 2


def test_extract_json_array_invalid() -> None:
    """Raises ValueError when input cannot be parsed as JSON."""
    with pytest.raises(ValueError, match="Failed to parse LLM translation response as JSON"):
        _extract_json_array("Not valid JSON at all")


@pytest.mark.asyncio
async def test_translate_transcript_segments_empty() -> None:
    """Translating an empty list returns empty list immediately."""
    result = await translate_transcript_segments([], source_language="zh")
    assert result == []


@pytest.mark.asyncio
async def test_translate_openai_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ValueError when OpenAI is active but key is empty."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        await translate_transcript_segments(
            [{"id": 0, "text": "hello"}],
            source_language="en",
            provider=TranslationProvider.OPENAI,
        )


@pytest.mark.asyncio
async def test_translate_anthropic_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ValueError when Anthropic is active but key is empty."""
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not configured"):
        await translate_transcript_segments(
            [{"id": 0, "text": "hello"}],
            source_language="en",
            provider=TranslationProvider.ANTHROPIC,
        )


@pytest.mark.asyncio
async def test_translate_gemini_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ValueError when Gemini is active but key is empty."""
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        await translate_transcript_segments(
            [{"id": 0, "text": "hello"}],
            source_language="en",
            provider=TranslationProvider.GEMINI,
        )


@pytest.mark.asyncio
async def test_translate_azure_openai_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ValueError when Azure OpenAI provider is selected but credentials are missing."""
    monkeypatch.setattr(settings, "azure_openai_api_key", "")
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://mock.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_deployment_name", "gpt-4o")

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY is not configured"):
        await translate_transcript_segments(
            [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
            source_language="en",
            provider=TranslationProvider.AZURE_OPENAI,
        )


@pytest.mark.asyncio
async def test_translate_azure_openai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successfully translates segments with mocked Azure OpenAI response."""
    monkeypatch.setattr(settings, "azure_openai_api_key", "azure-test-key")
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://mock.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_deployment_name", "gpt-4o")

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='[{"id": 0, "translated_text": "សួស្តី"}, {"id": 1, "translated_text": "អរគុណ"}]'
                )
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    input_segments = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello"},
        {"id": 1, "start": 1.6, "end": 2.5, "text": "Thank you"},
    ]

    with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
        translated = await translate_transcript_segments(
            input_segments,
            source_language="en",
            provider=TranslationProvider.AZURE_OPENAI,
        )

    assert len(translated) == 2
    assert translated[0]["translated_text"] == "សួស្តី"
    assert translated[1]["translated_text"] == "អរគុណ"


@pytest.mark.asyncio
async def test_translate_gemini_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successfully translates segments with mocked Gemini response."""
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaSy-test-key")

    mock_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '[{"id": 0, "translated_text": "សួស្តី"}, {"id": 1, "translated_text": "អរគុណ"}]'
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_json
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_resp

    input_segments = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello"},
        {"id": 1, "start": 1.6, "end": 2.5, "text": "Thank you"},
    ]

    with patch("httpx.AsyncClient", return_value=mock_client):
        translated = await translate_transcript_segments(
            input_segments,
            source_language="en",
            provider=TranslationProvider.GEMINI,
        )

    assert len(translated) == 2
    assert translated[0]["translated_text"] == "សួស្តី"
    assert translated[1]["translated_text"] == "អរគុណ"


@pytest.mark.asyncio
async def test_translate_openai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successfully translates segments with mocked OpenAI response."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-key")

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='[{"id": 0, "translated_text": "សួស្តី"}, {"id": 1, "translated_text": "អរគុណ"}]'
                )
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    input_segments = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello"},
        {"id": 1, "start": 1.6, "end": 2.5, "text": "Thank you"},
    ]

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        translated = await translate_transcript_segments(
            input_segments,
            source_language="en",
            provider=TranslationProvider.OPENAI,
        )

        assert len(translated) == 2
        assert translated[0]["original_text"] == "Hello"
        assert translated[0]["translated_text"] == "សួស្តី"
        assert translated[1]["original_text"] == "Thank you"
        assert translated[1]["translated_text"] == "អរគុណ"


@pytest.mark.asyncio
async def test_translate_anthropic_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successfully translates segments with mocked Anthropic response."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-key")

    mock_response = SimpleNamespace(
        content=[SimpleNamespace(text='[{"id": 0, "translated_text": "ជម្រាបសួរលោក"}]')]
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    input_segments = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "Greetings Sir"},
    ]

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        translated = await translate_transcript_segments(
            input_segments,
            source_language="en",
            provider=TranslationProvider.ANTHROPIC,
        )

        assert len(translated) == 1
        assert translated[0]["translated_text"] == "ជម្រាបសួរលោក"


def test_can_skip_when_translation_exists(tmp_path: Path) -> None:
    """can_skip returns True when translation_path exists and is non-empty."""
    trans_file = tmp_path / "translation.json"
    trans_file.write_text("[]", encoding="utf-8")

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        translation_path=trans_file,
    )
    stage = TranslationStage()
    assert stage.can_skip(ctx) is True


def test_can_skip_when_missing_or_empty(tmp_path: Path) -> None:
    """can_skip returns False when translation_path is None or 0 bytes."""
    stage = TranslationStage()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        translation_path=None,
    )
    assert stage.can_skip(ctx) is False

    empty_file = tmp_path / "empty.json"
    empty_file.touch()
    ctx.translation_path = empty_file
    assert stage.can_skip(ctx) is False


@pytest.mark.asyncio
async def test_translation_stage_missing_transcript(tmp_path: Path) -> None:
    """Stage fails cleanly if transcript_path is None or does not exist."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        transcript_path=None,
    )
    stage = TranslationStage()
    result = await stage.run(ctx)
    assert result.success is False
    assert "Transcript file not found" in result.message


@pytest.mark.asyncio
async def test_translation_stage_empty_transcript(tmp_path: Path) -> None:
    """Stage handles empty transcript file gracefully."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        transcript_path=transcript,
    )
    stage = TranslationStage()
    result = await stage.run(ctx)
    assert result.success is True
    assert ctx.translation_path is not None
    assert ctx.translation_path.exists()

    with ctx.translation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == []


@pytest.mark.asyncio
async def test_translation_stage_success(tmp_path: Path) -> None:
    """Stage runs translation service and saves translation output."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 1.0, "text": "Hi"}]),
        encoding="utf-8",
    )

    ctx = StageContext(
        episode_id=5,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        transcript_path=transcript,
        source_language="en",
    )

    mock_translated = [
        {"id": 0, "start": 0.0, "end": 1.0, "original_text": "Hi", "translated_text": "សួស្តី"}
    ]

    stage = TranslationStage()
    with patch(
        "mediadubflow.pipeline.stages.translation.translate_transcript_segments",
        new_callable=AsyncMock,
        return_value=mock_translated,
    ):
        result = await stage.run(ctx)
        assert result.success is True
        assert ctx.translation_path == tmp_path / "work" / "translation_5.json"
        assert ctx.translation_path.exists()

        with ctx.translation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["translated_text"] == "សួស្តី"
