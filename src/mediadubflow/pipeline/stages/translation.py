"""Stage 5 — Translate transcript lines to Khmer via LLM."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from mediadubflow.config.settings import settings
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.services.translation_service import (
    translate_title_to_khmer,
    translate_transcript_segments,
)


async def _load_previous_episodes_context(ctx: StageContext) -> str:
    """
    Load dialogue and translation context from prior episodes in the same project.
    Provides narrative continuity, character relationships, and pronoun consistency to the LLM.
    If an earlier episode failed, was skipped, or is missing artifacts, it is safely skipped.
    """
    if not ctx.project_id:
        return ""

    try:
        from mediadubflow.database.repository import list_episodes
        from mediadubflow.database.session import get_session

        async with get_session() as session:
            project_episodes = await list_episodes(session, ctx.project_id)
    except Exception as exc:
        logger.debug("[Episode {}] Could not query project episodes for context: {}", ctx.episode_id, exc)
        return ""

    if not project_episodes:
        return ""

    sorted_episodes = sorted(project_episodes, key=lambda e: e.episode_number)

    curr_ep_num = ctx.metadata.get("episode_number")
    if curr_ep_num is None:
        for ep in sorted_episodes:
            if ep.id == ctx.episode_id:
                curr_ep_num = ep.episode_number
                break

    if curr_ep_num is None or curr_ep_num <= 1:
        return ""

    prior_episodes = [ep for ep in sorted_episodes if ep.episode_number < curr_ep_num]
    if not prior_episodes:
        return ""

    # Focus on up to 3 most recent prior episodes to maintain high relevance without token bloat
    recent_priors = prior_episodes[-3:]
    context_blocks: list[str] = []

    for ep in recent_priors:
        ep_label = f"Episode {ep.episode_number}"
        if ep.title:
            ep_label += f" ({ep.title})"

        trans_file: Path | None = None
        if ep.translation_path:
            p = Path(ep.translation_path)
            if p.exists() and p.is_file():
                trans_file = p

        if not trans_file:
            candidate = settings.cache_dir / f"episode_{ep.id}" / f"translation_{ep.id}.json"
            if candidate.exists() and candidate.is_file():
                trans_file = candidate

        dialogue_lines: list[str] = []
        if trans_file:
            try:
                with trans_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    is_immediate_prev = (ep.episode_number == curr_ep_num - 1)
                    max_lines = 35 if is_immediate_prev else 20
                    sample = data[-max_lines:] if len(data) > max_lines else data
                    for item in sample:
                        spk = item.get("speaker") or "Speaker"
                        orig = str(item.get("text", "")).strip()
                        trans = str(item.get("translated_text", "")).strip()
                        if orig and trans:
                            dialogue_lines.append(f"  {spk}: \"{orig}\" -> \"{trans}\"")
                        elif orig:
                            dialogue_lines.append(f"  {spk}: \"{orig}\"")
            except Exception as exc:
                logger.debug(
                    "[Episode {}] Could not read translation for prior episode {}: {}",
                    ctx.episode_id,
                    ep.id,
                    exc,
                )

        if not dialogue_lines and ep.transcript_path:
            p = Path(ep.transcript_path)
            if p.exists() and p.is_file():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and data:
                        sample = data[-25:] if len(data) > 25 else data
                        for item in sample:
                            spk = item.get("speaker") or "Speaker"
                            txt = str(item.get("text", "")).strip()
                            if txt:
                                dialogue_lines.append(f"  {spk}: \"{txt}\"")
                except Exception as exc:
                    logger.debug(
                        "[Episode {}] Could not read transcript for prior episode {}: {}",
                        ctx.episode_id,
                        ep.id,
                        exc,
                    )

        if dialogue_lines:
            block = f"[{ep_label} Dialogue & Translation Context]:\n" + "\n".join(dialogue_lines)
            context_blocks.append(block)
        else:
            logger.debug(
                "[Episode {}] Prior episode {} has no translation or transcript; safely skipped",
                ctx.episode_id,
                ep.id,
            )

    if context_blocks:
        summary_text = "\n\n".join(context_blocks)
        logger.info(
            "[Episode {}] Loaded narrative dialogue context from {} previous episode(s) for continuity",
            ctx.episode_id,
            len(context_blocks),
        )
        return summary_text

    return ""


class TranslationStage(PipelineStage):
    """
    Send transcript segments to the configured LLM for context-aware
    Khmer translation.

    A rolling context window and series glossary are injected into the
    system prompt to maintain consistency across lines and episodes.
    """

    name = "Translation"

    def can_skip(self, ctx: StageContext) -> bool:
        """Skip if translation JSON already exists and is non-empty."""
        if ctx.translation_path is None or not ctx.translation_path.exists():
            return False
        try:
            return ctx.translation_path.stat().st_size > 0
        except OSError:
            return False

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info(
            "[Episode {}] Translating with provider={}",
            ctx.episode_id,
            settings.translation_provider,
        )

        if ctx.transcript_path is None or not ctx.transcript_path.exists():
            msg = f"Transcript file not found: {ctx.transcript_path}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        try:
            with ctx.transcript_path.open("r", encoding="utf-8") as f:
                segments = json.load(f)
        except Exception as exc:
            msg = f"Failed to read transcript file '{ctx.transcript_path}': {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not isinstance(segments, list):
            msg = f"Transcript JSON format invalid (expected list, got {type(segments)})"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if len(segments) > 5000:
            msg = f"Transcript has {len(segments)} segments, exceeding safety limit of 5000 segments per episode."
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        output_json = ctx.work_dir / f"translation_{ctx.episode_id}.json"

        if not segments:
            logger.warning("[Episode {}] Transcript has no segments to translate", ctx.episode_id)
            with output_json.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            ctx.translation_path = output_json
            return StageResult(
                success=True,
                message="Transcript is empty; 0 segments translated",
                context_updates={"translation_path": output_json},
            )

        # Extract optional glossary and sync with series_glossary.json for cross-episode consistency
        glossary: dict[str, str] = {}
        if isinstance(ctx.metadata.get("glossary"), dict):
            glossary.update(ctx.metadata["glossary"])

        # Store series glossary in project cache directory to keep user's export folder clean
        glossary_cache_dir = settings.cache_dir / f"project_{ctx.project_id}"
        glossary_cache_dir.mkdir(parents=True, exist_ok=True)
        series_glossary_file = glossary_cache_dir / "series_glossary.json"

        # Migrate and remove any legacy series_glossary.json from ctx.output_dir
        legacy_glossary_file = ctx.output_dir / "series_glossary.json"
        if legacy_glossary_file.exists():
            try:
                if not series_glossary_file.exists():
                    import shutil  # noqa: PLC0415
                    shutil.copy2(legacy_glossary_file, series_glossary_file)
                legacy_glossary_file.unlink()
            except OSError:
                pass

        if series_glossary_file.exists():
            try:
                with series_glossary_file.open("r", encoding="utf-8") as gf:
                    loaded = json.load(gf)
                    if isinstance(loaded, dict):
                        for k, v in loaded.items():
                            if k not in glossary:
                                glossary[k] = str(v)
            except Exception as exc:
                logger.warning("[Episode {}] Could not read series_glossary.json: {}", ctx.episode_id, exc)

        active_glossary = glossary if glossary else None

        # Load narrative and dialogue context from previous episode(s) in this series
        previous_context = await _load_previous_episodes_context(ctx)

        try:
            translated_segments = await translate_transcript_segments(
                segments,
                source_language=ctx.source_language,
                target_language="km",
                glossary=active_glossary,
                previous_context=previous_context,
                progress_callback=ctx.report_progress,
            )
        except Exception as exc:
            msg = f"Translation error: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if glossary:
            try:
                with series_glossary_file.open("w", encoding="utf-8") as gf:
                    json.dump(glossary, gf, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug("[Episode {}] Could not save series_glossary.json: {}", ctx.episode_id, exc)

        try:
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            msg = f"Failed to save translation file '{output_json}': {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        # Translate episode title into natural Khmer drama title
        # Consistency Rule: Translate drama series title once for the first episode,
        # then all subsequent episodes reuse the exact same series title with only the episode number incremented.
        import re  # noqa: PLC0415
        project_cache_dir = settings.cache_dir / f"project_{ctx.project_id}"
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        series_title_file = project_cache_dir / "series_title_km.txt"

        ep_num = int(ctx.metadata.get("episode_number") or 1)
        clean_series_name: str | None = None

        if series_title_file.exists():
            try:
                cached_series = series_title_file.read_text(encoding="utf-8").strip()
                if cached_series and any("\u1780" <= c <= "\u17ff" for c in cached_series):
                    clean_series_name = cached_series
            except OSError:
                pass

        if not clean_series_name:
            khmer_title = ctx.metadata.get("khmer_title") or ctx.metadata.get("title") or getattr(ctx, "title", None)
            if khmer_title and any("\u1780" <= c <= "\u17ff" for c in str(khmer_title)):
                clean_s = re.sub(r"^\[\s*ភាគ\s*[\d០-៩]+\s*\]\s*", "", str(khmer_title), flags=re.IGNORECASE)
                clean_s = re.sub(r"[\s_\-]+(?:ភាគ|ep(?:isode)?)\s*[\d០-៩]+.*$", "", clean_s, flags=re.IGNORECASE).strip()
                if not clean_s.startswith("រឿង"):
                    clean_s = f"រឿង {clean_s}"
                clean_series_name = clean_s
            else:
                raw_title = str(
                    ctx.metadata.get("project_name")
                    or ctx.metadata.get("title")
                    or ctx.source_file.stem
                )
                clean_raw = re.sub(
                    r"[\s_\-]+(?:ep(?:isode)?|part|ភាគ)\s*[\d០-៩]+.*$",
                    "",
                    raw_title.replace("_", " "),
                    flags=re.IGNORECASE,
                ).strip()
                if not clean_raw:
                    clean_raw = raw_title.replace("_", " ").strip()

                try:
                    translated_raw = await translate_title_to_khmer(
                        clean_raw,
                        source_language=ctx.source_language,
                        episode_number=1,
                    )
                    clean_s = re.sub(r"^\[\s*ភាគ\s*[\d០-៩]+\s*\]\s*", "", translated_raw, flags=re.IGNORECASE)
                    clean_s = re.sub(r"[\s_\-]+(?:ភាគ|ep(?:isode)?)\s*[\d០-៩]+.*$", "", clean_s, flags=re.IGNORECASE).strip()
                    if not clean_s.startswith("រឿង"):
                        clean_s = f"រឿង {clean_s}"
                    clean_series_name = clean_s
                except Exception as exc:
                    logger.warning(
                        "[Episode {}] Series title translation failed ({}), using fallback",
                        ctx.episode_id,
                        exc,
                    )
                    prefix = "" if clean_raw.startswith("រឿង") else "រឿង "
                    clean_series_name = f"{prefix}{clean_raw}"

            try:
                series_title_file.write_text(clean_series_name, encoding="utf-8")
                logger.info(
                    "[Episode {}] Persisted unified series title for project {}: '{}'",
                    ctx.episode_id,
                    ctx.project_id,
                    clean_series_name,
                )
            except OSError as exc:
                logger.debug("Could not cache series title: {}", exc)

        khmer_title = f"{clean_series_name} - ភាគ {ep_num:02d}"
        ctx.metadata["khmer_title"] = khmer_title
        ctx.metadata["title"] = khmer_title
        ctx.metadata["series_title"] = clean_series_name
        ctx.title = khmer_title

        ctx.translation_path = output_json
        msg = (
            f"Successfully translated {len(translated_segments)} segments to Khmer ({output_json})"
        )
        logger.info("[Episode {}] {}", ctx.episode_id, msg)

        context_updates: dict[str, object] = {"translation_path": output_json}
        if khmer_title:
            context_updates["title"] = khmer_title

        return StageResult(
            success=True,
            message=msg,
            context_updates=context_updates,
        )
