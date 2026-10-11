"""Stage 9 — Render the final Khmer-localized video."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.config.settings import PipelineOutputMode
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import run_ffmpeg


class VideoRenderingStage(PipelineStage):
    """
    Combine the original video stream with the mixed Khmer dubbed audio
    into the final output video (.mp4).
    """

    name = "Video Rendering"

    def can_skip(self, ctx: StageContext) -> bool:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            return True
        if not (
            ctx.output_video_path is not None
            and ctx.output_video_path.exists()
            and ctx.output_video_path.stat().st_size > 0
        ):
            return False

        # If localized Khmer title is expected, ensure existing file name is not an untranslated fallback
        khmer_title = ctx.metadata.get("khmer_title") or ctx.metadata.get("title") or getattr(ctx, "title", None)
        if khmer_title and any("\u1780" <= c <= "\u17ff" for c in str(khmer_title)):
            if not any("\u1780" <= c <= "\u17ff" for c in ctx.output_video_path.name):
                return False

        return True

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping video rendering (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="Video rendering skipped per output mode setting")

        logger.info("[Episode {}] Rendering final dubbed video", ctx.episode_id)

        # Output video goes directly into user-selected output directory
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        stem = ctx.source_file.stem
        ep_num = int(ctx.metadata.get("episode_number") or 1)
        khmer_title = ctx.metadata.get("khmer_title") or ctx.metadata.get("title") or getattr(ctx, "title", None)

        import re  # noqa: PLC0415
        import shutil  # noqa: PLC0415

        # Check if unified series title exists for the project
        project_cache_dir = settings.cache_dir / f"project_{ctx.project_id}"
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        series_title_file = project_cache_dir / "series_title_km.txt"

        clean_drama: str | None = None
        if series_title_file.exists():
            try:
                cached_series = series_title_file.read_text(encoding="utf-8").strip()
                if cached_series and any("\u1780" <= c <= "\u17ff" for c in cached_series):
                    clean_drama = cached_series
            except OSError:
                pass

        if not clean_drama:
            khmer_title = ctx.metadata.get("khmer_title") or ctx.metadata.get("title") or getattr(ctx, "title", None)
            if khmer_title and any("\u1780" <= c <= "\u17ff" for c in str(khmer_title)):
                clean_s = re.sub(r"^\[\s*ភាគ\s*[\d០-៩]+\s*\]\s*", "", str(khmer_title), flags=re.IGNORECASE)
                clean_s = re.sub(r"[\s_\-]+(?:ភាគ|ep(?:isode)?)\s*[\d០-៩]+.*$", "", clean_s, flags=re.IGNORECASE).strip()
                if not clean_s.startswith("រឿង"):
                    clean_s = f"រឿង {clean_s}"
                clean_drama = clean_s
            else:
                raw_title = str(
                    ctx.metadata.get("project_name")
                    or ctx.metadata.get("title")
                    or getattr(ctx, "title", None)
                    or stem
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
                    from mediadubflow.services.translation_service import translate_title_to_khmer  # noqa: PLC0415
                    translated = await translate_title_to_khmer(
                        clean_raw,
                        source_language=ctx.source_language,
                        episode_number=1,
                    )
                    clean_s = re.sub(r"^\[\s*ភាគ\s*[\d០-៩]+\s*\]\s*", "", translated, flags=re.IGNORECASE)
                    clean_s = re.sub(r"[\s_\-]+(?:ភាគ|ep(?:isode)?)\s*[\d០-៩]+.*$", "", clean_s, flags=re.IGNORECASE).strip()
                    if not clean_s.startswith("រឿង"):
                        clean_s = f"រឿង {clean_s}"
                    clean_drama = clean_s
                except Exception as exc:
                    logger.warning(
                        "[Episode {}] Could not translate series title for video rendering: {}",
                        ctx.episode_id,
                        exc,
                    )
                    prefix = "" if clean_raw.startswith("រឿង") else "រឿង "
                    clean_drama = f"{prefix}{clean_raw}"

            try:
                series_title_file.write_text(clean_drama, encoding="utf-8")
                logger.info(
                    "[Episode {}] Persisted series title for video rendering: '{}'",
                    ctx.episode_id,
                    clean_drama,
                )
            except OSError as exc:
                logger.debug("Could not cache series title: {}", exc)

        safe_drama = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", clean_drama).strip(". ")
        ep_prefix = f"[ភាគ {ep_num:02d}]"
        video_filename = f"{ep_prefix} {safe_drama}.mp4"
        output_video = ctx.output_dir / video_filename

        # Determine audio source (prefer mixed audio, fallback to tts audio)
        audio_source = ctx.mixed_audio_path or ctx.tts_audio_path
        if audio_source is None or not audio_source.exists():
            msg = f"No dubbed audio track found to render video for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        logger.info(
            "[Episode {}] Muxing video '{}' with audio '{}' -> '{}'",
            ctx.episode_id,
            ctx.source_file.name,
            audio_source.name,
            output_video.name,
        )

        # Render to scratch directory first to prevent leaving partial/corrupted files in output_dir
        temp_render_path = ctx.work_dir / f"temp_render_{ctx.episode_id}.mp4"
        if temp_render_path.exists():
            try:
                temp_render_path.unlink()
            except OSError:
                pass

        # Fast muxing: copy video stream directly (-c:v copy), encode audio to AAC
        args = [
            "-y",
            "-i",
            str(ctx.source_file),
            "-i",
            str(audio_source),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(temp_render_path),
        ]

        code, _, stderr = await run_ffmpeg(args)
        if code != 0 or not temp_render_path.exists() or temp_render_path.stat().st_size == 0:
            msg = f"Video rendering failed: {stderr[:200]}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            if temp_render_path.exists():
                try:
                    temp_render_path.unlink()
                except OSError:
                    pass
            return StageResult(success=False, message=msg)

        # Move finalized video into destination
        shutil.move(str(temp_render_path), str(output_video))

        # Ensure only the single clean dubbed video exists for this episode in output_dir
        # Remove any stale/untranslated versions or legacy files from earlier attempts
        stale_candidates = [
            ctx.output_dir / f"{stem}_dubbed.mp4",
            ctx.output_dir / f"{stem}.mp4",
            ctx.output_dir / f"episode_{ctx.episode_id}_dubbed.mp4",
            ctx.output_dir / f"episode_{ctx.episode_id}.mp4",
            ctx.output_dir / "series_glossary.json",
        ]
        if ctx.output_video_path and ctx.output_video_path != output_video:
            stale_candidates.append(ctx.output_video_path)

        # Detect and clean up any older/differently-titled video files for this episode number
        try:
            for item in ctx.output_dir.iterdir():
                if item.is_file() and item != output_video and item.suffix.lower() == ".mp4":
                    if item.name.startswith(f"[ភាគ {ep_num:02d}]") or item.name.startswith(f"[ភាគ {ep_num}]"):
                        stale_candidates.append(item)
        except OSError:
            pass

        for stale in stale_candidates:
            if stale.exists() and stale != output_video:
                try:
                    if stale.name == "series_glossary.json":
                        cached_glossary = settings.cache_dir / f"project_{ctx.project_id}" / "series_glossary.json"
                        if not cached_glossary.exists():
                            cached_glossary.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(stale, cached_glossary)
                    stale.unlink()
                    logger.info(
                        "[Episode {}] Removed stale artifact from output folder: {}",
                        ctx.episode_id,
                        stale.name,
                    )
                except OSError as exc:
                    logger.debug(
                        "[Episode {}] Could not remove stale file {}: {}",
                        ctx.episode_id,
                        stale,
                        exc,
                    )

        ctx.output_video_path = output_video
        logger.info("[Episode {}] Final dubbed video rendered at {}", ctx.episode_id, output_video)
        context_updates: dict[str, object] = {"output_video_path": output_video}
        if khmer_title:
            context_updates["title"] = khmer_title

        return StageResult(
            success=True,
            message="Final dubbed video rendered successfully",
            context_updates=context_updates,
        )
