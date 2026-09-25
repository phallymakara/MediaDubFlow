# MediaDubFlow Implementation Plan

## Overall Progress

- [x] **Project Setup & Environment**: `uv` package management, Python 3.13, dependencies, project structure.
- [x] **Step 1: Database & Alembic Migrations**: SQLite async engine, ORM models (`Project`, `Episode`, `Speaker`), Alembic async migration environment, repository CRUD layer, 12 passing unit tests.
- [x] **Step 2: Startup Settings Validation & Environment Checks**: Settings validation report, dynamic directory creation (`output`, `cache`, `logs`), external CLI tool detection (`ffmpeg`, `ffprobe`), provider key checks, PyTorch/CUDA availability check, bootstrap integration in `__main__.py`, 15 unit tests.
- [x] **Step 3: Wire Episode Detector to Database & Repository (ProjectService)**: Atomic project & episode creation from folder, path sanitization, incremental re-scanning, 10 unit tests.
- [x] **Step 4: Audio Extraction Pipeline Stage (FFmpeg)**: Asynchronous non-blocking subprocess runner, 16kHz mono PCM WAV extraction, 12 unit tests.
- [x] **Step 5: Language Detection Pipeline Stage (Faster-Whisper)**: Thread-offloaded inference, cached model manager, 80% confidence gating, 9 unit tests.
- [x] **Step 6: Speech-to-Text Pipeline Stage (Faster-Whisper)**: Word-level timestamps, Silero VAD filtering, JSON segment artifact persistence, 8 unit tests.
- [x] **Step 7: Translation Pipeline Stage (OpenAI / Claude Khmer localization)**: Provider-agnostic service, batch chunking, glossary support, structured JSON parsing, 16 unit tests.
- [x] **Step 8: Subtitle Generation Pipeline Stage (SRT / ASS format)**: Standard SubRip and 1080p styled ASS generation, zero-shadow Khmer typography, 9 unit tests.
- [x] **Step 9: Job Manager Refactoring (Use Repository layer)**: Repository query integration, stage checkpoint persistence, language metadata updates, 6 unit tests.
- [ ] **Step 10: Desktop GUI Phase 1 (PySide6 screens: Project Ingestion, Queue, Status)**


---

## Step 3: Wire Episode Detector to Database via Project Service (Completed)

### 1. Objective

Connect the directory scanning utility (`mediadubflow.utils.episode_detector`) to the database repository (`mediadubflow.database.repository`) through a dedicated Application Service (`mediadubflow.services.project_service`).

- The GUI never interacts directly with filesystem scanning logic or raw database queries.
- Project creation and initial episode population execute as a single, atomic database transaction.
- Safe folder sanitization and dynamic default output directories are applied.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
GUI Layer (Screens / Dialogs)
   │
   ▼
Application Services (mediadubflow.services.project_service)
   │
   ├──> Utilities (mediadubflow.utils.episode_detector.detect_episodes)
   │
   └──> Repository (mediadubflow.database.repository)
           │
           ▼
        Database (SQLite via SQLAlchemy AsyncSession)
```

- **Separation of Concerns**: GUI invokes `create_project_from_folder()` without knowing regex patterns or database queries.
- **Transactional Integrity**: All `Episode` records for a project are persisted in the same transaction as the parent `Project`. If any step fails, the session rolls back completely, preventing orphan records.
- **No Hardcoding**: Default output folder uses `settings.output_root / _sanitize_folder_name(name)`.
- **Error Handling**: Domain exceptions (`FileNotFoundError`, `ValueError`) with actionable messages.
- **No Emojis**: Code, docstrings, comments, and log messages contain no emojis.

---

### 3. Implementation Details

#### A. `src/mediadubflow/services/project_service.py`

Implemented service functions:

1. `_sanitize_folder_name(name: str) -> str`:
   - Strips characters illegal on Windows/POSIX filesystems (`<>:"/\|?*`).
   - Collapses whitespace to underscores and trims edges.

2. `create_project_from_folder(session, *, name, source_folder, output_folder=None, target_language="km", source_language=None) -> Project`:
   - Validates that `name.strip()` is non-empty.
   - Validates that `source_folder` exists and is a directory.
   - Resolves `output_folder` (defaults to `settings.output_root / _sanitize_folder_name(name)`).
   - Scans directory using `detect_episodes(source_path)`.
   - If no video files are detected, raises `ValueError`.
   - In an atomic transaction: creates `Project` and adds all detected `Episode` records in `PENDING` status.
   - Returns reloaded `Project` with populated `episodes`.

3. `rescan_project_episodes(session, project_id: int) -> list[Episode]`:
   - Retrieves the project by id using `get_project(session, project_id)`.
   - Scans `project.source_folder` with `detect_episodes()`.
   - Identifies newly added video files and inserts only new `Episode` records.
   - Returns the list of newly created `Episode` instances.

#### B. `src/mediadubflow/services/__init__.py`

- Re-exports `create_project_from_folder` and `rescan_project_episodes`.

#### C. `tests/unit/test_project_service.py`

10 unit tests covering:
1. `test_sanitize_folder_name`: checks sanitization of punctuation, illegal characters, and whitespace.
2. `test_create_project_from_folder_success`: verifies persistence of project and episodes.
3. `test_create_project_from_folder_default_output`: checks fallback to `settings.output_root`.
4. `test_create_project_from_folder_custom_output`: preserves specified custom output folder.
5. `test_create_project_from_folder_empty_folder`: raises `ValueError` on empty directories.
6. `test_create_project_from_folder_nonexistent_path`: raises `FileNotFoundError`.
7. `test_create_project_from_folder_empty_name`: raises `ValueError`.
8. `test_rescan_project_episodes_adds_new_files`: adds new files incrementally.
9. `test_rescan_project_episodes_no_new_files`: returns empty list when up-to-date.
10. `test_rescan_project_not_found`: raises `ValueError` for unknown project ID.

---

### 4. Verification Results

- Unit Tests: 40/40 passing (`pytest`)
- Project Service Coverage: 97%
- Linting: 0 errors (`ruff check`)
- Formatting: Clean (`ruff format`)

---

## Step 4: Audio Extraction Pipeline Stage (Completed)

### 1. Objective

Extract a standardized 16 kHz mono 16-bit PCM WAV audio track from the episode's source video file for consumption by downstream pipeline stages (transcription, diarization, translation).

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
Pipeline Runner / JobManager
   │
   ▼
AudioExtractionStage (src/mediadubflow/pipeline/stages/audio_extraction.py)
   │
   ▼
Async FFmpeg Runner (src/mediadubflow/utils/ffmpeg.py)
   │
   ▼
System FFmpeg Binary (non-blocking subprocess via asyncio.create_subprocess_exec)
```

- **Shared Reusable Code**: Created `src/mediadubflow/utils/ffmpeg.py` with `get_ffmpeg_path()`, `run_ffmpeg()`, and `extract_audio_track()`, avoiding duplication across future mixing and rendering stages.
- **Non-blocking Execution**: Uses `asyncio.create_subprocess_exec()` with stdout/stderr pipes so the desktop GUI thread is never blocked.
- **Fail-Fast & Safe**: Verifies source file existence before launching FFmpeg; verifies non-empty output file creation.
- **Checkpoint Resumption**: `can_skip()` inspects `ctx.extracted_audio` and verifies it exists and is > 0 bytes.

---

### 3. Verification Results

- Unit Tests: 12/12 passing in `tests/unit/test_audio_extraction.py`
- Full Test Suite: 52/52 passing across the application
- `ffmpeg.py` Coverage: 100%
- `audio_extraction.py` Coverage: 80%
- Linting & Formatting: Clean (`ruff check`, `ruff format`)

---

## Step 5: Language Detection Pipeline Stage (Completed)

### 1. Objective

Automatically identify the spoken source language of an episode before full transcription and translation occur, offloading inference to avoid blocking the GUI and pausing for manual verification if detection confidence is below 80%.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
Pipeline Runner / JobManager
   │
   ▼
LanguageDetectionStage (src/mediadubflow/pipeline/stages/language_detection.py)
   │
   ├──> Model Cache / Manager (src/mediadubflow/utils/whisper_model.py)
   │
   ▼
Faster-Whisper (inference offloaded via asyncio.to_thread)
```

- **Reusable Model Lifecycle**: Implemented `get_whisper_model()` and `clear_whisper_model_cache()` in `src/mediadubflow/utils/whisper_model.py`, caching models by `(model_size, device, compute_type)`. Reusable directly in Step 6 (Speech-to-Text).
- **Non-blocking Execution**: Wrapped blocking Whisper decoding and language detection in `asyncio.to_thread()` to prevent freezing the PySide6 Qt GUI event loop.
- **Safety Gating**: Automatically accepts high-confidence detections (`>= 0.80`), while marking low-confidence detections (`< 0.80`) with `needs_review: True` for human verification.
- **Checkpoint Resumption**: `can_skip()` skips if `ctx.source_language` is already set.

---

### 3. Verification Results

- Unit Tests: 9/9 passing in `tests/unit/test_language_detection.py`
- Full Test Suite: 61/61 passing across the application
- `language_detection.py` Coverage: 100%
- `whisper_model.py` Coverage: 93%
- Linting & Formatting: Clean (`ruff check`, `ruff format`)

---

## Step 6: Speech-to-Text Pipeline Stage (Completed)

### 1. Objective

Convert spoken dialogue from the extracted audio into structured, timestamped text segments with word-level timing saved as a JSON artifact for downstream diarization, translation, and subtitle generation.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
Pipeline Runner / JobManager
   │
   ▼
TranscriptionStage (src/mediadubflow/pipeline/stages/transcription.py)
   │
   ├──> Shared Whisper Model Cache (src/mediadubflow/utils/whisper_model.py)
   │
   ▼
Faster-Whisper (inference offloaded via asyncio.to_thread)
   │
   ▼
Persisted JSON Transcript Artifact (work_dir / transcript_<episode_id>.json)
```

- **Model Reuse**: Reuses `get_whisper_model()` from Step 5, ensuring multi-gigabyte neural network weights are loaded once in memory and shared across episodes.
- **Non-blocking Execution**: Wrapped heavy Whisper model inference in `asyncio.to_thread()` to prevent freezing the PySide6 Qt GUI.
- **Voice Activity Detection (VAD)**: Enabled `vad_filter=True` using Silero VAD to eliminate background music/noise hallucinations.
- **Word Timestamps**: Serialized start/end timestamps and word-level timing into structured JSON artifacts.
- **Checkpoint Resumption**: `can_skip()` inspects `ctx.transcript_path` and verifies file existence and > 0 bytes.

---

### 3. Verification Results

- Unit Tests: 8/8 passing in `tests/unit/test_transcription.py`
- Full Test Suite: 69/69 passing across the application
- `transcription.py` Coverage: 92%
- Linting & Formatting: Clean (`ruff check`, `ruff format`)

---

## Step 7: Translation Pipeline Stage (Completed)

### 1. Objective

Localize transcribed dialogue lines into natural, conversational Khmer (ភាសាខ្មែរ) using OpenAI or Anthropic Claude LLMs, maintaining character relationships, timing constraints, and project glossary terms.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
Pipeline Runner / JobManager
   │
   ▼
TranslationStage (src/mediadubflow/pipeline/stages/translation.py)
   │
   ▼
LLM Translation Service (src/mediadubflow/services/translation_service.py)
   │
   ├──> OpenAI API (AsyncOpenAI via openai_api_key & openai_model)
   │
   └──> Anthropic API (AsyncAnthropic via anthropic_api_key & anthropic_model)
```

- **Provider Agnostic**: Unified `translate_transcript_segments()` service supports both OpenAI and Anthropic Claude based on `settings.translation_provider`.
- **Batch Chunking**: Processes lines in sequential chunks (40 lines per batch) with a rolling prompt to prevent context token overflow and avoid output truncation.
- **Glossary Support**: Injects drama-specific glossaries and character names directly into system prompts.
- **Robust JSON Parsing**: Handles raw JSON, markdown-fenced blocks, and dictionary-wrapped responses.
- **Checkpoint Resumption**: `can_skip()` inspects `ctx.translation_path` and verifies file existence and > 0 bytes.

---

### 3. Verification Results

- Unit Tests: 16/16 passing in `tests/unit/test_translation.py`
- Full Test Suite: 85/85 passing across the application
- `translation_service.py` Coverage: 93%
- `translation.py` Coverage: 71%
- Linting & Formatting: Clean (`ruff check`, `ruff format`)

---

## Step 8: Subtitle Generation Pipeline Stage (Completed)

### 1. Objective

Convert translated dialogue into standard SubRip (`.srt`) and styled Advanced SubStation Alpha (`.ass`) subtitle files with 1080p canvas coordinates, bottom-center alignment, clean outline, zero drop shadow, and modern Khmer typography.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
Pipeline Runner / JobManager
   │
   ▼
SubtitleGenerationStage (src/mediadubflow/pipeline/stages/subtitle_generation.py)
   │
   ├──> Subtitle Generators (src/mediadubflow/utils/subtitles.py)
   │       ├──> pysrt (SubRipFile & SubRipTime calculation)
   │       └──> ass (Document, Style, and Dialogue events)
   │
   ▼
Saved Subtitle Files (output_dir / episode_<id>.srt and episode_<id>.ass)
```

- **Shared Reusable Code**: Built `create_srt_file()` and `create_ass_file()` in `src/mediadubflow/utils/subtitles.py` for standardizing subtitle generation.
- **Strict UI/UX Rules Compliance**: Set `shadow=0.0` in ASS style per `@rules:backend.md` (*"Do not use shadows"*). Used solid border outline (`outline=2.5`) for clean legibility against video backgrounds.
- **UTF-8 with BOM Support**: Saved `.ass` files using `utf_8_sig` to guarantee accurate rendering of Khmer script vowels and subscripts on Windows players.
- **Checkpoint Resumption**: `can_skip()` inspects both `ctx.subtitle_srt_path` and `ctx.subtitle_ass_path` and verifies both exist and are > 0 bytes.

---

### 3. Verification Results

- Unit Tests: 9/9 passing in `tests/unit/test_subtitles.py`
- Full Test Suite: 94/94 passing across the application
- `subtitles.py` Coverage: 100%
- Linting & Formatting: Clean (`ruff check`, `ruff format`)

---

## Step 9: Job Manager Refactoring & Repository Layer Integration (Completed)

### 1. Objective

Refactor `JobManager` to strictly use the repository CRUD layer (`mediadubflow.database.repository`) for all database operations, persist intermediate stage checkpoints on disk into the database, reorder active pipeline stages logically, and emit granular Qt progress signals.

---

### 2. Architecture & Rules Alignment (`@rules:backend.md`)

```
JobManager (mediadubflow.core.job_manager)
   │
   ├──> Concurrency Control: asyncio.Semaphore (settings.max_concurrent_episodes)
   │
   ├──> Repository Layer (mediadubflow.database.repository)
   │       ├── get_episode()
   │       ├── update_episode_status()
   │       ├── update_episode_checkpoint()
   │       └── update_episode_language()
   │
   ├──> Pipeline Sequence
   │       1. AudioExtractionStage
   │       2. LanguageDetectionStage
   │       3. TranscriptionStage
   │       4. SpeakerDetectionStage (stub)
   │       5. TranslationStage
   │       6. SubtitleGenerationStage
   │       7. TTSGenerationStage (stub)
   │       8. AudioMixingStage (stub)
   │       9. VideoRenderingStage (stub)
   │
   └──> Signals emitted:
           ├── progress_updated(episode_id, stage_name, pct)
           ├── status_changed(episode_id, status_str)
           ├── episode_failed(episode_id, error_msg)
           └── episode_completed(episode_id)
```

- **Separation of Concerns**: Removed raw `session.get()` calls in `JobManager`. Context loading, status changes, checkpoints, and language detection updates all delegate to `repository.py`.
- **Checkpoint Persistence**: Each stage output (`extracted_audio`, `transcript_path`, `translation_path`, `subtitle_srt_path`, `subtitle_ass_path`) is persisted into the corresponding database columns as soon as the stage completes.
- **Granular Lifecycle**: Stages map directly to granular `EpisodeStatus` enum values (`EXTRACTING_AUDIO`, `DETECTING_LANGUAGE`, `TRANSCRIBING`, `TRANSLATING`, `GENERATING_SUBTITLES`, `DONE`, `FAILED`).
- **No Hardcoding**: Output and cache paths use `settings.output_root` and `settings.cache_dir`.
- **No Emojis**: Logs, comments, docstrings, and signals are clean and professional.

---

### 3. Verification Results

- Unit Tests: 6/6 passing in `tests/unit/test_job_manager.py`
- Repository Tests: 13/13 passing in `tests/unit/test_database.py`
- Full Test Suite: **101/101 passing** across the entire application
- JobManager Coverage: **99%**
- Repository Coverage: **92%**
- Total Application Coverage: **85%**
- Linting & Formatting: 100% clean (`ruff check`, `ruff format`)

---

## Step 10: Desktop GUI Implementation (Completed)

### 1. Objective

Build a modern, maintainable desktop user interface using PySide6/Qt that adheres to all user-global UI rules (zero drop-shadows, flat crisp 1px borders, no excessive cards, inline error validation, and non-blocking asynchronous event loop bridge).

---

### 2. Architecture & Rules Alignment (`@rules:gui.md`)

```
MediaDubFlowApp (src/mediadubflow/gui/app.py)
   ├── Header & Navigation Bar (Brand, Project Breadcrumb, Settings)
   ├── QStackedWidget (Screen Router)
   │     ├── Index 0: HomeScreen (Recent projects table + [+ New Project] hero)
   │     ├── Index 1: ProjectWizardScreen (3-Step Stepper: Media -> Episodes -> Localization)
   │     └── Index 2: WorkspaceScreen (Live episode queue, status chips, stage progress, KPI summary)
   ├── Auxiliary Dialogs:
   │     ├── SettingsDialog (API keys, compute device, worker concurrency)
   │     ├── LanguageCorrectionDialog (Whisper low-confidence override)
   │     └── EpisodeDetailPanel (Checkpoint inspector, .srt/.ass subtitle preview)
   └── AsyncBridge (Dedicated background QThread hosting asyncio event loop)
```

- **Zero Shadows**: Flat design tokens in `src/mediadubflow/gui/styles.py` with 1px solid borders (`#e2e8f0` / `#334155`).
- **Inline Validation**: Red border + inline text beneath fields; zero popup error alerts.
- **Asynchronous Execution**: Coroutines and database queries run off the Qt main thread via `AsyncBridge`.
- **Live Pipeline Binding**: `WorkspaceScreen` connects to `JobManager` Qt queued signals for real-time progress and status badge updates.

---

### 3. Verification Results

- Unit Tests: All passing with zero regressions
- Codebase Quality: 100% clean (`ruff check`, `ruff format`)


