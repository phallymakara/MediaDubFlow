# MediaDubFlow

**AI-powered desktop tool for automatically localizing movies and drama episodes into Khmer.**

MediaDubFlow takes a folder of video episodes and runs them through a fully automated AI pipeline:
language detection → speech-to-text → speaker detection → Khmer translation → subtitle generation → Khmer TTS → audio mixing → final video render.

---

## Technology Stack

| Area | Technology |
|---|---|
| Desktop GUI | PySide6 / Qt 6 |
| Language | Python 3.13+ |
| Speech-to-Text | faster-whisper (Whisper large-v3) |
| Speaker Detection | pyannote.audio |
| Translation | OpenAI / Anthropic LLM |
| Text-to-Speech | Coqui TTS (Khmer model) |
| Audio / Video | FFmpeg |
| Database | SQLite + SQLAlchemy (async) + Alembic |
| Config | pydantic-settings |
| Logging | loguru |
| Packaging | uv + PyInstaller |

---

## Project Structure

```
MediaDubFlow/
├── src/
│   └── mediadubflow/
│       ├── __main__.py          # Entry point
│       ├── config/              # Settings (pydantic-settings + .env)
│       ├── models/              # SQLAlchemy ORM models
│       ├── database/            # Async session + migrations (Alembic)
│       ├── pipeline/
│       │   ├── base.py          # PipelineStage interface + StageContext
│       │   └── stages/          # One module per pipeline stage
│       ├── core/
│       │   └── job_manager.py   # Async episode queue + concurrency control
│       ├── services/            # Provider-agnostic AI service wrappers
│       ├── gui/                 # PySide6 application window + screens
│       └── utils/               # Shared helpers (episode detection, logging…)
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/                     # Dev/ops helper scripts
├── logs/                        # Runtime log files (git-ignored)
├── .env.example                 # Environment variable template
├── pyproject.toml               # uv project config + dependencies
└── Documents/                   # Project proposal
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- FFmpeg installed and available on `PATH`
- CUDA toolkit (optional, for GPU acceleration)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd MediaDubFlow

# 2. Create virtual environment and install all dependencies
uv sync --dev

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings

# 4. Run the application
uv run mediadubflow
```

---

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

---

## Phased Roadmap

| Phase | Scope |
|---|---|
| **MVP (Phase 1)** | Single-episode: language detection → transcription → translation → .srt export |
| **Phase 2** | Speaker detection, Khmer TTS, voice assignment, audio mixing, single-episode dubbed video |
| **Phase 3** | Batch processing, episode queue, checkpoint/resume, concurrency controls, series glossary |
| **Phase 4** | QC dashboard, manual review UI, voice consistency, Windows + macOS packaging |

---

## Configuration

All sensitive configuration (API keys, tokens) must be provided via `.env`.
See [`.env.example`](.env.example) for all available options.

**Never commit `.env` to version control.**
