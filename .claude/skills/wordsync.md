# WordSync Architecture Guide

## Folder Structure

```
letrix_leituras_guiadas/
├── config.yaml                   # Main configuration (Pydantic-validated)
├── .env                          # API keys (OPENAI_API_KEY, GOOGLE_API_KEY)
├── pyproject.toml                # Package config (v1.0.0)
│
├── wordsync/                     # Python package
│   ├── cli.py                    # Typer CLI — all commands
│   ├── config.py                 # Pydantic settings + YAML loader
│   ├── transcribe.py             # Whisper transcription (openai/local/whisperx)
│   ├── alignment.py              # Needleman-Wunsch DP text alignment + Gemini verification
│   ├── prosody.py                # Gemini 2.5 Flash prosody analysis
│   ├── classify.py               # Gap classification (keep/fill/review)
│   ├── validate.py               # Heuristic + AI timestamp validation
│   ├── process.py                # Pipeline orchestration (10+ steps)
│   └── build.py                  # Jinja2 HTML generation
│
├── templates/                    # Jinja2 templates for player pages
│   ├── page.html.jinja2          # Main HTML template
│   ├── styles.css                # Player CSS (karaoke highlighting)
│   └── player.js                 # WordSyncPlayer class (JS)
│
├── images/                       # Shared header/card SVGs & PNGs
│
└── content/                      # All pages
    └── livro3-let5/              # One page
        ├── index.html            # Generated HTML (deploy this)
        ├── timestamps.json       # Word-level timestamps
        ├── audio.mp3             # Converted audio
        ├── styles.css            # Player styles (if not bundled)
        ├── player.js             # Player script (if not bundled)
        ├── images/               # Copied header images
        └── content/              # Source files (NOT deployed)
            ├── text.txt          # Reference text
            ├── referencia.txt    # Bibliographic reference
            └── audio.mp3         # Original audio recording
```

## text.txt Format

```
Eu fui lá não sei aonde
EU FUI LÁ NÃO SEI AONDE,
VISITAR NÃO SEI A QUEM.
VOLTEI ASSIM, NÃO SEI COMO,
GOSTANDO NÃO SEI DE QUEM.
```

- **Line 1** = title (mixed case) — also spoken in audio by default
- **Lines 2+** = body text (UPPERCASE for poetry/reading), one verse per line
- Line breaks in text.txt create visible line breaks in the HTML output
- First non-empty line is always parsed as the title

## Pipeline Steps (process.py → `process_sync()`)

| # | Step | Module | API |
|---|------|--------|-----|
| 1 | Transcribe audio | `transcribe.py` | OpenAI Whisper |
| 2 | Align to reference text | `alignment.py` | Needleman-Wunsch DP |
| 3 | Verify title + missing words | `alignment.py` | Gemini 2.5 Flash |
| 4 | Correct word durations | `prosody.py` | Gemini 2.5 Flash |
| 5 | Fix overlapping words | `process.py` | — |
| 6 | Prosody analysis (breath, pitch) | `prosody.py` | Gemini 2.5 Flash |
| 7 | Classify gaps (keep/fill) | `classify.py` | — |
| 8 | Cross-validate timestamps | `validate.py` | Gemini / GPT-4o |
| 9 | Apply classifications + finalize | `process.py` | — |
| 10 | Enforce minimum duration (200ms) | `process.py` | — |
| 11 | Heuristic validation | `validate.py` | — |
| 12 | Add unspoken title (if needed) | `process.py` | — |
| 13 | Calculate quality metrics | `process.py` | — |

## Module Map

| Module | Purpose |
|--------|---------|
| `cli.py` | Typer CLI with Rich output — sync, build, restyle, batch, preview, validate, info |
| `config.py` | Pydantic Settings + YAML config loader. Key classes: Settings, TranscriptionSettings, ProsodySettings, GapThresholds, PortugueseRules |
| `transcribe.py` | Audio→words with timestamps. Classes: Word (word/start/end/confidence), TranscriptionResult. Providers: openai, local, whisperx |
| `alignment.py` | Needleman-Wunsch global alignment of reference↔transcribed words. Classes: AlignedWord, TextStructure, AlignmentResult. Gemini verifies title and missing words |
| `prosody.py` | Gemini 2.5 Flash native audio analysis. Detects breath pauses, pitch resets, emphasis. Classes: ProsodyEvent, GapAnalysis, ProsodyResult, WordDurationCorrection |
| `classify.py` | Gap classification using duration thresholds + punctuation + prosody + Portuguese liaison rules. Enums: GapAction (KEEP/FILL/REVIEW), GapType (MICRO/SHORT/MEDIUM/NATURAL/SENTENCE) |
| `validate.py` | Heuristic validation (free) + AI cross-validation. Flags: TOO_SHORT, TOO_LONG, LARGE_GAP, OVERLAP, SYLLABLE_MISMATCH |
| `process.py` | Orchestrates pipeline. Classes: SyncResult (final output with words, metrics, prosody), QualityMetrics. Functions: process_sync(), discover_pages() |
| `build.py` | Jinja2 HTML generation. Functions: build_page(), build_batch(). Copies audio, CSS/JS, images. Reads referencia.txt. Handles bundle_assets embedding |

## Template & Player Architecture

### HTML Structure (page.html.jinja2)

- **Header**: decorative SVGs + logo
- **Player card**: play/pause button, timeline bar, time display, avatar
- **Karaoke text**:
  - Title in `<h1>` — word spans with `data-start`/`data-end` attributes
  - Body in `#transcript` — word spans with timestamps
  - Reference section (from referencia.txt)
  - Hidden metrics div (Ctrl+M to toggle)

### Word Rendering

Each word is a `<span class="word">` with attributes:
```html
<span class="word" data-start="0.12" data-end="0.45" data-confidence="0.95">word</span>
```

### CSS Classes for Words

| Class | Meaning |
|-------|---------|
| `.word` | Base word span |
| `.word.active` | Currently playing word (bright yellow highlight) |
| `.word.highlight` | Upcoming word (subtle highlight) |
| `.word.read` | Already played word (stays highlighted) |
| `.word.no-sync` | Unspoken title word (no timestamps, displayed but not synced) |

### Title Mirror System (player.js)

`WordSyncPlayer.buildTitleMirror()` maps title words in `<h1>` to matching body words in `#transcript`. When a body word highlights, the corresponding title word mirrors the highlight state. This creates dual-highlighting of the same word in both title and body.

### Key CSS Variables

```css
--color-player: #fdbe3f;        /* Yellow player controls */
--color-highlight: #ffcb18;     /* Karaoke highlight color */
--color-bg-page: #f9eacd;       /* Beige page background */
--font-primary: "Montserrat", sans-serif;
```

## Config Essentials (config.yaml)

```yaml
transcription:
  provider: "local"              # IMPORTANT: change to "openai" or use --provider openai
  whisper_model: "whisper-1"

prosody:
  enabled: true
  model: "gemini-2.5-flash"

validation:
  enabled: false                 # Disabled by default

output:
  bundle_assets: true            # CSS/JS embedded in HTML

gap_classification:
  micro_gap_max: 50              # ms — always fill
  short_gap_max: 150             # ms — usually fill
  medium_gap_max: 400            # ms — context dependent
  natural_pause_max: 600         # ms — usually keep
  sentence_boundary_min: 600     # ms — always keep

quality:
  min_confidence: 0.90
  min_word_duration_ms: 200
```

**API Keys** (in `.env`):
- `OPENAI_API_KEY` — required for Whisper transcription
- `GOOGLE_API_KEY` — required for Gemini prosody/verification
- `ANTHROPIC_API_KEY` — optional

## Common Workflows

### Adding a New Page

```bash
# 1. Create source directory
mkdir -p content/livro3-letXX/content

# 2. Add source files
#    - content/livro3-letXX/content/text.txt      (title + body)
#    - content/livro3-letXX/content/referencia.txt (bibliography)
#    - content/livro3-letXX/content/audio.mp3      (recording)

# 3. Sync (generates index.html, timestamps.json, copies assets)
wordsync sync content/livro3-letXX/content/audio.mp3 \
              content/livro3-letXX/content/text.txt \
              --provider openai \
              -o content/livro3-letXX
```

### Design Iteration (Templates)

```bash
# Edit templates/styles.css, templates/player.js, or templates/page.html.jinja2
# Then rebuild all pages with:
wordsync restyle

# Or rebuild a single page:
wordsync restyle livro3-let5
```

### Batch Processing

```bash
# Process all pages in content/ that have content/audio.mp3
wordsync batch --provider openai

# Without prosody analysis (faster, fewer API calls)
wordsync batch --provider openai --no-prosody
```

### Previewing Pages

```bash
# Start local HTTP server and open browser
wordsync preview

# Preview specific page
wordsync preview livro3-let5

# Custom port
wordsync preview --port 3000
```

### Checking Quality

```bash
# Validate timestamps for a page
wordsync validate content/livro3-let5/timestamps.json

# Verbose output (show each word)
wordsync validate content/livro3-let5/timestamps.json --verbose

# Check config & API status
wordsync info
```

## CLI Reference

### `wordsync sync <audio> <text>`
Process audio+text pair through full pipeline.

| Flag | Short | Description |
|------|-------|-------------|
| `--output` | `-o` | Output directory |
| `--title` | `-t` | Override title |
| `--provider` | `-p` | `openai` / `local` / `whisperx` |
| `--local` | `-l` | Use local Whisper (no API) |
| `--no-prosody` | | Disable Gemini prosody |
| `--no-validate` | | Disable AI validation |
| `--json-only` | | Output JSON only (no HTML) |
| `--skip-title` | | Title is not spoken in audio |

### `wordsync build <page-id>`
Rebuild HTML from existing timestamps.json.

| Flag | Short | Description |
|------|-------|-------------|
| `--output` | `-o` | Output path |
| `--embed-audio` | | Embed audio as base64 |

### `wordsync restyle [page-id]`
Rebuild HTML with current templates. Omit page-id to restyle all pages.

### `wordsync batch`
Process all pages discovered in content/.

| Flag | Short | Description |
|------|-------|-------------|
| `--content` | `-c` | Content directory |
| `--output` | `-o` | Output directory |
| `--no-prosody` | | Disable prosody |
| `--no-validate` | | Disable validation |

### `wordsync preview [page-id]`
Start local HTTP server for previewing.

| Flag | Short | Description |
|------|-------|-------------|
| `--port` | `-p` | Server port (default: 8000) |

### `wordsync validate <timestamps.json>`
Check timestamp quality.

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Show detailed word analysis |

### `wordsync info`
Show configuration, API key status, and paths.

### Global Flags
| Flag | Short | Description |
|------|-------|-------------|
| `--version` | `-v` | Show version |
| `--help` | `-h` | Show help |
