# WordSync Engine

**Karaoke Word-Sync for Letrix Leituras Guiadas**

A multi-modal LLM-powered engine for generating accurate word-level timestamps for karaoke-style guided reading applications.

## Features

- **Multi-modal Processing Pipeline**
  - Whisper API for primary transcription
  - Gemini 2.5 Flash for prosodic analysis (breath pauses, pitch resets)
  - GPT-4o for cross-validation (90% fewer hallucinations)
  - Intelligent gap classification (keep natural pauses, fill artifacts)

- **Portuguese Language Optimized**
  - Article + noun liaison rules ("o amigo")
  - Preposition connections ("de água")
  - Educational pacing preservation

- **CLI-Driven Workflow**
  - Process single files or batch operations
  - Preview server for testing
  - Quality validation tools

## Installation

```bash
# Install from source
pip install -e .

# With local Whisper support (offline mode)
pip install -e ".[local]"

# Development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: OpenAI for Whisper + GPT-4o
OPENAI_API_KEY=sk-...

# Recommended: Google for Gemini prosody analysis
GOOGLE_API_KEY=AIza...

# Optional: Anthropic for Claude fallback
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Process Audio

```bash
# Single file
wordsync sync audio.mp3 text.txt -o output/

# With page title
wordsync sync audio.mp3 text.txt -t "My Story" -o output/

# Local Whisper (offline, no API)
wordsync sync audio.mp3 text.txt --local
```

### 3. Preview Results

```bash
# Start preview server
wordsync preview page-001

# Or preview all pages
wordsync preview
```

### 4. Batch Processing

```bash
# Process all pages in content/
wordsync batch

# Custom directories
wordsync batch --content ./my-content --output ./my-output
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `wordsync sync <audio> <text>` | Process single audio/text pair |
| `wordsync build <page-id>` | Rebuild HTML from existing timestamps |
| `wordsync batch` | Process all pages in content directory |
| `wordsync preview [page-id]` | Start local preview server |
| `wordsync validate <timestamps.json>` | Check quality metrics |
| `wordsync info` | Show configuration and API status |

### Sync Options

```bash
wordsync sync audio.mp3 text.txt \
  --output ./output \           # Output directory
  --title "My Story" \          # Page title
  --no-prosody \                # Disable Gemini analysis
  --no-validate \               # Disable GPT-4o validation
  --local \                     # Use local Whisper
  --json-only                   # Output JSON only, no HTML
```

## Project Structure

```
letrix_leituras_guiadas/
├── wordsync/                    # Python package
│   ├── __init__.py
│   ├── cli.py                   # CLI commands (Typer)
│   ├── config.py                # Settings (Pydantic)
│   ├── transcribe.py            # Whisper integration
│   ├── prosody.py               # Gemini prosody analysis
│   ├── validate.py              # GPT-4o validation
│   ├── classify.py              # Gap classification
│   ├── process.py               # Pipeline orchestration
│   └── build.py                 # HTML generation
│
├── templates/                   # HTML/CSS/JS templates
│   ├── page.html.jinja2
│   ├── styles.css
│   └── player.js
│
├── content/                     # Input files
│   └── page-XXX/
│       ├── audio.mp3
│       └── text.txt
│
├── output/                      # Generated pages
│   └── page-XXX/
│       ├── index.html
│       └── timestamps.json
│
├── .env.example                 # API key template
├── config.yaml                  # Project configuration
└── pyproject.toml               # Python dependencies
```

## Content Directory Structure

Place your audio/text pairs in the `content/` directory:

```
content/
├── page-001/
│   ├── audio.mp3
│   └── text.txt
├── page-002/
│   ├── audio.mp3
│   └── text.txt
└── ...
```

The engine will auto-discover pages and process them in order.

## Configuration

### config.yaml

```yaml
# General settings
project:
  name: "My Project"
  language: "pt"

# Gap classification thresholds (ms)
gap_classification:
  micro_gap_max: 50      # Always fill
  short_gap_max: 150     # Usually fill
  medium_gap_max: 400    # Context-dependent
  natural_pause_max: 600 # Usually keep
  sentence_boundary_min: 600  # Always keep

# Portuguese rules
portuguese_rules:
  article_noun_liaison: true
  preposition_liaison: true
  articles: ["o", "a", "os", "as", "um", "uma"]
  prepositions: ["de", "da", "do", "em", "na", "no", "para"]

# Output settings
output:
  include_json: true
  embed_audio: false
  bundle_assets: true
```

## Pipeline Architecture

```
INPUT                    PROCESSING                                    OUTPUT
─────                    ──────────                                    ──────
audio.mp3  ──┐
             ├──► Whisper ──► Gemini Audio ──► Gap Classifier ──► HTML + JSON
text.txt   ──┘    (base)      (prosody)        (keep/fill)
                     │            │
                     └────────────┴──► GPT-4o Transcribe (validation)
```

### Gap Classification Logic

| Gap Type | Duration | Audio Evidence | Action |
|----------|----------|----------------|--------|
| Micro-gap | <50ms | - | Always FILL |
| Short gap | 50-150ms | No breath/pitch | FILL |
| Medium gap | 150-400ms | Context-dependent | Analyze |
| Natural pause | 400-600ms | Breath or pitch | KEEP |
| Sentence boundary | >600ms | Punctuation + audio | KEEP |

## API Providers

### Required: OpenAI
- **Whisper API**: Primary transcription with word timestamps
- **GPT-4o Audio**: Cross-validation (optional but recommended)

### Recommended: Google
- **Gemini 2.5 Flash**: Prosodic analysis (breath pauses, pitch resets)

### Optional: Anthropic
- **Claude**: Text analysis fallback

## Quality Metrics

Output includes quality metrics:

```json
{
  "average_confidence": 0.94,
  "gaps_preserved": 8,
  "gaps_filled": 12,
  "low_confidence_words": 2,
  "prosody_preserved_score": 0.98,
  "timing_precision_ms": 45
}
```

Validate with:

```bash
wordsync validate output/page-001/timestamps.json
```

## Fallback Strategy

When APIs are unavailable:

| Tier | Condition | Strategy |
|------|-----------|----------|
| 1 | Gemini unavailable | Duration heuristics + punctuation rules |
| 2 | GPT-4o unavailable | Single-source Whisper with lower confidence |
| 3 | All APIs down | Local whisper-timestamped + Portuguese rules |

## API Cost Estimate

| Service | Per Page | 45 Pages |
|---------|----------|----------|
| Whisper API | ~$0.006 | ~$0.27 |
| Gemini 2.5 Flash | ~$0.02 | ~$0.90 |
| GPT-4o Audio | ~$0.02 | ~$0.90 |
| **Total** | ~$0.05 | **~$2.07** |

## Keyboard Shortcuts (Player)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| ← | Seek back 5s |
| → | Seek forward 5s |
| ↑ | Previous word |
| ↓ | Next word |
| Home | Go to start |
| End | Go to end |
| Ctrl+M | Toggle metrics |

## Python API

```python
from wordsync import process_sync, build_page

# Process audio
result = process_sync(
    audio_path="audio.mp3",
    text_path="text.txt",
    title="My Story"
)

# Access results
print(f"Words: {len(result.words)}")
print(f"Confidence: {result.metrics.average_confidence:.1%}")

# Build HTML
build_page(result, "output/index.html")

# Save JSON
result.save_json("output/timestamps.json")
```

## Embedding in WordPress

The generated pages are iframe-ready:

```html
<iframe
  src="https://your-domain.com/leituras/page-001/"
  width="100%"
  height="600"
  frameborder="0"
  allow="autoplay"
></iframe>
```

## Troubleshooting

### "OpenAI API key not configured"

Set `OPENAI_API_KEY` in your `.env` file or use `--local` flag.

### "Local Whisper not installed"

Install local dependencies:
```bash
pip install wordsync[local]
```

### Low confidence warnings

Review flagged words and manually adjust timestamps if needed.

### Audio not playing

Ensure audio files are in a supported format (MP3, WAV, OGG).

## License

MIT License - See LICENSE file for details.

## Credits

- **Whisper**: OpenAI's speech recognition model
- **Gemini**: Google's multimodal AI
- **GPT-4o**: OpenAI's latest multimodal model
- **Typer**: Modern CLI framework
- **Jinja2**: Template engine
