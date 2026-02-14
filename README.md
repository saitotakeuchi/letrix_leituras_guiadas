# Letrix Leituras Guiadas

Karaoke-style guided reading pages for Letrix. Each page syncs highlighted words to audio using Whisper transcription + Gemini prosody analysis.

## Folder structure

```
content/
  livro3-let5/                   ← S3-ready: upload this folder directly
    index.html                   ← generated HTML page
    timestamps.json              ← word-level timestamps
    audio.mp3                    ← converted audio (build output)
    styles.css                   ← player styles
    player.js                    ← player script
    images/                      ← header/card images
    content/                     ← source files (not deployed)
      text.txt                   ← reference text (one line per verse)
      referencia.txt             ← book/page reference for header
      WhatsApp-Audio-....mp3     ← original audio recording

templates/                       ← Jinja2 templates for HTML generation
wordsync/                        ← Python package (CLI + pipeline)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and add API keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENAI_API_KEY` — Whisper transcription
- `GOOGLE_API_KEY` — Gemini prosody analysis (recommended)

## Creating a new page

1. Create the folder structure:
   ```bash
   mkdir -p content/livro3-letXX/content
   ```

2. Add source files to `content/livro3-letXX/content/`:
   - **audio file** — the recorded reading (mp3, m4a, wav)
   - **text.txt** — reference text, first line is the title, one verse per line:
     ```
     CHARADAS
     NÃO FALO, MAS SEMPRE EXPLICO.
     SOU UM AMIGO CALADO.
     ```
   - **referencia.txt** — book/page reference for the header:
     ```
     Livro 3 — Leitura 5
     ```

3. Run the sync pipeline (see below).

## Syncing a page

Processes audio + text to generate word-level timestamps:

```bash
wordsync sync content/livro3-letXX/content/audio.mp3 content/livro3-letXX/content/text.txt
```

This outputs `timestamps.json`, `index.html`, `audio.mp3`, `styles.css`, `player.js`, and `images/` to `content/livro3-letXX/`.

Options:
```bash
--title "Custom Title"    # Override title from text.txt
--skip-title              # Title is not spoken in the audio
--no-prosody              # Skip Gemini prosody analysis
--local                   # Use local Whisper (no API)
--json-only               # Output timestamps.json only
```

## Rebuilding HTML

Regenerate HTML from existing `timestamps.json` (no re-transcription):

```bash
wordsync build livro3-let5
```

## Batch processing

Process all pages in `content/`:

```bash
wordsync batch
```

## Previewing

Start a local server to test in the browser:

```bash
wordsync preview livro3-let5    # Single page
wordsync preview                # All pages
```

## Configuration

Settings are loaded from `config.yaml`, `.env`, and defaults. See `wordsync info` for current configuration.

Key settings in `config.yaml`:
- `project.name` — project title
- `project.language` — language code (default: `pt`)
- `transcription.provider` — `openai`, `local`, or `whisperx`
- `prosody.enabled` — enable Gemini prosody analysis
- `output.bundle_assets` — inline CSS/JS into HTML

## Deploying to S3

Each page folder is self-contained and S3-ready. Upload the page folder directly:

```bash
aws s3 sync content/livro3-let5/ s3://your-bucket/leituras/livro3-let5/ \
  --exclude "content/*"
```

The `content/` subfolder contains source files and should be excluded from deployment.

## CLI reference

| Command | Description |
|---------|-------------|
| `wordsync sync <audio> <text>` | Process audio/text pair |
| `wordsync build <page-id>` | Rebuild HTML from timestamps.json |
| `wordsync batch` | Process all pages in content/ |
| `wordsync preview [page-id]` | Local preview server |
| `wordsync validate <timestamps.json>` | Check quality metrics |
| `wordsync info` | Show config and API status |
