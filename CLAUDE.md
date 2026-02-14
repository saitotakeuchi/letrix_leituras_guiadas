# Letrix Leituras Guiadas - WordSync Engine

Karaoke-style guided reading pages that sync highlighted words to audio using Whisper transcription + Gemini prosody analysis.

## Critical Rules

- **Title is ALWAYS spoken in audio** — the `skip_title_audio` flag is auto-overridden when the title exists in `text.txt`. Only use `--skip-title` when the audio genuinely does not contain the title.
- **`bundle_assets: true`** (default in config.yaml) — CSS/JS are embedded directly in HTML. After editing templates (`templates/styles.css`, `templates/player.js`, `templates/page.html.jinja2`), run `wordsync restyle` to rebuild all pages.
- **`content/<page>/content/`** = source files (text.txt, referencia.txt, audio) — NOT deployed. `content/<page>/` root = generated output (index.html, timestamps.json, audio.mp3, images/).
- **Provider: OpenAI Whisper API** — requires `OPENAI_API_KEY` in `.env`. Config default is `"local"` but local whisper is not installed. Always use `--provider openai` or change `transcription.provider` in `config.yaml`.
- **Gemini** is used for prosody analysis, duration correction, and word verification — requires `GOOGLE_API_KEY` in `.env`.

## Key Commands

```bash
wordsync sync <audio> <text> --provider openai    # Process audio+text pair
wordsync build <page-id>                           # Rebuild HTML from timestamps.json
wordsync restyle [page-id]                         # Rebuild HTML with updated templates (all pages if omitted)
wordsync batch                                     # Process all pages in content/
wordsync preview [page-id]                         # Local HTTP preview server
wordsync validate <timestamps.json>                # Check timestamp quality
wordsync info                                      # Show config & API key status
```

## Quick Reference

- **text.txt format**: Line 1 = title (mixed case), lines 2+ = body (UPPERCASE). One verse per line.
- **Python package**: `wordsync/` — CLI (Typer), config (Pydantic), pipeline modules.
- **Templates**: `templates/` — Jinja2 HTML, CSS, JS for the karaoke player.

Run `/wordsync` for the full architecture guide.
