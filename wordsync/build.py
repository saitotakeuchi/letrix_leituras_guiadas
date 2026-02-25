"""
WordSync Build Module

Generates HTML pages with embedded word timestamps for karaoke playback.
Uses Jinja2 templates based on the existing Letrix design.

Output formats:
- Self-contained HTML with embedded CSS/JS
- JSON with word timestamps
- Quality metrics report
"""

import base64
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

import html as html_module

from wordsync.config import Settings, get_settings
from wordsync.process import SyncResult, discover_pages, load_sync_result
from wordsync.transcribe import get_audio_mime_type


def _format_referencia(text: str) -> str:
    """Apply ABNT italic formatting to a bibliographic reference.

    Detects reference type and wraps the appropriate title in <em> tags.
    Returns HTML-safe string with <em> around the italicized portion.
    """
    stripped = text.strip()

    # Skip: very short, "Cultura popular", "sem referência."
    if len(stripped) < 15 or stripped.lower() in ("cultura popular", "sem referência."):
        return html_module.escape(stripped)

    safe = html_module.escape(stripped)

    # --- 1. "Texto não publicado" ---
    texto_match = re.search(r'\.\s*Texto não publicado', stripped)
    if texto_match:
        before = stripped[:texto_match.start()]
        author_end = _find_author_end(before)
        title = before[author_end:].strip().rstrip('.')
        if title:
            safe_title = html_module.escape(title)
            safe = safe.replace(safe_title, '<em>' + safe_title + '</em>', 1)
        return safe

    # --- 2. "In:" references ---
    in_match = re.search(r'\.\s*In:\s*', stripped)
    if in_match:
        after_in = stripped[in_match.end():]

        # Author after "In:" — ALL-CAPS name ending with period
        author_after = re.match(
            r'((?:[A-ZÀ-Ú]{2,}(?:,\s*[^.]+)?\.\s*)+)',
            after_in,
        )
        title_start = author_after.end() if author_after else 0
        rest = after_in[title_start:]

        # Title ends at boundary
        end_m = re.search(
            r'\.\s+(?=\d+\.\s*ed\.)'
            r'|\.\s+(?=s\.\s*p\.)'
            r'|\.\s+(?=\d+\s+CD\b)'
            r'|\.\s+(?=[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*\s*:)'
            r'|,\s+(?=p\.\s*\d)'
            r'|,\s+(?=[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*\s*,)'
            r'|\.\s*$',
            rest,
        )
        book_title = rest[:end_m.start()].strip() if end_m else rest.strip()
        book_title = book_title.rstrip('.')

        # If title has internal sentence boundaries ("? " or ". ") and the
        # last segment is short + followed by metadata, it's a periodical name
        # e.g., "QUE PAPO é este? Folhinha" → just "Folhinha"
        for sep in ('? ', '. '):
            if sep in book_title:
                last_seg = book_title.rsplit(sep, 1)[-1].strip()
                if last_seg and len(last_seg) < 30:
                    book_title = last_seg
                    break

        if book_title:
            safe_title = html_module.escape(book_title)
            safe_in = html_module.escape(in_match.group().strip())
            safe = safe.replace(safe_in, '<em>' + safe_in + '</em>', 1)
            safe = safe.replace(safe_title, '<em>' + safe_title + '</em>', 1)
        return safe

    # --- 3. "Disponível em:" (online documents — checked before book/periodical) ---
    if 'Disponível em:' in stripped:
        disp_pos = stripped.index('Disponível em:')
        before = stripped[:disp_pos].rstrip()
        # Remove trailing period/space but remember if ? or !
        trail_char = before[-1] if before else ''
        before_clean = before.rstrip('.?! ')
        # If there's also publication info (City: Publisher) before Disponível,
        # extract title between author and pub info
        city_in_before = re.search(
            r'[.,;!?]\s*[A-ZÀ-Ú][a-zà-ú]+\s*:\s*[A-ZÀ-Ú]',
            before_clean,
        )
        if city_in_before:
            # Title is between author end and city:publisher
            before_city = before_clean[:city_in_before.start()]
            author_end = _find_author_end(before_city)
            title = before_city[author_end:].strip().rstrip('.')
        else:
            author_end = _find_author_end(before_clean)
            title = before_clean[author_end:].strip().rstrip('.')
        # Restore trailing ? or ! (part of the title)
        if trail_char in '?!' and not title.endswith(trail_char):
            title = title + trail_char
        if title:
            safe_title = html_module.escape(title)
            safe = safe.replace(safe_title, '<em>' + safe_title + '</em>', 1)
        return safe

    # --- 4. Periodical articles ---
    # Periodical name appears after article title, followed by ", City,"
    # or ", metadata". Must NOT have "City: Publisher" pattern.
    has_city_pub = re.search(r'[A-ZÀ-Ú][a-zà-ú]+\s*:\s*[A-ZÀ-Ú]', stripped)
    if not has_city_pub:
        # Find each ". " or "? " boundary, check if what follows is
        # "PeriodicalName, ..." with ABNT metadata. Use the LAST matching
        # boundary (closest to the metadata), not the first.
        best_name = None
        for m in re.finditer(r'[.?!]\s+', stripped):
            after = stripped[m.end():]
            comma = after.find(',')
            if comma <= 0:
                continue
            name = after[:comma].strip()
            rest_after = after[comma + 1:]
            if not name or not name[0].isupper() or len(name) > 50:
                continue
            if re.search(
                r'\b(?:ano|n\.|p\.|v\.)\s*\d'
                r'|(?:jan|fev|mar|abr|maio|jun|jul|ago|set|out|nov|dez)\b',
                rest_after,
                re.IGNORECASE,
            ):
                best_name = name
        if best_name:
            safe_name = html_module.escape(best_name)
            safe = safe.replace(safe_name, '<em>' + safe_name + '</em>', 1)
            return safe

    # --- 5. Book references (City: Publisher pattern) ---
    # City names may have lowercase prepositions: "Rio de Janeiro", "Santa Cruz"
    city_pub_match = re.search(
        r'[.,;!?]\s*(?:\d+\.\s*ed\.\s+)?'
        r'[A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:de|da|do|dos|das|[A-ZÀ-Ú])[a-zà-ú]*)*'
        r'\s*:\s*[A-ZÀ-Ú]',
        stripped,
    )
    if city_pub_match:
        before_city = stripped[:city_pub_match.start()]
        # Include trailing ! or ? in the title (e.g., "Adivinhe se puder!")
        trail = stripped[city_pub_match.start():city_pub_match.start() + 1]
        if trail in '!?':
            before_city += trail
        author_end = _find_author_end(before_city)
        title = before_city[author_end:].strip().rstrip('.')
        # Strip "; ilustrações..." or similar appendages
        semi_pos = title.find(';')
        if semi_pos > 0:
            title = title[:semi_pos].strip()
        ilust = re.search(r'\.\s+[Ii]lustra', title)
        if ilust:
            title = title[:ilust.start()].strip()
        # No-author book: "ENTRY. BookTitle. City:" — take last ". "-segment
        if author_end == 0 and '. ' in title:
            title = title.rsplit('. ', 1)[-1]
        if title:
            safe_title = html_module.escape(title)
            safe = safe.replace(safe_title, '<em>' + safe_title + '</em>', 1)
        return safe

    return safe


def _find_author_end(text: str) -> int:
    """Find where the author block ends in a reference string.

    Scans ". " positions to find the first one that ends the author block
    (i.e., is followed by the title, not by another author name).
    Returns 0 for no-author entries.
    """
    # No-author patterns: ALL-CAPS word followed by lowercase
    # "6 CURIOSIDADES sobre...", "CONHEÇA Polly...", "PAPO de gato..."
    # "MAMÍFEROS voadores...", "JOGOS e brincadeiras..."
    if re.match(r'(?:\d+\s+)?[A-ZÀ-Ú]+\s+[a-zà-ú]', text):
        return 0
    # "A ORIGEM dos..."
    if re.match(r'[A-ZÀ-Ú]\s+[A-ZÀ-Ú]+\s+[a-zà-ú]', text):
        return 0

    # Institutional: "BRASIL. Ministério da Educação. Secretaria..."
    # Each segment after ALL-CAPS starts with a capitalized word
    inst = re.match(r'[A-ZÀ-Ú]{2,}\.\s+', text)
    if inst:
        pos = inst.end()
        while pos < len(text):
            next_seg = re.match(r'([^.]+)\.\s*', text[pos:])
            if not next_seg:
                break
            seg_text = next_seg.group(1).strip()
            if re.match(r'[A-ZÀ-Ú][a-zà-ú]', seg_text):
                pos += next_seg.end()
            else:
                break
        return pos

    # Standard/multi-author: scan ". " boundaries
    # An author-internal period is one that follows a single-letter initial
    # (e.g., "A." in "Josca A.") or a preposition (e.g., "de." in "S. de.").
    # The first ". " NOT followed by an ALL-CAPS surname and NOT an
    # initial/preposition ends the author block.
    last_author_dot = -1  # track last ". " that was part of author
    pos = 0
    while True:
        dot = text.find('. ', pos)
        if dot == -1:
            break

        after = text[dot + 2:]

        # Single-letter initial before dot: author-internal
        if dot >= 1 and text[dot - 1].isalpha() and (dot < 2 or text[dot - 2] in ' ,;'):
            last_author_dot = dot
            pos = dot + 2
            continue

        # Preposition before dot: author-internal
        pre_word = re.search(r'(\w+)\s*$', text[:dot])
        if pre_word and pre_word.group(1).lower() in ('de', 'da', 'das', 'dos', 'do'):
            last_author_dot = dot
            pos = dot + 2
            continue

        # What follows is ALL-CAPS surname? More authors ahead
        if re.match(r'[A-ZÀ-Ú]{2,}[,.\s;]', after):
            last_author_dot = dot
            pos = dot + 2
            continue

        # This ". " ends the author block
        return dot + 2

    # No clear ". " boundary found.
    # For multi-author with semicolons (e.g., "AUTHOR1; AUTHOR2. Title"),
    # find the last "; ALLCAPS" pattern, then scan for a non-initial/preposition ". "
    last_semi = -1
    for sm in re.finditer(r';\s*[A-ZÀ-Ú]{2,}', text):
        last_semi = sm.end()
    if last_semi > 0:
        search_pos = last_semi
        while True:
            dot_after = text.find('. ', search_pos)
            if dot_after == -1:
                break
            # Skip single-letter initials
            if dot_after >= 1 and text[dot_after - 1].isalpha() and (
                dot_after < 2 or text[dot_after - 2] in ' ,;'
            ):
                search_pos = dot_after + 2
                continue
            # Skip prepositions
            pw = re.search(r'(\w+)\s*$', text[:dot_after])
            if pw and pw.group(1).lower() in ('de', 'da', 'das', 'dos', 'do'):
                search_pos = dot_after + 2
                continue
            return dot_after + 2

    # Use last author-internal dot as fallback
    if last_author_dot >= 0:
        return last_author_dot + 2

    # Check for terminal "."
    dot = text.rfind('.')
    if dot > 0:
        return dot + 2 if dot < len(text) - 1 and text[dot + 1] == ' ' else dot + 1

    return 0


def build_page(
    sync_result: SyncResult,
    output_path: str | Path | None = None,
    settings: Settings | None = None,
    template_name: str | None = None,
    embed_audio: bool | None = None,
    source_audio_path: str | Path | None = None,
) -> Path:
    """
    Build HTML page from sync result.

    Args:
        sync_result: Processed sync result with word timestamps
        output_path: Output file path (default: output/<audio_file>.html)
        settings: Settings instance
        template_name: Template file name (default: from settings)
        embed_audio: Embed audio as base64 (default: from settings)
        source_audio_path: Path to source audio file to copy to output folder

    Returns:
        Path to generated HTML file
    """
    import shutil

    settings = settings or get_settings()
    template_name = template_name or settings.templates.page_template
    embed_audio = embed_audio if embed_audio is not None else settings.output.embed_audio

    # Determine output path
    if output_path is None:
        audio_stem = Path(sync_result.audio_file).stem
        output_path = settings.content_dir / audio_stem / "index.html"
    else:
        output_path = Path(output_path)

    output_dir = output_path.parent

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio file to output directory as audio.mp3
    if source_audio_path:
        source_audio_path = Path(source_audio_path)
        if source_audio_path.exists():
            dest_audio = output_dir / "audio.mp3"
            shutil.copy2(source_audio_path, dest_audio)

    # Copy CSS and JS files to output directory if not bundling
    if not settings.output.bundle_assets:
        css_source = settings.templates_dir / settings.templates.styles
        if css_source.exists():
            dest_css = output_dir / "styles.css"
            shutil.copy2(css_source, dest_css)

        js_source = settings.templates_dir / settings.templates.player
        if js_source.exists():
            dest_js = output_dir / "player.js"
            shutil.copy2(js_source, dest_js)

    # Copy images folder to output directory
    images_source = settings.project_root / "images"
    if images_source.exists() and images_source.is_dir():
        dest_images = output_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_source, dest_images)

    # Load template
    env = _get_template_env(settings)
    template = env.get_template(template_name)

    # Look for referencia.txt in the same directory as source audio
    referencia = None
    if source_audio_path:
        referencia_path = Path(source_audio_path).parent / "referencia.txt"
        if referencia_path.exists():
            referencia = referencia_path.read_text(encoding="utf-8").strip()

    # Apply ABNT italic formatting to reference
    if referencia:
        referencia = _format_referencia(referencia)

    # Prepare template context
    context = _build_template_context(
        sync_result=sync_result,
        settings=settings,
        embed_audio=embed_audio,
        has_local_audio=source_audio_path is not None,
        referencia=referencia,
    )

    # Render template
    html_content = template.render(**context)

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Write JSON if enabled
    if settings.output.include_json:
        json_path = output_path.with_suffix(".json")
        sync_result.save_json(json_path)

    return output_path


def build_batch(
    sync_results: list[SyncResult] | None = None,
    pages_config: list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    progress_callback: Any = None,
) -> list[Path]:
    """
    Build multiple HTML pages.

    Args:
        sync_results: Pre-processed sync results
        pages_config: Page configurations to process
        output_dir: Output directory (default: from settings)
        settings: Settings instance
        progress_callback: Optional callback(page_id, current, total)

    Returns:
        List of paths to generated HTML files
    """
    settings = settings or get_settings()
    output_dir = Path(output_dir) if output_dir else settings.content_dir

    results = sync_results or []

    # Process pages if sync_results not provided
    if not results and pages_config:
        from wordsync.process import process_sync

        for i, page in enumerate(pages_config):
            page_id = page.get("id", f"page-{i+1:03d}")
            if progress_callback:
                progress_callback(page_id, i + 1, len(pages_config))

            result = process_sync(
                audio_path=page["audio"],
                text_path=page.get("text"),
                title=page.get("title"),
                settings=settings,
            )
            results.append(result)

    # Build HTML for each result
    output_paths = []
    total = len(results)

    for i, result in enumerate(results):
        page_id = Path(result.audio_file).stem
        if progress_callback:
            progress_callback(page_id, i + 1, total)

        output_path = output_dir / page_id / "index.html"
        build_page(result, output_path, settings)
        output_paths.append(output_path)

    # Generate index page
    index_path = _build_index_page(results, output_dir, settings)
    output_paths.insert(0, index_path)

    return output_paths


def _get_template_env(settings: Settings) -> Environment:
    """Create Jinja2 environment with template directory."""
    templates_dir = settings.templates_dir

    # Check if templates directory exists
    if not templates_dir.exists():
        # Use built-in default templates
        templates_dir = Path(__file__).parent.parent / "templates"

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters
    env.filters["format_time"] = _format_time
    env.filters["to_json"] = lambda x: json.dumps(x, ensure_ascii=False)

    return env


def _build_template_context(
    sync_result: SyncResult,
    settings: Settings,
    embed_audio: bool,
    has_local_audio: bool = False,
    referencia: str | None = None,
) -> dict[str, Any]:
    """Build context dictionary for template rendering."""
    # Prepare word data for template
    words_data = []
    for word in sync_result.words:
        word_data = {
            "word": word.word,
            "start": round(word.start, 3),
            "end": round(word.end, 3),
            "confidence": round(word.confidence, 3),
        }
        if word.line_break_after:
            word_data["line_break_after"] = True
        if word.is_title:
            word_data["is_title"] = True
        words_data.append(word_data)

    # Audio source - use local audio.mp3 if available
    if has_local_audio:
        audio_src = "./audio.mp3"
    else:
        audio_src = f"audio/{sync_result.audio_file}"

    audio_data_uri = None

    if embed_audio:
        audio_path = _find_audio_file(sync_result.audio_file, settings)
        if audio_path and audio_path.exists():
            mime_type = get_audio_mime_type(audio_path)
            with open(audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            audio_data_uri = f"data:{mime_type};base64,{audio_base64}"
            audio_src = audio_data_uri

    # Load CSS and JS
    css_content = _load_asset(settings.templates_dir / settings.templates.styles, settings)
    js_content = _load_asset(settings.templates_dir / settings.templates.player, settings)

    # Classify: prose if any body line > 80 chars
    lines = (sync_result.full_text or "").split("\n")
    body_lines = lines[1:]  # skip title
    is_prose = any(len(line) > 80 for line in body_lines)

    return {
        # Page content
        "title": sync_result.title or "Leitura Guiada",
        "full_text": sync_result.full_text,
        "words": words_data,
        "word_count": len(words_data),
        "is_prose": is_prose,

        # Audio
        "audio_src": audio_src,
        "audio_file": sync_result.audio_file,
        "duration": sync_result.duration,
        "duration_formatted": _format_time(sync_result.duration),

        # Embedded assets (if bundling)
        "css_content": css_content if settings.output.bundle_assets else None,
        "js_content": js_content if settings.output.bundle_assets else None,
        "embed_assets": settings.output.bundle_assets,

        # Quality metrics
        "metrics": sync_result.metrics.to_dict() if sync_result.metrics else None,
        "low_confidence_words": sync_result.low_confidence_words,

        # Settings
        "language": sync_result.language,
        "project_name": settings.project_name,

        # Reference
        "referencia": referencia,

        # Per-livro theming
        "livro_number": _extract_livro_number(sync_result.audio_file),
        "logo_file": f"letrix-{_extract_livro_number(sync_result.audio_file)}.png",
    }


def _extract_livro_number(audio_file: str) -> int:
    """Extract livro number from audio filename (e.g. 'livro3_let5.mp3' → 3)."""
    match = re.search(r'livro(\d+)', audio_file)
    return int(match.group(1)) if match else 3


def _load_asset(path: Path, settings: Settings) -> str:
    """Load asset file content."""
    if path.exists():
        return path.read_text(encoding="utf-8")

    # Try fallback in templates directory
    fallback = settings.templates_dir / path.name
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")

    return ""


def _find_audio_file(filename: str, settings: Settings) -> Path | None:
    """Find audio file in content directory."""
    # Search in content directory
    for audio_path in settings.content_dir.rglob(filename):
        return audio_path

    # Search by stem
    stem = Path(filename).stem
    for ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
        for audio_path in settings.content_dir.rglob(f"{stem}{ext}"):
            return audio_path

    return None


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    if not seconds or seconds < 0:
        return "0:00"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def _build_index_page(
    results: list[SyncResult],
    output_dir: Path,
    settings: Settings,
) -> Path:
    """Build index page listing all pages."""
    index_path = output_dir / "index.html"

    pages_list = []
    for result in results:
        page_id = Path(result.audio_file).stem
        pages_list.append({
            "id": page_id,
            "title": result.title or page_id,
            "duration": _format_time(result.duration),
            "word_count": len(result.words),
            "confidence": round(result.metrics.average_confidence * 100) if result.metrics else 0,
            "url": f"{page_id}/index.html",
        })

    # Simple index template
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{settings.project_name} - Index</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Montserrat', -apple-system, sans-serif;
            background: #f9eacd;
            padding: 2rem;
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2a2a2a;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .pages-list {{
            list-style: none;
        }}
        .page-item {{
            background: #fdbe3f;
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}
        .page-link {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            text-decoration: none;
            color: white;
            transition: background 0.2s;
        }}
        .page-link:hover {{
            background: #d69a21;
        }}
        .page-title {{
            font-weight: 600;
            font-size: 1.1rem;
        }}
        .page-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .summary {{
            text-align: center;
            margin-top: 2rem;
            color: #838383;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{settings.project_name}</h1>
        <ul class="pages-list">
"""

    for page in pages_list:
        html += f"""            <li class="page-item">
                <a href="{page['url']}" class="page-link">
                    <span class="page-title">{page['title']}</span>
                    <span class="page-meta">
                        <span>{page['duration']}</span>
                        <span>{page['word_count']} words</span>
                        <span>{page['confidence']}% confidence</span>
                    </span>
                </a>
            </li>
"""

    html += f"""        </ul>
        <p class="summary">{len(pages_list)} pages generated</p>
    </div>
</body>
</html>
"""

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    return index_path


def create_preview_server(
    page_dir: str | Path,
    port: int = 8000,
) -> None:
    """
    Start a simple HTTP server for previewing pages.

    Args:
        page_dir: Directory containing HTML pages
        port: Server port (default: 8000)
    """
    import http.server
    import socketserver
    from functools import partial

    page_dir = Path(page_dir)

    Handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(page_dir))

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Preview server running at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
