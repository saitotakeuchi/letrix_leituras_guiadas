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
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from wordsync.config import Settings, get_settings
from wordsync.process import SyncResult, discover_pages, load_sync_result
from wordsync.transcribe import get_audio_mime_type


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

    return {
        # Page content
        "title": sync_result.title or "Leitura Guiada",
        "full_text": sync_result.full_text,
        "words": words_data,
        "word_count": len(words_data),

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
    }


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
