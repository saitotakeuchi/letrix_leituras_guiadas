"""
WordSync CLI Module

Command-line interface for the WordSync Engine.
Built with Typer for a modern CLI experience.

Commands:
    sync     - Process audio/text pair to generate timestamps
    build    - Build HTML page from sync result
    batch    - Process all pages in content directory
    preview  - Start local server to preview pages
    validate - Check quality metrics of timestamps
    info     - Show configuration and API status
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from wordsync import __version__
from wordsync.config import get_settings, reload_settings

app = typer.Typer(
    name="wordsync",
    help="WordSync Engine - Karaoke Word-Sync for Letrix",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
) -> None:
    """WordSync Engine - Karaoke Word-Sync for Letrix."""
    if version:
        console.print(f"[bold]WordSync[/bold] version {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def sync(
    audio: Annotated[
        Path,
        typer.Argument(help="Path to audio file (mp3, wav, etc.)"),
    ],
    text: Annotated[
        Optional[Path],
        typer.Argument(help="Path to reference text file"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory"),
    ] = None,
    title: Annotated[
        Optional[str],
        typer.Option("--title", "-t", help="Page title"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Transcription provider: openai, local, whisperx"),
    ] = None,
    no_prosody: Annotated[
        bool,
        typer.Option("--no-prosody", help="Disable prosody analysis"),
    ] = False,
    no_validate: Annotated[
        bool,
        typer.Option("--no-validate", help="Disable GPT-4o validation"),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", "-l", help="Use local Whisper instead of API"),
    ] = False,
    json_only: Annotated[
        bool,
        typer.Option("--json-only", help="Output JSON only, no HTML"),
    ] = False,
    skip_title: Annotated[
        bool,
        typer.Option("--skip-title", help="Title is not spoken in audio"),
    ] = False,
) -> None:
    """
    Process audio file to generate word-level timestamps.

    This runs the full sync pipeline:
    1. Transcribe with Whisper
    2. Analyze prosody with Gemini (optional)
    3. Classify gaps
    4. Validate with GPT-4o (optional)
    5. Generate HTML and JSON output
    """
    from wordsync.process import process_sync
    from wordsync.build import build_page

    settings = get_settings()

    # Handle provider selection
    if provider:
        settings.transcription.provider = provider
    elif local:
        settings.transcription.provider = "local"

    if not audio.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio}")
        raise typer.Exit(1)

    # Check API keys
    missing = settings.validate_required_keys()
    if missing and not local:
        console.print("[yellow]Warning:[/yellow] Missing API keys:")
        for key in missing:
            console.print(f"  - {key}")
        console.print("Using available providers or fallbacks.")

    console.print(Panel(f"[bold]Processing:[/bold] {audio.name}", expand=False))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Run pipeline
        task = progress.add_task("Transcribing audio...", total=None)

        try:
            result = process_sync(
                audio_path=audio,
                text_path=text,
                title=title,
                skip_title_audio=skip_title,
                use_prosody=not no_prosody,
                use_validation=not no_validate,
                settings=settings,
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        progress.update(task, description="Building output...")

        # Determine output location
        # Output to content/page-name/output/ folder (same folder as input audio)
        if output:
            output_dir = Path(output)
        else:
            # Output to 'output' subfolder in same directory as audio file
            output_dir = audio.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "timestamps.json"
        result.save_json(json_path)

        # Build HTML (this also copies audio and CSS)
        html_path = None
        if not json_only:
            html_path = build_page(
                result,
                output_dir / "index.html",
                settings,
                source_audio_path=audio,
            )

        progress.update(task, description="Done!")

    # Print summary
    console.print()
    _print_sync_summary(result, json_path, html_path)


@app.command()
def build(
    page_id: Annotated[
        str,
        typer.Argument(help="Page ID or path to timestamps.json"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path"),
    ] = None,
    embed_audio: Annotated[
        bool,
        typer.Option("--embed-audio", help="Embed audio as base64"),
    ] = False,
) -> None:
    """
    Build HTML page from existing timestamps.

    Use this to regenerate HTML from a timestamps.json file.
    """
    from wordsync.process import load_sync_result
    from wordsync.build import build_page

    settings = get_settings()

    # Find timestamps file
    if page_id.endswith(".json"):
        json_path = Path(page_id)
    else:
        json_path = settings.output_dir / page_id / "timestamps.json"

    if not json_path.exists():
        console.print(f"[red]Error:[/red] Timestamps not found: {json_path}")
        raise typer.Exit(1)

    console.print(f"Loading: {json_path}")

    result = load_sync_result(json_path)

    output_path = output or json_path.parent / "index.html"
    html_path = build_page(result, output_path, settings, embed_audio=embed_audio)

    console.print(f"[green]Built:[/green] {html_path}")


@app.command()
def batch(
    content_dir: Annotated[
        Optional[Path],
        typer.Option("--content", "-c", help="Content directory"),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory"),
    ] = None,
    no_prosody: Annotated[
        bool,
        typer.Option("--no-prosody", help="Disable prosody analysis"),
    ] = False,
    no_validate: Annotated[
        bool,
        typer.Option("--no-validate", help="Disable GPT-4o validation"),
    ] = False,
) -> None:
    """
    Process all pages in the content directory.

    Discovers pages from the content/ folder structure and
    processes each one through the full pipeline.
    """
    from wordsync.process import process_sync, discover_pages
    from wordsync.build import build_page, _build_index_page

    settings = get_settings()

    content_dir = Path(content_dir) if content_dir else settings.content_dir
    output_dir = Path(output_dir) if output_dir else settings.output_dir

    if not content_dir.exists():
        console.print(f"[red]Error:[/red] Content directory not found: {content_dir}")
        raise typer.Exit(1)

    # Discover pages
    pages = discover_pages(content_dir)

    if not pages:
        console.print(f"[yellow]No pages found in:[/yellow] {content_dir}")
        console.print("Expected structure: content/page-001/audio.mp3")
        raise typer.Exit(1)

    console.print(f"[bold]Found {len(pages)} pages[/bold]")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(pages))

        for page in pages:
            page_id = page["id"]
            progress.update(task, description=f"Processing {page_id}...")

            try:
                result = process_sync(
                    audio_path=page["audio"],
                    text_path=page.get("text"),
                    title=page.get("title"),
                    use_prosody=not no_prosody,
                    use_validation=not no_validate,
                    settings=settings,
                )
                results.append(result)

                # Build HTML
                page_output = output_dir / page_id
                build_page(result, page_output / "index.html", settings)
                result.save_json(page_output / "timestamps.json")

                progress.advance(task)

            except Exception as e:
                console.print(f"[red]Error processing {page_id}:[/red] {e}")
                continue

    # Build index
    _build_index_page(results, output_dir, settings)

    # Summary
    console.print()
    console.print(f"[green]Processed {len(results)}/{len(pages)} pages[/green]")
    console.print(f"Output: {output_dir}")


@app.command()
def preview(
    page_id: Annotated[
        Optional[str],
        typer.Argument(help="Page ID to preview (or all)"),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port"),
    ] = 8000,
) -> None:
    """
    Start local server to preview generated pages.

    Opens a web browser to view the karaoke player.
    """
    from wordsync.build import create_preview_server
    import webbrowser

    settings = get_settings()

    # Determine directory to serve
    if page_id:
        serve_dir = settings.output_dir / page_id
        if not serve_dir.exists():
            console.print(f"[red]Error:[/red] Page not found: {serve_dir}")
            raise typer.Exit(1)
        url = f"http://localhost:{port}/index.html"
    else:
        serve_dir = settings.output_dir
        url = f"http://localhost:{port}"

    if not serve_dir.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {serve_dir}")
        console.print("Run 'wordsync sync' or 'wordsync batch' first.")
        raise typer.Exit(1)

    console.print(f"[bold]Serving:[/bold] {serve_dir}")
    console.print(f"[bold]URL:[/bold] {url}")
    console.print()

    # Open browser
    webbrowser.open(url)

    # Start server
    create_preview_server(serve_dir, port)


@app.command()
def validate(
    timestamps: Annotated[
        Path,
        typer.Argument(help="Path to timestamps.json file"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed analysis"),
    ] = False,
) -> None:
    """
    Check quality metrics of generated timestamps.

    Analyzes timestamps and reports potential issues.
    """
    from wordsync.process import load_sync_result

    settings = get_settings()

    if not timestamps.exists():
        console.print(f"[red]Error:[/red] File not found: {timestamps}")
        raise typer.Exit(1)

    result = load_sync_result(timestamps)

    # Quality table
    table = Table(title="Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    if result.metrics:
        m = result.metrics

        # Confidence
        conf_status = "[green]OK" if m.average_confidence >= 0.9 else (
            "[yellow]WARN" if m.average_confidence >= 0.8 else "[red]LOW"
        )
        table.add_row("Average Confidence", f"{m.average_confidence:.1%}", conf_status)

        # Timing precision
        prec_status = "[green]OK" if m.timing_precision_ms <= 50 else (
            "[yellow]WARN" if m.timing_precision_ms <= 100 else "[red]HIGH"
        )
        table.add_row("Timing Precision", f"{m.timing_precision_ms:.0f}ms", prec_status)

        # Gap statistics
        table.add_row("Gaps Preserved", str(m.gaps_preserved), "")
        table.add_row("Gaps Filled", str(m.gaps_filled), "")

        review_status = "[green]OK" if m.gaps_review == 0 else "[yellow]REVIEW"
        table.add_row("Gaps for Review", str(m.gaps_review), review_status)

        # Word accuracy
        acc_status = "[green]OK" if m.word_accuracy >= 0.95 else (
            "[yellow]WARN" if m.word_accuracy >= 0.9 else "[red]LOW"
        )
        table.add_row("Word Accuracy", f"{m.word_accuracy:.1%}", acc_status)

    console.print(table)

    # Low confidence words
    if result.low_confidence_words:
        console.print()
        console.print(f"[yellow]Low confidence words ({len(result.low_confidence_words)}):[/yellow]")
        for word in result.low_confidence_words[:10]:
            console.print(f"  - {word}")
        if len(result.low_confidence_words) > 10:
            console.print(f"  ... and {len(result.low_confidence_words) - 10} more")

    # Verbose output
    if verbose:
        console.print()
        console.print("[bold]Word Details:[/bold]")

        words_table = Table()
        words_table.add_column("#", justify="right")
        words_table.add_column("Word")
        words_table.add_column("Start", justify="right")
        words_table.add_column("End", justify="right")
        words_table.add_column("Conf", justify="right")

        for i, word in enumerate(result.words[:20]):
            conf_color = "green" if word.confidence >= 0.9 else (
                "yellow" if word.confidence >= 0.8 else "red"
            )
            words_table.add_row(
                str(i + 1),
                word.word,
                f"{word.start:.3f}",
                f"{word.end:.3f}",
                f"[{conf_color}]{word.confidence:.2f}[/{conf_color}]",
            )

        if len(result.words) > 20:
            words_table.add_row("...", "...", "...", "...", "...")

        console.print(words_table)


@app.command()
def info() -> None:
    """
    Show configuration and API status.

    Displays current settings and checks API connectivity.
    """
    settings = get_settings()

    console.print(Panel("[bold]WordSync Configuration[/bold]", expand=False))

    # API Keys
    console.print("\n[bold]API Status:[/bold]")
    api_table = Table(show_header=False)
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Status")

    api_table.add_row(
        "OpenAI",
        "[green]Configured" if settings.has_openai else "[red]Not configured"
    )
    api_table.add_row(
        "Google (Gemini)",
        "[green]Configured" if settings.has_google else "[yellow]Not configured"
    )
    api_table.add_row(
        "Anthropic",
        "[green]Configured" if settings.has_anthropic else "[dim]Not configured"
    )

    console.print(api_table)

    # Paths
    console.print("\n[bold]Paths:[/bold]")
    path_table = Table(show_header=False)
    path_table.add_column("Path", style="cyan")
    path_table.add_column("Location")

    path_table.add_row("Project Root", str(settings.project_root))
    path_table.add_row("Content Dir", str(settings.content_dir))
    path_table.add_row("Output Dir", str(settings.output_dir))
    path_table.add_row("Templates Dir", str(settings.templates_dir))

    console.print(path_table)

    # Settings
    console.print("\n[bold]Settings:[/bold]")
    console.print(f"  Language: {settings.language}")
    console.print(f"  Transcription Provider: {settings.transcription.provider}")
    console.print(f"  Whisper Model: {settings.transcription.whisper_model}")
    console.print(f"  WhisperX Model: {settings.transcription.whisperx_model}")
    console.print(f"  Prosody Enabled: {settings.prosody.enabled}")
    console.print(f"  Validation Enabled: {settings.validation.enabled}")

    # Check for missing requirements
    missing = settings.validate_required_keys()
    if missing:
        console.print("\n[yellow]Missing required keys:[/yellow]")
        for key in missing:
            console.print(f"  - {key}")


def _print_sync_summary(result, json_path: Path, html_path: Optional[Path]) -> None:
    """Print sync result summary."""
    table = Table(title="Sync Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Words", str(len(result.words)))
    table.add_row("Duration", f"{result.duration:.1f}s")

    if result.metrics:
        table.add_row("Confidence", f"{result.metrics.average_confidence:.1%}")
        table.add_row("Gaps Preserved", str(result.metrics.gaps_preserved))
        table.add_row("Gaps Filled", str(result.metrics.gaps_filled))

    console.print(table)

    console.print()
    console.print(f"[green]JSON:[/green] {json_path}")
    if html_path:
        console.print(f"[green]HTML:[/green] {html_path}")

    if result.low_confidence_words:
        console.print()
        console.print(f"[yellow]Low confidence words: {len(result.low_confidence_words)}[/yellow]")


if __name__ == "__main__":
    app()
