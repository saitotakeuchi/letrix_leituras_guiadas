"""
WordSync Processing Module

Orchestrates the full sync pipeline:
1. Transcribe audio with Whisper
2. Analyze prosody with Gemini (optional)
3. Classify gaps
4. Validate with GPT-4o (optional)
5. Apply classifications and finalize timestamps

Also handles:
- Reference text alignment
- Punctuation attachment
- Quality metrics calculation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wordsync.config import Settings, get_settings
from wordsync.transcribe import (
    TranscriptionResult,
    Word,
    transcribe_audio,
)
from wordsync.alignment import align_transcription_robust
from wordsync.prosody import (
    ProsodyResult,
    analyze_prosody,
    analyze_word_durations,
    apply_duration_corrections,
)
from wordsync.classify import (
    ClassificationResult,
    GapAction,
    classify_gaps,
    apply_classifications,
)
from wordsync.validate import (
    ValidationResult,
    validate_timestamps,
    validate_timestamps_heuristic,
    HeuristicValidationResult,
)


@dataclass
class QualityMetrics:
    """Quality metrics for the sync result."""

    average_confidence: float
    gaps_preserved: int
    gaps_filled: int
    gaps_review: int
    low_confidence_words: int
    prosody_preserved_score: float
    timing_precision_ms: float
    word_accuracy: float
    # Alignment metrics
    alignment_method: str = "dp"  # "dp", "greedy"
    words_matched: int = 0
    words_interpolated: int = 0  # Missing from Whisper, timestamps interpolated
    words_verified: int = 0  # Verified by Gemini
    title_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "average_confidence": round(self.average_confidence, 3),
            "gaps_preserved": self.gaps_preserved,
            "gaps_filled": self.gaps_filled,
            "gaps_review": self.gaps_review,
            "low_confidence_words": self.low_confidence_words,
            "prosody_preserved_score": round(self.prosody_preserved_score, 3),
            "timing_precision_ms": round(self.timing_precision_ms, 1),
            "word_accuracy": round(self.word_accuracy, 3),
            "alignment_method": self.alignment_method,
            "words_matched": self.words_matched,
            "words_interpolated": self.words_interpolated,
            "words_verified": self.words_verified,
            "title_verified": self.title_verified,
        }


@dataclass
class SyncResult:
    """Final result of the sync pipeline."""

    audio_file: str
    language: str
    duration: float
    words: list[Word]
    full_text: str
    title: str | None = None

    # Pipeline results
    transcription: TranscriptionResult | None = None
    prosody: ProsodyResult | None = None
    classification: ClassificationResult | None = None
    validation: ValidationResult | None = None
    heuristic_validation: HeuristicValidationResult | None = None

    # Quality
    metrics: QualityMetrics | None = None
    low_confidence_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "audio_file": self.audio_file,
            "language": self.language,
            "duration": round(self.duration, 3),
            "title": self.title,
            "full_text": self.full_text,
            "word_count": len(self.words),
            "words": [w.to_dict() for w in self.words],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "low_confidence_words": self.low_confidence_words,
        }
        # Include heuristic validation issues if present
        if self.heuristic_validation and self.heuristic_validation.has_issues:
            result["timestamp_issues"] = self.heuristic_validation.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def process_sync(
    audio_path: str | Path,
    text_path: str | Path | None = None,
    reference_text: str | None = None,
    title: str | None = None,
    skip_title_audio: bool = False,
    use_prosody: bool = True,
    use_validation: bool = True,
    settings: Settings | None = None,
) -> SyncResult:
    """
    Run the full word-sync pipeline.

    Pipeline steps (v4 - Heuristic validation + Gemini):
    1. Transcribe audio with Whisper
    2. Align to reference text (if provided)
    3. FIX OVERLAPPING WORDS (fixes Whisper timestamp issues early, reverse iteration)
    4. GEMINI DURATION ANALYSIS (ask Gemini to listen and correct wrong word durations)
    5. Analyze prosody with Gemini (if enabled)
    6. Classify gaps
    7. Validate with Gemini/GPT-4o (if enabled) - Gemini preferred (native audio)
    8. Apply classifications and finalize
    9. ENFORCE MIN DURATION (runs ONCE at end, reverse iteration, no force borrow)
    10. HEURISTIC VALIDATION (flag suspicious timestamps, zero cost)

    Args:
        audio_path: Path to audio file
        text_path: Path to reference text file (optional)
        reference_text: Reference text string (alternative to text_path)
        title: Page title
        skip_title_audio: If True, title is not spoken in audio (use only body for alignment)
        use_prosody: Enable prosody analysis
        use_validation: Enable GPT-4o validation (default False for performance)
        settings: Settings instance

    Returns:
        SyncResult with processed word timestamps
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    # Load reference text if provided
    import re
    title_word_count = 0
    if text_path and not reference_text:
        text_path = Path(text_path)
        if text_path.exists():
            full_content = text_path.read_text(encoding="utf-8").strip()
            lines = full_content.split('\n')

            # Find title (first non-empty line) and body start
            title_line = None
            body_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip():
                    if title_line is None:
                        title_line = line.strip()
                        if not title:
                            title = title_line
                        title_word_count = len(re.findall(r'\S+', title))
                    else:
                        # First non-empty line after title is body start
                        body_start_idx = i
                        break

            # Title is always spoken in the audio — ignore skip_title_audio
            if skip_title_audio and title_line:
                skip_title_audio = False

            if skip_title_audio and body_start_idx > 0:
                # Title NOT in audio - use only body for alignment
                body_lines = [l for l in lines[body_start_idx:] if l.strip()]
                reference_text = '\n'.join(body_lines)
                if settings.debug:
                    print(f"skip_title_audio: using body only for alignment")
            else:
                # Title IS in audio - use full content
                reference_text = full_content

    # Step 1: Transcribe
    transcription = transcribe_audio(audio_path, settings=settings)

    # Step 2: Align to reference text using robust DP alignment
    if reference_text:
        # Use robust alignment (Needleman-Wunsch DP + Gemini verification)
        transcription = align_transcription_robust(
            transcription=transcription,
            reference_text=reference_text,
            audio_path=audio_path,
            settings=settings,
        )
        # Mark line breaks based on reference text structure
        transcription = _mark_line_breaks(transcription, reference_text)
        # Mark title words if spoken (skip_title_audio=False)
        if not skip_title_audio and title_word_count > 0:
            transcription = _mark_title_words(transcription, title_word_count)

    # Step 3: GEMINI DURATION ANALYSIS
    # Ask Gemini to listen and correct any wrong word durations
    # This fixes Whisper's incorrect timestamps based on actual audio
    # NOTE: This runs BEFORE overlap fix because it may introduce new overlaps
    duration_analysis = analyze_word_durations(audio_path, transcription, settings)
    if duration_analysis.corrections:
        transcription.words = apply_duration_corrections(
            transcription.words,
            duration_analysis,
            min_confidence=0.7,
        )
        if settings.debug:
            print(f"Gemini corrected {len(duration_analysis.corrections)} word durations")

    # Step 4: FIX OVERLAPPING WORDS
    # IMPORTANT: This runs AFTER Gemini corrections to fix any overlaps they may introduce
    # Whisper and Gemini both can produce overlapping word timestamps
    transcription.words = _fix_overlapping_words(transcription.words)

    # Step 5: Prosody analysis
    prosody = None
    if use_prosody and settings.prosody.enabled:
        prosody = analyze_prosody(audio_path, transcription, settings)

    # Step 6: Classify gaps
    classification = classify_gaps(transcription, prosody, settings)

    # Step 7: Validation
    validation = None
    if use_validation and settings.validation.enabled:
        validation = validate_timestamps(audio_path, transcription, settings)

    # Step 8: Apply classifications and finalize
    final_words = _finalize_timestamps(
        transcription.words,
        classification,
        validation,
        settings,
    )

    # Step 9: FINAL DURATION CHECK (only run ONCE, at the end)
    # Use the simpler ensure_minimum_duration which extends into gaps
    final_words = ensure_minimum_duration(final_words, min_duration_ms=200)

    # Step 10: Heuristic validation (flag suspicious timestamps)
    # This is a zero-cost check that catches obvious issues
    heuristic_validation = validate_timestamps_heuristic(
        final_words,
        transcription.duration,
        transcription.language,
    )
    if heuristic_validation.has_issues and settings.debug:
        print(f"[WARN] Heuristic validation found {len(heuristic_validation.issues)} timestamp issues:")
        for issue in heuristic_validation.issues[:5]:  # Show first 5
            print(f"  - {issue.message}")
        if len(heuristic_validation.issues) > 5:
            print(f"  ... and {len(heuristic_validation.issues) - 5} more")

    # Step 11: Add unspoken title words (AFTER all timing adjustments)
    # This must happen last so timing functions don't corrupt title word timestamps
    if skip_title_audio and title:
        title_words = _create_unspoken_title_words(title)
        final_words = title_words + final_words

    # Calculate quality metrics
    metrics = _calculate_metrics(
        words=final_words,
        classification=classification,
        validation=validation,
        prosody=prosody,
    )

    # Collect low confidence words
    low_confidence = []
    threshold = settings.quality.review_threshold
    for word in final_words:
        if word.confidence < threshold:
            low_confidence.append(word.word)

    return SyncResult(
        audio_file=audio_path.name,
        language=transcription.language,
        duration=transcription.duration,
        words=final_words,
        full_text=reference_text or transcription.full_text,
        title=title,
        transcription=transcription,
        prosody=prosody,
        classification=classification,
        validation=validation,
        heuristic_validation=heuristic_validation,
        metrics=metrics,
        low_confidence_words=low_confidence,
    )


def _fix_overlapping_words(words: list[Word], min_gap_ms: float = 10) -> list[Word]:
    """
    Fix overlapping words by moving start times earlier (not forward).

    Runs FIRST in the pipeline to fix words that Whisper returned
    with overlapping or too-close timestamps.

    CRITICAL CHANGE (v2): Process in REVERSE order - fix later words first

    Args:
        words: List of words with timestamps
        min_gap_ms: Minimum gap between words in milliseconds

    Returns:
        List of words with fixed timestamps
    """
    if not words:
        return words

    min_gap_sec = min_gap_ms / 1000

    # Process in REVERSE - fix later words first
    for i in range(len(words) - 1, 0, -1):
        current = words[i]
        prev_word = words[i - 1]
        gap = current.start - prev_word.end

        if gap < min_gap_sec:
            # Need to create gap - try to move CURRENT word's start forward
            # (extend into its own duration, not borrow from prev)
            needed = min_gap_sec - gap
            current_duration = current.end - current.start

            # Only if current word has enough duration to spare
            if current_duration - needed >= 0.15:
                current.start += needed
            else:
                # Otherwise, shorten prev word's end
                prev_word.end = current.start - min_gap_sec

    return words


def _finalize_timestamps(
    words: list[Word],
    classification: ClassificationResult,
    validation: ValidationResult | None,
    settings: Settings,
) -> list[Word]:
    """
    Finalize word timestamps by applying all adjustments.

    Priority:
    1. Validated timestamps (if available and confident)
    2. Gap-adjusted timestamps
    3. Original timestamps
    """
    # Start with gap-adjusted timestamps
    adjusted_words = apply_classifications(words, classification)

    # Apply validation if available
    if validation and validation.average_confidence > 0.7:
        validated_words = validation.get_validated_words()

        # Merge validated timestamps with adjusted words
        final_words = []
        for i, word in enumerate(adjusted_words):
            if i < len(validated_words):
                validated = validated_words[i]
                # Use validated timestamps if confidence is high
                if validated.confidence > settings.quality.review_threshold:
                    final_words.append(Word(
                        word=word.word,  # Keep original word text
                        start=validated.start,
                        end=validated.end,
                        confidence=validated.confidence,
                        line_break_after=word.line_break_after,
                        is_title=word.is_title,
                    ))
                else:
                    final_words.append(word)
            else:
                final_words.append(word)
        return final_words

    return adjusted_words


def ensure_minimum_duration(words: list[Word], min_duration_ms: int = 200) -> list[Word]:
    """
    Ensure all words have minimum visible duration.

    This is a simpler, more predictable approach than _enforce_minimum_durations.
    It extends short words by borrowing from gaps AFTER them.

    Strategy:
    - Process words in order
    - For each short word, extend its end into the gap after it
    - Only take up to 80% of the gap (leave some breathing room)
    - Last word gets extended unconditionally

    Args:
        words: List of words with timestamps
        min_duration_ms: Minimum duration in milliseconds (default 200ms)

    Returns:
        List of words with adjusted timestamps
    """
    if not words:
        return words

    min_duration = min_duration_ms / 1000.0

    for i, word in enumerate(words):
        duration = word.end - word.start

        if duration >= min_duration:
            continue

        needed = min_duration - duration

        # Try to extend into gap after this word
        if i < len(words) - 1:
            gap_after = words[i + 1].start - word.end
            # Don't take more than 80% of gap to leave some breathing room
            extension = min(needed, gap_after * 0.8)
            word.end += extension
        else:
            # Last word - just extend it
            word.end = word.start + min_duration

    return words


def _calculate_metrics(
    words: list[Word],
    classification: ClassificationResult,
    validation: ValidationResult | None,
    prosody: ProsodyResult | None,
) -> QualityMetrics:
    """Calculate quality metrics for the sync result."""
    # Average confidence
    avg_confidence = (
        sum(w.confidence for w in words) / len(words)
        if words else 0.0
    )

    # Gap statistics
    gaps_preserved = classification.gaps_kept if classification else 0
    gaps_filled = classification.gaps_filled if classification else 0
    gaps_review = classification.gaps_review if classification else 0

    # Low confidence words
    low_confidence_count = sum(1 for w in words if w.confidence < 0.8)

    # Prosody preservation score
    prosody_score = 0.0
    if prosody:
        total_gaps = len(prosody.gap_analyses)
        if total_gaps > 0:
            preserved = len([g for g in prosody.gap_analyses
                           if g.recommendation == "keep" and
                           any(c.action == GapAction.KEEP
                               for c in classification.classifications
                               if abs(c.gap_start - g.gap_start) < 0.1)])
            prosody_score = preserved / total_gaps

    # Timing precision
    timing_precision = 0.0
    if validation and validation.words_matched > 0:
        timing_precision = validation.average_deviation_ms
    else:
        # Estimate from word durations
        if words:
            durations = [(w.end - w.start) * 1000 for w in words]
            avg_duration = sum(durations) / len(durations)
            # Assume precision is roughly proportional to average duration
            timing_precision = min(50, avg_duration * 0.1)

    # Word accuracy (based on confidence)
    word_accuracy = avg_confidence

    return QualityMetrics(
        average_confidence=avg_confidence,
        gaps_preserved=gaps_preserved,
        gaps_filled=gaps_filled,
        gaps_review=gaps_review,
        low_confidence_words=low_confidence_count,
        prosody_preserved_score=prosody_score,
        timing_precision_ms=timing_precision,
        word_accuracy=word_accuracy,
    )


def _mark_line_breaks(
    transcription: TranscriptionResult,
    reference_text: str,
) -> TranscriptionResult:
    """
    Mark words that have line breaks after them based on reference text.

    Analyzes the reference text to find line break positions and marks
    the corresponding words in the transcription.

    Expected text file format:
        Each line is a verse/sentence that will get a line break after it.
        Empty lines are ignored but can be used for visual organization.

    Example text.txt:
        NÃO FALO, MAS SEMPRE EXPLICO.
        SOU UM AMIGO CALADO.
        ABRO AS PORTAS.

    Args:
        transcription: Transcription result with aligned words
        reference_text: Original reference text with line breaks

    Returns:
        TranscriptionResult with line_break_after marked on words
    """
    import re

    if not reference_text or not transcription.words:
        return transcription

    # Split reference text into lines and filter out empty lines
    lines = [line for line in reference_text.split('\n') if line.strip()]

    # Build a list of (word, has_line_break_after) from reference text
    ref_word_breaks = []
    for i, line in enumerate(lines):
        line_words = re.findall(r'\S+', line)
        for j, word in enumerate(line_words):
            is_last_word_in_line = (j == len(line_words) - 1)
            # Add line break after last word of each line (except the very last line)
            has_break = is_last_word_in_line and i < len(lines) - 1
            ref_word_breaks.append((word, has_break))

    # Match words and transfer line break markers
    # We need to handle case differences and punctuation
    ref_idx = 0
    for word in transcription.words:
        if ref_idx >= len(ref_word_breaks):
            break

        ref_word, has_break = ref_word_breaks[ref_idx]

        # Clean both words for comparison
        word_clean = re.sub(r'[^\w]', '', word.word.lower())
        ref_clean = re.sub(r'[^\w]', '', ref_word.lower())

        if word_clean == ref_clean:
            word.line_break_after = has_break
            ref_idx += 1
        else:
            # Try to find matching word ahead in reference
            found = False
            for look_ahead in range(1, min(5, len(ref_word_breaks) - ref_idx)):
                look_ref, look_break = ref_word_breaks[ref_idx + look_ahead]
                look_clean = re.sub(r'[^\w]', '', look_ref.lower())
                if word_clean == look_clean:
                    word.line_break_after = look_break
                    ref_idx = ref_idx + look_ahead + 1
                    found = True
                    break
            if not found:
                ref_idx += 1

    return transcription


def _mark_title_words(
    transcription: TranscriptionResult,
    title_word_count: int,
) -> TranscriptionResult:
    """
    Mark the first N words as title words.

    Title words are displayed in the header but still synced with audio.
    They should not appear in the body text.

    Args:
        transcription: Transcription result with aligned words
        title_word_count: Number of words in the title

    Returns:
        TranscriptionResult with is_title marked on first N words
    """
    if not transcription.words or title_word_count <= 0:
        return transcription

    for i, word in enumerate(transcription.words):
        if i < title_word_count:
            word.is_title = True
        else:
            break

    return transcription


def _create_unspoken_title_words(title: str) -> list[Word]:
    """
    Create title words without timestamps.

    Used when title is not spoken in audio (--skip-title flag).
    Title words get is_title=True but start=0, end=0 (no sync).

    Args:
        title: Title text

    Returns:
        List of Word objects for the title
    """
    import re

    if not title:
        return []

    title_words_text = re.findall(r'\S+', title)

    # Create title Word objects with no timing
    title_words = []
    for i, word_text in enumerate(title_words_text):
        is_last = (i == len(title_words_text) - 1)
        title_words.append(Word(
            word=word_text,
            start=0.0,  # No timing
            end=0.0,    # No timing
            confidence=1.0,
            is_title=True,
            line_break_after=is_last,  # Break after last title word
        ))

    return title_words


def discover_pages(content_dir: str | Path) -> list[dict[str, Any]]:
    """
    Discover pages from content directory structure.

    Supports two layouts:
        New (S3-ready):
            content/livro3-let5/content/audio.mp3   ← source files in content/ subfolder
            content/livro3-let5/content/text.txt

        Legacy:
            content/livro3-let5/audio.mp3            ← source files in page root
            content/livro3-let5/text.txt

    Returns:
        List of page configs with 'id', 'audio', 'text', 'title',
        'page_dir' (page root), and 'source_dir' (where source files live).
    """
    content_dir = Path(content_dir)
    pages = []

    if not content_dir.exists():
        return pages

    for page_dir in sorted(content_dir.iterdir()):
        if not page_dir.is_dir():
            continue

        # Determine source directory: prefer content/ subfolder, fall back to page root
        source_dir = page_dir / "content"
        if not source_dir.exists() or not source_dir.is_dir():
            source_dir = page_dir

        # Look for audio file in source directory
        audio_file = None
        for ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            candidates = list(source_dir.glob(f"*{ext}"))
            if candidates:
                audio_file = candidates[0]
                break

        if not audio_file:
            continue

        # Look for text file in source directory
        text_file = None
        for name in ["text.txt", "texto.txt", "transcript.txt"]:
            candidate = source_dir / name
            if candidate.exists():
                text_file = candidate
                break

        pages.append({
            "id": page_dir.name,
            "audio": str(audio_file),
            "text": str(text_file) if text_file else None,
            "title": page_dir.name.replace("-", " ").replace("_", " ").title(),
            "page_dir": str(page_dir),
            "source_dir": str(source_dir),
        })

    return pages


def load_sync_result(path: str | Path) -> SyncResult:
    """
    Load a SyncResult from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Reconstructed SyncResult
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    words = [
        Word(
            word=w["word"],
            start=w["start"],
            end=w["end"],
            confidence=w.get("confidence", 1.0),
            line_break_after=w.get("line_break_after", False),
            is_title=w.get("is_title", False),
        )
        for w in data.get("words", [])
    ]

    metrics = None
    if data.get("metrics"):
        m = data["metrics"]
        metrics = QualityMetrics(
            average_confidence=m.get("average_confidence", 0.0),
            gaps_preserved=m.get("gaps_preserved", 0),
            gaps_filled=m.get("gaps_filled", 0),
            gaps_review=m.get("gaps_review", 0),
            low_confidence_words=m.get("low_confidence_words", 0),
            prosody_preserved_score=m.get("prosody_preserved_score", 0.0),
            timing_precision_ms=m.get("timing_precision_ms", 0.0),
            word_accuracy=m.get("word_accuracy", 0.0),
        )

    return SyncResult(
        audio_file=data.get("audio_file", ""),
        language=data.get("language", "pt"),
        duration=data.get("duration", 0.0),
        words=words,
        full_text=data.get("full_text", ""),
        title=data.get("title"),
        metrics=metrics,
        low_confidence_words=data.get("low_confidence_words", []),
    )
