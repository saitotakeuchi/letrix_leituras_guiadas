"""
WordSync Validation Module

Cross-validates timestamps using audio transcription capabilities.
Supports multiple validation approaches:
1. Heuristic validation (zero cost, catches obvious issues)
2. Gemini-based validation (native audio understanding)
3. GPT-4o validation (legacy, for backwards compatibility)

Validation process:
1. Run heuristic checks to flag suspicious timestamps
2. Optionally use Gemini to verify/correct flagged issues
3. Calculate confidence scores for each word
"""

import base64
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from wordsync.config import Settings, get_settings
from wordsync.transcribe import TranscriptionResult, Word


class TimestampIssueType(str, Enum):
    """Types of timestamp issues detected by heuristics."""

    TOO_SHORT = "too_short"  # Word duration unrealistically short
    TOO_LONG = "too_long"  # Word duration unrealistically long
    LARGE_GAP = "large_gap"  # Suspiciously large gap before word
    SYLLABLE_MISMATCH = "syllable_mismatch"  # Duration doesn't match syllable count
    ALIGNMENT_DRIFT = "alignment_drift"  # Last word ends far from audio end
    OVERLAP = "overlap"  # Words overlap in time
    ROUND_TIMESTAMPS = "round_timestamps"  # Suspiciously round timestamps


@dataclass
class TimestampIssue:
    """A detected timestamp issue."""

    word_index: int
    issue_type: TimestampIssueType
    message: str
    severity: str = "warning"  # "warning" or "error"
    suggested_correction: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word_index": self.word_index,
            "type": self.issue_type.value,
            "message": self.message,
            "severity": self.severity,
            "suggested_correction": self.suggested_correction,
        }


@dataclass
class HeuristicValidationResult:
    """Result of heuristic timestamp validation."""

    issues: list[TimestampIssue]
    words_checked: int
    issues_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "words_checked": self.words_checked,
            "total_issues": len(self.issues),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues_by_type": self.issues_by_type,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class WordValidation:
    """Validation result for a single word."""

    word: str
    primary_start: float
    primary_end: float
    secondary_start: float | None = None
    secondary_end: float | None = None
    final_start: float = 0.0
    final_end: float = 0.0
    deviation_start_ms: float = 0.0
    deviation_end_ms: float = 0.0
    confidence: float = 1.0
    matched: bool = True  # Whether word was found in secondary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word": self.word,
            "primary": {
                "start": round(self.primary_start, 3),
                "end": round(self.primary_end, 3),
            },
            "secondary": {
                "start": round(self.secondary_start, 3) if self.secondary_start else None,
                "end": round(self.secondary_end, 3) if self.secondary_end else None,
            } if self.matched else None,
            "final": {
                "start": round(self.final_start, 3),
                "end": round(self.final_end, 3),
            },
            "deviation_ms": {
                "start": round(self.deviation_start_ms, 1),
                "end": round(self.deviation_end_ms, 1),
            },
            "confidence": round(self.confidence, 3),
            "matched": self.matched,
        }


@dataclass
class ValidationResult:
    """Result of timestamp validation."""

    audio_file: str
    primary_provider: str
    secondary_provider: str
    word_validations: list[WordValidation]
    average_confidence: float
    average_deviation_ms: float
    words_matched: int
    words_unmatched: int
    low_confidence_words: list[str] = field(default_factory=list)
    raw_secondary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audio_file": self.audio_file,
            "providers": {
                "primary": self.primary_provider,
                "secondary": self.secondary_provider,
            },
            "summary": {
                "average_confidence": round(self.average_confidence, 3),
                "average_deviation_ms": round(self.average_deviation_ms, 1),
                "words_matched": self.words_matched,
                "words_unmatched": self.words_unmatched,
                "low_confidence_count": len(self.low_confidence_words),
            },
            "low_confidence_words": self.low_confidence_words,
            "validations": [v.to_dict() for v in self.word_validations],
        }

    def get_validated_words(self) -> list[Word]:
        """Get list of Words with validated timestamps."""
        return [
            Word(
                word=v.word,
                start=v.final_start,
                end=v.final_end,
                confidence=v.confidence,
            )
            for v in self.word_validations
        ]


def validate_timestamps(
    audio_path: str | Path,
    transcription: TranscriptionResult,
    settings: Settings | None = None,
    use_gemini: bool = True,
) -> ValidationResult:
    """
    Validate timestamps using cross-validation.

    By default, uses Gemini for validation (native audio understanding).
    Can fall back to GPT-4o if Gemini is unavailable or if use_gemini=False.

    Args:
        audio_path: Path to audio file
        transcription: Primary transcription result
        settings: Settings instance
        use_gemini: If True, prefer Gemini over GPT-4o for validation

    Returns:
        ValidationResult with cross-validated timestamps

    Note:
        Falls back to heuristic validation if no audio model is available.
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    if not settings.validation.enabled:
        return _validate_with_heuristics(transcription, settings)

    # Try Gemini first (preferred - native audio understanding)
    if use_gemini and settings.has_google:
        try:
            return validate_timestamps_with_gemini(
                audio_path,
                transcription.words,
                settings,
            )
        except Exception as e:
            if settings.debug:
                print(f"Gemini validation failed: {e}, trying GPT-4o")

    # Fall back to GPT-4o
    if settings.has_openai:
        try:
            return _validate_with_gpt4o(audio_path, transcription, settings)
        except Exception as e:
            if settings.debug:
                print(f"GPT-4o validation failed: {e}, falling back to heuristics")

    # Final fallback to heuristics
    return _validate_with_heuristics(transcription, settings)


def _validate_with_gpt4o(
    audio_path: Path,
    transcription: TranscriptionResult,
    settings: Settings,
) -> ValidationResult:
    """
    Validate using GPT-4o audio capabilities.

    GPT-4o provides independent transcription with timestamps
    that can be cross-referenced with the primary Whisper transcription.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    # Encode audio as base64
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine audio format
    suffix = audio_path.suffix.lower()
    format_map = {
        ".mp3": "mp3",
        ".wav": "wav",
        ".ogg": "ogg",
        ".flac": "flac",
        ".m4a": "m4a",
    }
    audio_format = format_map.get(suffix, "mp3")

    # Build the prompt
    words_list = ", ".join([f'"{w.word}"' for w in transcription.words[:50]])
    prompt = f"""Transcribe this Portuguese audio and provide word-level timestamps.

The expected words are approximately: {words_list}...

Return JSON with this exact format:
{{
    "words": [
        {{"word": "WORD", "start": 0.0, "end": 0.5}},
        ...
    ]
}}

Important:
- Provide timestamps in seconds with 3 decimal precision
- Include all spoken words
- Match the expected text as closely as possible
- Preserve original capitalization and punctuation
"""

    try:
        response = client.chat.completions.create(
            model=settings.validation.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": audio_format,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # Parse response
        response_text = response.choices[0].message.content
        secondary_data = json.loads(response_text)
        secondary_words = secondary_data.get("words", [])

    except Exception as e:
        # If GPT-4o audio fails, try without audio
        if settings.debug:
            print(f"GPT-4o audio mode failed: {e}, using text-only validation")

        # Fall back to just comparing with heuristics
        return _validate_with_heuristics(transcription, settings)

    # Cross-validate timestamps
    return _cross_validate(
        primary_words=transcription.words,
        secondary_words=secondary_words,
        audio_file=audio_path.name,
        primary_provider=transcription.provider,
        secondary_provider="gpt4o",
        settings=settings,
        raw_secondary=secondary_data,
    )


def _validate_with_heuristics(
    transcription: TranscriptionResult,
    settings: Settings,
) -> ValidationResult:
    """
    Validate using heuristic rules when GPT-4o is unavailable.

    Uses timing consistency and expected word durations to
    calculate confidence scores.
    """
    validations = []
    low_confidence_words = []

    quality = settings.quality

    for i, word in enumerate(transcription.words):
        duration = word.end - word.start
        duration_ms = duration * 1000

        # Calculate heuristic confidence
        confidence = 1.0

        # Penalize very short words (< 50ms)
        if duration_ms < 50:
            confidence *= 0.7

        # Penalize very long words (> 2000ms)
        if duration_ms > 2000:
            confidence *= 0.8

        # Penalize words with suspiciously round timestamps
        if _is_round_timestamp(word.start) and _is_round_timestamp(word.end):
            confidence *= 0.9

        # Check timing consistency with neighbors
        if i > 0:
            prev_word = transcription.words[i - 1]
            gap = word.start - prev_word.end

            # Penalize overlapping words
            if gap < 0:
                confidence *= 0.5

            # Penalize large gaps that aren't at punctuation
            if gap > 1.0 and not _has_punctuation(prev_word.word):
                confidence *= 0.8

        # Use original word confidence if available
        if word.confidence < 1.0:
            confidence *= word.confidence

        # Clamp confidence
        confidence = max(0.1, min(1.0, confidence))

        # Track low confidence words
        if confidence < quality.review_threshold:
            low_confidence_words.append(word.word)

        validations.append(WordValidation(
            word=word.word,
            primary_start=word.start,
            primary_end=word.end,
            secondary_start=None,
            secondary_end=None,
            final_start=word.start,
            final_end=word.end,
            deviation_start_ms=0,
            deviation_end_ms=0,
            confidence=confidence,
            matched=False,
        ))

    avg_confidence = (
        sum(v.confidence for v in validations) / len(validations)
        if validations else 0.0
    )

    return ValidationResult(
        audio_file=transcription.audio_file,
        primary_provider=transcription.provider,
        secondary_provider="heuristics",
        word_validations=validations,
        average_confidence=avg_confidence,
        average_deviation_ms=0,
        words_matched=0,
        words_unmatched=len(validations),
        low_confidence_words=low_confidence_words,
    )


def _cross_validate(
    primary_words: list[Word],
    secondary_words: list[dict[str, Any]],
    audio_file: str,
    primary_provider: str,
    secondary_provider: str,
    settings: Settings,
    raw_secondary: dict[str, Any],
) -> ValidationResult:
    """
    Cross-validate timestamps between two transcriptions.

    Uses word alignment to match words between transcriptions
    and calculates median timestamps when available.
    """
    validations = []
    low_confidence_words = []
    total_deviation_start = 0.0
    total_deviation_end = 0.0
    words_matched = 0
    words_unmatched = 0

    quality = settings.quality
    use_median = settings.validation.use_median_timestamps

    # Create lookup for secondary words
    secondary_lookup = _build_word_lookup(secondary_words)

    for i, primary_word in enumerate(primary_words):
        # Try to find matching word in secondary
        secondary_match = _find_matching_word(
            primary_word,
            i,
            secondary_lookup,
            secondary_words,
        )

        if secondary_match:
            words_matched += 1

            secondary_start = secondary_match.get("start", primary_word.start)
            secondary_end = secondary_match.get("end", primary_word.end)

            # Calculate deviation
            deviation_start = abs(primary_word.start - secondary_start) * 1000
            deviation_end = abs(primary_word.end - secondary_end) * 1000
            total_deviation_start += deviation_start
            total_deviation_end += deviation_end

            # Determine final timestamps
            if use_median:
                final_start = (primary_word.start + secondary_start) / 2
                final_end = (primary_word.end + secondary_end) / 2
            else:
                # Use primary if deviation is small, otherwise use secondary
                max_dev = quality.max_timing_deviation
                final_start = primary_word.start if deviation_start < max_dev else secondary_start
                final_end = primary_word.end if deviation_end < max_dev else secondary_end

            # BOUNDS CHECK: Ensure minimum duration (50ms)
            # If validation produces ultra-short duration, fall back to primary
            final_duration = final_end - final_start
            bounds_check_failed = final_duration < 0.05  # Less than 50ms

            if bounds_check_failed:
                final_start = primary_word.start
                final_end = primary_word.end

            # Calculate confidence based on agreement
            max_deviation = max(deviation_start, deviation_end)
            if bounds_check_failed:
                # Reduce confidence when bounds check fails
                confidence = min(0.5, 0.7 - (max_deviation / 1000))
            elif max_deviation < 50:
                confidence = 0.98
            elif max_deviation < 100:
                confidence = 0.92
            elif max_deviation < 200:
                confidence = 0.85
            elif max_deviation < 500:
                confidence = 0.7
            else:
                confidence = 0.5

            validations.append(WordValidation(
                word=primary_word.word,
                primary_start=primary_word.start,
                primary_end=primary_word.end,
                secondary_start=secondary_start,
                secondary_end=secondary_end,
                final_start=final_start,
                final_end=final_end,
                deviation_start_ms=deviation_start,
                deviation_end_ms=deviation_end,
                confidence=confidence,
                matched=True,
            ))

        else:
            words_unmatched += 1

            # No match - use primary with lower confidence
            confidence = 0.6 * primary_word.confidence

            validations.append(WordValidation(
                word=primary_word.word,
                primary_start=primary_word.start,
                primary_end=primary_word.end,
                secondary_start=None,
                secondary_end=None,
                final_start=primary_word.start,
                final_end=primary_word.end,
                deviation_start_ms=0,
                deviation_end_ms=0,
                confidence=confidence,
                matched=False,
            ))

        # Track low confidence
        if validations[-1].confidence < quality.review_threshold:
            low_confidence_words.append(primary_word.word)

    # Calculate averages
    avg_confidence = (
        sum(v.confidence for v in validations) / len(validations)
        if validations else 0.0
    )

    avg_deviation = (
        (total_deviation_start + total_deviation_end) / (2 * words_matched)
        if words_matched > 0 else 0.0
    )

    return ValidationResult(
        audio_file=audio_file,
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        word_validations=validations,
        average_confidence=avg_confidence,
        average_deviation_ms=avg_deviation,
        words_matched=words_matched,
        words_unmatched=words_unmatched,
        low_confidence_words=low_confidence_words,
        raw_secondary=raw_secondary,
    )


def _build_word_lookup(words: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Build lookup table mapping normalized words to indices."""
    import re
    lookup: dict[str, list[int]] = {}

    for i, word_data in enumerate(words):
        word = word_data.get("word", "")
        normalized = re.sub(r'[^\w]', '', word.lower())
        if normalized:
            if normalized not in lookup:
                lookup[normalized] = []
            lookup[normalized].append(i)

    return lookup


def _find_matching_word(
    primary_word: Word,
    primary_index: int,
    secondary_lookup: dict[str, list[int]],
    secondary_words: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find matching word in secondary transcription."""
    import re

    normalized = re.sub(r'[^\w]', '', primary_word.word.lower())
    if not normalized:
        return None

    if normalized not in secondary_lookup:
        return None

    candidates = secondary_lookup[normalized]

    # Find best match by position proximity
    best_match = None
    best_distance = float('inf')

    for secondary_idx in candidates:
        distance = abs(secondary_idx - primary_index)
        if distance < best_distance:
            best_distance = distance
            best_match = secondary_words[secondary_idx]

    # Only match if reasonably close in position
    if best_distance <= len(secondary_words) * 0.2:  # Within 20% of transcript
        return best_match

    return None


def _is_round_timestamp(timestamp: float) -> bool:
    """Check if timestamp is suspiciously round (e.g., 1.0, 2.5)."""
    # Check if close to 0.25 second intervals
    remainder = timestamp % 0.25
    return remainder < 0.01 or remainder > 0.24


def _has_punctuation(word: str) -> bool:
    """Check if word ends with significant punctuation."""
    word = word.strip()
    return word and word[-1] in ".!?;:"


# =============================================================================
# HEURISTIC VALIDATION (Zero cost, catches obvious issues)
# =============================================================================


def validate_timestamps_heuristic(
    words: list[Word],
    audio_duration: float,
    language: str = "pt",
) -> HeuristicValidationResult:
    """
    Detect suspicious timestamps using heuristics (no API call).

    This is a fast, zero-cost validation pass that catches obvious timestamp
    issues like:
    - Words with unrealistically short/long durations
    - Large gaps between words in continuous speech
    - Duration/syllable count mismatches
    - Alignment drift at end of audio

    Args:
        words: List of Word objects with timestamps
        audio_duration: Total audio duration in seconds
        language: Language code (used for syllable estimation)

    Returns:
        HeuristicValidationResult with detected issues
    """
    issues: list[TimestampIssue] = []
    issues_by_type: dict[str, int] = {}

    def add_issue(
        word_index: int,
        issue_type: TimestampIssueType,
        message: str,
        severity: str = "warning",
        suggested_correction: dict[str, float] | None = None,
    ) -> None:
        issues.append(TimestampIssue(
            word_index=word_index,
            issue_type=issue_type,
            message=message,
            severity=severity,
            suggested_correction=suggested_correction,
        ))
        type_key = issue_type.value
        issues_by_type[type_key] = issues_by_type.get(type_key, 0) + 1

    for i, word in enumerate(words):
        # Skip title words with no timing
        if word.start == 0 and word.end == 0 and word.is_title:
            continue

        duration = word.end - word.start
        duration_ms = duration * 1000

        # Check 1: Duration too short (<80ms)
        # Most words need at least 80ms to be perceivable
        if duration_ms < 80:
            syllables = _estimate_syllables(word.word, language)
            add_issue(
                word_index=i,
                issue_type=TimestampIssueType.TOO_SHORT,
                message=f"Word '{word.word}' has very short duration ({duration_ms:.0f}ms)",
                severity="error" if duration_ms < 50 else "warning",
            )

        # Check 2: Duration too long (>1500ms for single word)
        # Most single words are spoken in under 1.5 seconds
        if duration_ms > 1500:
            add_issue(
                word_index=i,
                issue_type=TimestampIssueType.TOO_LONG,
                message=f"Word '{word.word}' has unusually long duration ({duration_ms:.0f}ms)",
                severity="warning",
            )

        # Check 3: Large gaps between words (>500ms without punctuation)
        # In continuous reading, gaps over 500ms are usually pauses, not
        # normal speech. If there's no punctuation, it might be misalignment.
        if i > 0:
            prev_word = words[i - 1]
            # Skip if previous word has no timing (title word)
            if not (prev_word.start == 0 and prev_word.end == 0 and prev_word.is_title):
                gap = word.start - prev_word.end
                gap_ms = gap * 1000

                if gap_ms > 500 and not _has_punctuation(prev_word.word):
                    add_issue(
                        word_index=i,
                        issue_type=TimestampIssueType.LARGE_GAP,
                        message=f"Large gap ({gap_ms:.0f}ms) before '{word.word}' without punctuation",
                        severity="warning",
                    )
                elif gap_ms > 800:
                    # Even with punctuation, 800ms+ is suspicious
                    add_issue(
                        word_index=i,
                        issue_type=TimestampIssueType.LARGE_GAP,
                        message=f"Very large gap ({gap_ms:.0f}ms) before '{word.word}'",
                        severity="warning",
                    )

        # Check 4: Syllable/duration mismatch
        # Multi-syllable words with very short durations
        syllables = _estimate_syllables(word.word, language)
        if syllables >= 3 and duration_ms < 150:
            add_issue(
                word_index=i,
                issue_type=TimestampIssueType.SYLLABLE_MISMATCH,
                message=f"Word '{word.word}' ({syllables} syllables) has only {duration_ms:.0f}ms duration",
                severity="error",
            )
        elif syllables >= 2 and duration_ms < 100:
            add_issue(
                word_index=i,
                issue_type=TimestampIssueType.SYLLABLE_MISMATCH,
                message=f"Word '{word.word}' ({syllables} syllables) has only {duration_ms:.0f}ms duration",
                severity="warning",
            )

        # Check 5: Overlapping words
        if i > 0:
            prev_word = words[i - 1]
            if not (prev_word.start == 0 and prev_word.end == 0 and prev_word.is_title):
                if word.start < prev_word.end:
                    overlap_ms = (prev_word.end - word.start) * 1000
                    add_issue(
                        word_index=i,
                        issue_type=TimestampIssueType.OVERLAP,
                        message=f"Word '{word.word}' overlaps with '{prev_word.word}' by {overlap_ms:.0f}ms",
                        severity="error",
                    )

        # Check 6: Suspiciously round timestamps
        # If both start and end are on 0.5s boundaries, might be placeholder
        if _is_round_timestamp(word.start) and _is_round_timestamp(word.end):
            if _is_very_round(word.start) and _is_very_round(word.end):
                add_issue(
                    word_index=i,
                    issue_type=TimestampIssueType.ROUND_TIMESTAMPS,
                    message=f"Word '{word.word}' has suspiciously round timestamps ({word.start:.1f}-{word.end:.1f})",
                    severity="warning",
                )

    # Check 7: Alignment drift at end
    # If last word ends significantly before audio ends (>10%), might be drift
    if words:
        # Find last word with actual timing
        last_timed_word = None
        for w in reversed(words):
            if not (w.start == 0 and w.end == 0 and w.is_title):
                last_timed_word = w
                break

        if last_timed_word and audio_duration > 0:
            end_gap = audio_duration - last_timed_word.end
            if end_gap > audio_duration * 0.15 and end_gap > 1.0:
                add_issue(
                    word_index=len(words) - 1,
                    issue_type=TimestampIssueType.ALIGNMENT_DRIFT,
                    message=f"Last word ends at {last_timed_word.end:.2f}s but audio is {audio_duration:.2f}s ({end_gap:.2f}s gap)",
                    severity="warning",
                )

    return HeuristicValidationResult(
        issues=issues,
        words_checked=len(words),
        issues_by_type=issues_by_type,
    )


def _estimate_syllables(word: str, language: str = "pt") -> int:
    """
    Estimate the number of syllables in a word.

    Uses vowel counting as a simple heuristic. Not perfect but good enough
    for detecting obvious mismatches.
    """
    import re

    # Clean word (remove punctuation)
    word = re.sub(r'[^\w]', '', word.lower())
    if not word:
        return 1

    # Portuguese vowels (including accented)
    vowels = "aeiouáéíóúâêîôûàèìòùãõ"

    # Count vowel groups (consecutive vowels = 1 syllable)
    syllable_count = 0
    in_vowel = False

    for char in word:
        if char in vowels:
            if not in_vowel:
                syllable_count += 1
                in_vowel = True
        else:
            in_vowel = False

    # Minimum 1 syllable
    return max(1, syllable_count)


def _is_very_round(timestamp: float) -> bool:
    """Check if timestamp is on a 0.5 second boundary."""
    remainder = timestamp % 0.5
    return remainder < 0.02 or remainder > 0.48


# =============================================================================
# GEMINI VALIDATION (Replaces GPT-4o, uses native audio understanding)
# =============================================================================


def validate_timestamps_with_gemini(
    audio_path: Path,
    words: list[Word],
    settings: Settings,
    issues: list[TimestampIssue] | None = None,
) -> ValidationResult:
    """
    Use Gemini to validate and correct suspicious timestamps.

    Gemini has native audio understanding, making it better suited for
    timestamp validation than models with bolted-on audio support.

    Args:
        audio_path: Path to audio file
        words: List of Word objects with timestamps
        settings: Settings instance
        issues: Optional list of heuristic-detected issues to focus on

    Returns:
        ValidationResult with validated/corrected timestamps
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)

    # Prepare audio data
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Get MIME type
    suffix = audio_path.suffix.lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }
    mime_type = mime_types.get(suffix, "audio/mpeg")

    # Build word list with current timestamps
    words_info = [
        {
            "index": i,
            "word": w.word,
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "duration_ms": round((w.end - w.start) * 1000, 1),
        }
        for i, w in enumerate(words)
        if not (w.start == 0 and w.end == 0 and w.is_title)
    ]

    # If we have specific issues to check, highlight them
    issue_context = ""
    if issues:
        issue_lines = []
        for issue in issues[:10]:  # Limit to first 10 issues
            issue_lines.append(f"- Word {issue.word_index}: {issue.message}")
        issue_context = f"""

KNOWN ISSUES TO VERIFY:
The following timestamps have been flagged as potentially incorrect:
{chr(10).join(issue_lines)}

Please pay special attention to these words and verify their timestamps.
"""

    prompt = f"""Listen to this Portuguese audio and verify the word timestamps.

I have timestamps from Whisper that may contain errors. Please listen carefully
and identify words whose START TIME or END TIME doesn't match what you hear.

Current timestamps:
{json.dumps(words_info, indent=2, ensure_ascii=False)}
{issue_context}

For each word that needs correction, provide the accurate timestamps.
Focus especially on:
1. Words after pauses (start times often wrong)
2. Very short words like articles (durations often compressed)
3. Last few words (alignment drift is common)

Return JSON with this structure:
{{
    "words": [
        {{
            "word": "WORD",
            "start": 0.0,
            "end": 0.5,
            "confidence": 0.95,
            "corrected": true,
            "original_start": 0.1,
            "original_end": 0.4
        }},
        ...
    ],
    "overall_confidence": 0.85
}}

Include ALL words in the response, marking "corrected": true only for words
where you changed the timestamps.
"""

    # Create audio part
    audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)

    try:
        response = client.models.generate_content(
            model=settings.prosody.model,  # Use same model as prosody analysis
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        # Parse response
        response_text = response.text
        data = json.loads(response_text)
        gemini_words = data.get("words", [])

    except Exception as e:
        if settings.debug:
            print(f"Gemini validation failed: {e}")
        # Fall back to returning original words with heuristic confidence
        return _create_passthrough_result(words, audio_path.name, "gemini_error")

    # Cross-validate and build result
    return _cross_validate(
        primary_words=words,
        secondary_words=gemini_words,
        audio_file=audio_path.name,
        primary_provider="whisper",
        secondary_provider="gemini",
        settings=settings,
        raw_secondary=data,
    )


def _create_passthrough_result(
    words: list[Word],
    audio_file: str,
    provider: str,
) -> ValidationResult:
    """Create a ValidationResult that passes through original timestamps."""
    validations = []

    for word in words:
        # Skip title words with no timing
        if word.start == 0 and word.end == 0 and word.is_title:
            validations.append(WordValidation(
                word=word.word,
                primary_start=0.0,
                primary_end=0.0,
                final_start=0.0,
                final_end=0.0,
                confidence=1.0,
                matched=False,
            ))
            continue

        validations.append(WordValidation(
            word=word.word,
            primary_start=word.start,
            primary_end=word.end,
            final_start=word.start,
            final_end=word.end,
            confidence=word.confidence,
            matched=False,
        ))

    avg_confidence = (
        sum(v.confidence for v in validations) / len(validations)
        if validations else 0.0
    )

    return ValidationResult(
        audio_file=audio_file,
        primary_provider="whisper",
        secondary_provider=provider,
        word_validations=validations,
        average_confidence=avg_confidence,
        average_deviation_ms=0,
        words_matched=0,
        words_unmatched=len(validations),
    )
