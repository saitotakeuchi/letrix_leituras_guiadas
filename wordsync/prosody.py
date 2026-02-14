"""
WordSync Prosody Analysis Module

Uses Gemini 2.5 Flash native audio capabilities to analyze:
- Breath pauses (natural breathing points)
- Pitch resets (sentence/phrase boundaries)
- Emphasis patterns (stressed words)
- Syllable stress (for Portuguese)

This analysis helps distinguish natural pauses from transcription artifacts.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from wordsync.config import Settings, get_settings
from wordsync.transcribe import TranscriptionResult, Word


class ProsodyEventType(str, Enum):
    """Types of prosodic events."""

    BREATH_PAUSE = "breath_pause"
    PITCH_RESET = "pitch_reset"
    EMPHASIS = "emphasis"
    SENTENCE_END = "sentence_end"
    PHRASE_BOUNDARY = "phrase_boundary"
    HESITATION = "hesitation"


@dataclass
class WordDurationCorrection:
    """Correction for a single word's duration."""

    word_index: int
    word: str
    original_start: float
    original_end: float
    corrected_start: float
    corrected_end: float
    confidence: float
    reason: str  # e.g., "3-syllable word cannot be 40ms"

    @property
    def original_duration_ms(self) -> float:
        return (self.original_end - self.original_start) * 1000

    @property
    def corrected_duration_ms(self) -> float:
        return (self.corrected_end - self.corrected_start) * 1000


@dataclass
class WordDurationAnalysis:
    """Result of Gemini word duration analysis."""

    audio_file: str
    corrections: list[WordDurationCorrection]
    confidence: float
    provider: str = "gemini"

    @property
    def words_corrected(self) -> int:
        return len(self.corrections)


@dataclass
class ProsodyEvent:
    """A detected prosodic event."""

    event_type: ProsodyEventType
    start_time: float  # Seconds
    end_time: float | None = None  # Seconds (None for point events)
    confidence: float = 0.8
    description: str = ""

    @property
    def duration(self) -> float:
        """Duration of event in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3) if self.end_time else None,
            "confidence": round(self.confidence, 3),
            "description": self.description,
        }


@dataclass
class GapAnalysis:
    """Analysis of a gap between words."""

    gap_start: float
    gap_end: float
    word_before: str
    word_after: str
    has_breath: bool = False
    has_pitch_reset: bool = False
    has_emphasis_change: bool = False
    is_sentence_boundary: bool = False
    confidence: float = 0.8
    recommendation: str = "analyze"  # "keep", "fill", "analyze"

    @property
    def duration_ms(self) -> float:
        """Gap duration in milliseconds."""
        return (self.gap_end - self.gap_start) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_start": round(self.gap_start, 3),
            "gap_end": round(self.gap_end, 3),
            "duration_ms": round(self.duration_ms, 1),
            "word_before": self.word_before,
            "word_after": self.word_after,
            "has_breath": self.has_breath,
            "has_pitch_reset": self.has_pitch_reset,
            "has_emphasis_change": self.has_emphasis_change,
            "is_sentence_boundary": self.is_sentence_boundary,
            "confidence": round(self.confidence, 3),
            "recommendation": self.recommendation,
        }


@dataclass
class ProsodyResult:
    """Result of prosody analysis."""

    audio_file: str
    events: list[ProsodyEvent]
    gap_analyses: list[GapAnalysis]
    confidence: float = 0.8
    provider: str = "gemini"
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def breath_pauses(self) -> list[ProsodyEvent]:
        """Get all breath pause events."""
        return [e for e in self.events if e.event_type == ProsodyEventType.BREATH_PAUSE]

    @property
    def pitch_resets(self) -> list[ProsodyEvent]:
        """Get all pitch reset events."""
        return [e for e in self.events if e.event_type == ProsodyEventType.PITCH_RESET]

    @property
    def gaps_to_keep(self) -> list[GapAnalysis]:
        """Get gaps recommended to keep."""
        return [g for g in self.gap_analyses if g.recommendation == "keep"]

    @property
    def gaps_to_fill(self) -> list[GapAnalysis]:
        """Get gaps recommended to fill."""
        return [g for g in self.gap_analyses if g.recommendation == "fill"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audio_file": self.audio_file,
            "confidence": round(self.confidence, 3),
            "provider": self.provider,
            "events": [e.to_dict() for e in self.events],
            "gap_analyses": [g.to_dict() for g in self.gap_analyses],
            "summary": {
                "total_events": len(self.events),
                "breath_pauses": len(self.breath_pauses),
                "pitch_resets": len(self.pitch_resets),
                "gaps_to_keep": len(self.gaps_to_keep),
                "gaps_to_fill": len(self.gaps_to_fill),
            },
        }


def analyze_prosody(
    audio_path: str | Path,
    transcription: TranscriptionResult,
    settings: Settings | None = None,
) -> ProsodyResult:
    """
    Analyze prosodic features in audio.

    Uses Gemini 2.5 Flash native audio capabilities to detect
    breath pauses, pitch resets, and other prosodic markers.

    Args:
        audio_path: Path to audio file
        transcription: Transcription result with word timestamps
        settings: Settings instance

    Returns:
        ProsodyResult with events and gap analyses

    Raises:
        ValueError: If Google API key not configured
        RuntimeError: If analysis fails
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    if not settings.prosody.enabled:
        # Return empty result with heuristic-based analysis
        return _analyze_with_heuristics(audio_path, transcription, settings)

    if not settings.has_google:
        # Fall back to heuristics
        return _analyze_with_heuristics(audio_path, transcription, settings)

    try:
        return _analyze_with_gemini(audio_path, transcription, settings)
    except Exception as e:
        if settings.debug:
            print(f"Gemini analysis failed: {e}, falling back to heuristics")
        return _analyze_with_heuristics(audio_path, transcription, settings)


def _analyze_with_gemini(
    audio_path: Path,
    transcription: TranscriptionResult,
    settings: Settings,
) -> ProsodyResult:
    """
    Analyze prosody using Gemini 2.5 Flash.

    Sends audio to Gemini with a prompt to identify prosodic features.
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

    # Build gap information for context
    gaps_info = _extract_gaps(transcription)
    gaps_text = "\n".join([
        f"- Gap at {g['start']:.2f}s-{g['end']:.2f}s ({g['duration_ms']:.0f}ms): "
        f"'{g['word_before']}' -> '{g['word_after']}'"
        for g in gaps_info[:20]  # Limit to first 20 gaps
    ])

    prompt = f"""Analyze the prosodic features in this Portuguese audio recording.
Focus on identifying:

1. BREATH PAUSES: Points where the speaker takes a breath
2. PITCH RESETS: Points where pitch drops/resets (sentence boundaries)
3. EMPHASIS: Words that are stressed or emphasized
4. PHRASE BOUNDARIES: Natural pause points between phrases

Here are the gaps between transcribed words that need analysis:
{gaps_text}

For each gap, determine if it contains:
- A breath (audible inhalation)
- A pitch reset (falling intonation)
- A sentence boundary marker

Return your analysis as JSON with this structure:
{{
    "events": [
        {{"type": "breath_pause", "time": 1.5, "confidence": 0.9}},
        {{"type": "pitch_reset", "time": 2.3, "confidence": 0.85}},
        {{"type": "emphasis", "time": 3.1, "word": "importante", "confidence": 0.8}}
    ],
    "gap_analyses": [
        {{
            "gap_start": 1.2,
            "gap_end": 1.5,
            "has_breath": true,
            "has_pitch_reset": false,
            "is_sentence_boundary": false,
            "recommendation": "keep",
            "confidence": 0.85
        }}
    ],
    "overall_confidence": 0.82
}}

Important:
- Times are in seconds
- confidence is 0-1
- recommendation is "keep" (preserve natural pause) or "fill" (transcription artifact)
- Focus on gaps > 50ms as those are most important to classify
"""

    # Create audio part
    audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)

    response = client.models.generate_content(
        model=settings.prosody.model,
        contents=[audio_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    # Parse response
    response_text = response.text
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise RuntimeError(f"Failed to parse Gemini response: {response_text[:200]}")

    # Convert to ProsodyResult
    events = []
    for event_data in data.get("events", []):
        event_type_str = event_data.get("type", "phrase_boundary")
        try:
            event_type = ProsodyEventType(event_type_str)
        except ValueError:
            event_type = ProsodyEventType.PHRASE_BOUNDARY

        events.append(ProsodyEvent(
            event_type=event_type,
            start_time=event_data.get("time", 0.0),
            confidence=event_data.get("confidence", 0.8),
            description=event_data.get("word", ""),
        ))

    gap_analyses = []
    for gap_data in data.get("gap_analyses", []):
        # Find matching words
        word_before = ""
        word_after = ""
        gap_start = gap_data.get("gap_start", 0.0)
        gap_end = gap_data.get("gap_end", 0.0)

        for i, word in enumerate(transcription.words[:-1]):
            if abs(word.end - gap_start) < 0.1:
                word_before = word.word
                word_after = transcription.words[i + 1].word
                break

        gap_analyses.append(GapAnalysis(
            gap_start=gap_start,
            gap_end=gap_end,
            word_before=word_before,
            word_after=word_after,
            has_breath=gap_data.get("has_breath", False),
            has_pitch_reset=gap_data.get("has_pitch_reset", False),
            is_sentence_boundary=gap_data.get("is_sentence_boundary", False),
            confidence=gap_data.get("confidence", 0.8),
            recommendation=gap_data.get("recommendation", "analyze"),
        ))

    return ProsodyResult(
        audio_file=audio_path.name,
        events=events,
        gap_analyses=gap_analyses,
        confidence=data.get("overall_confidence", 0.8),
        provider="gemini",
        raw_response=data,
    )


def _analyze_with_heuristics(
    audio_path: Path,
    transcription: TranscriptionResult,
    settings: Settings,
) -> ProsodyResult:
    """
    Analyze prosody using duration-based heuristics.

    Falls back to this when Gemini is unavailable.
    Uses gap duration and punctuation to infer prosodic features.
    """
    thresholds = settings.gap_thresholds
    events = []
    gap_analyses = []

    gaps_info = _extract_gaps(transcription)

    for gap in gaps_info:
        duration_ms = gap["duration_ms"]
        word_before = gap["word_before"]

        # Determine prosodic features based on duration and punctuation
        has_breath = duration_ms > thresholds.natural_pause_max
        has_pitch_reset = _ends_with_sentence_punctuation(word_before)
        is_sentence_boundary = has_pitch_reset and duration_ms > thresholds.sentence_boundary_min

        # Determine recommendation
        if duration_ms < thresholds.micro_gap_max:
            recommendation = "fill"
            confidence = 0.95
        elif duration_ms < thresholds.short_gap_max:
            # Check for Portuguese liaisons
            if _is_portuguese_liaison(word_before, gap["word_after"], settings):
                recommendation = "fill"
                confidence = 0.85
            else:
                recommendation = "analyze"
                confidence = 0.6
        elif duration_ms < thresholds.medium_gap_max:
            recommendation = "analyze"
            confidence = 0.5
        elif duration_ms < thresholds.natural_pause_max:
            recommendation = "keep" if has_pitch_reset else "analyze"
            confidence = 0.7 if has_pitch_reset else 0.5
        else:
            recommendation = "keep"
            confidence = 0.85

        gap_analyses.append(GapAnalysis(
            gap_start=gap["start"],
            gap_end=gap["end"],
            word_before=word_before,
            word_after=gap["word_after"],
            has_breath=has_breath,
            has_pitch_reset=has_pitch_reset,
            is_sentence_boundary=is_sentence_boundary,
            confidence=confidence,
            recommendation=recommendation,
        ))

        # Create events for significant pauses
        if has_pitch_reset:
            events.append(ProsodyEvent(
                event_type=ProsodyEventType.PITCH_RESET,
                start_time=gap["start"],
                confidence=0.7 if _ends_with_sentence_punctuation(word_before) else 0.5,
            ))

        if is_sentence_boundary:
            events.append(ProsodyEvent(
                event_type=ProsodyEventType.SENTENCE_END,
                start_time=gap["start"],
                confidence=0.8,
            ))

    return ProsodyResult(
        audio_file=audio_path.name,
        events=events,
        gap_analyses=gap_analyses,
        confidence=0.6,  # Lower confidence for heuristics
        provider="heuristics",
    )


def _extract_gaps(transcription: TranscriptionResult) -> list[dict[str, Any]]:
    """Extract gaps between consecutive words."""
    gaps = []

    for i in range(len(transcription.words) - 1):
        current = transcription.words[i]
        next_word = transcription.words[i + 1]

        gap_start = current.end
        gap_end = next_word.start
        duration_ms = (gap_end - gap_start) * 1000

        if duration_ms > 0:  # Only include positive gaps
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration_ms": duration_ms,
                "word_before": current.word,
                "word_after": next_word.word,
            })

    return gaps


def _ends_with_sentence_punctuation(word: str) -> bool:
    """Check if word ends with sentence-ending punctuation."""
    sentence_endings = ".!?;:"
    word = word.strip()
    if not word:
        return False
    return word[-1] in sentence_endings


def _is_portuguese_liaison(word_before: str, word_after: str, settings: Settings) -> bool:
    """
    Check if words should be connected (Portuguese liaison rules).

    Returns True if the gap should be filled to maintain natural speech flow.
    """
    rules = settings.portuguese_rules
    word_before_clean = word_before.strip().lower().rstrip(".,!?;:")
    word_after_clean = word_after.strip().lower()

    # Article + noun liaison
    if rules.article_noun_liaison:
        if word_before_clean in rules.articles:
            return True

    # Preposition liaison
    if rules.preposition_liaison:
        if word_before_clean in rules.prepositions:
            return True

    return False


def analyze_word_durations(
    audio_path: str | Path,
    transcription: TranscriptionResult,
    settings: Settings | None = None,
) -> WordDurationAnalysis:
    """
    Ask Gemini to analyze and correct word durations.

    Sends audio to Gemini with Whisper's timestamps, asks it to identify
    any words whose duration doesn't match what's actually spoken.

    Args:
        audio_path: Path to audio file
        transcription: Transcription result with word timestamps
        settings: Settings instance

    Returns:
        WordDurationAnalysis with corrections for any problematic words
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    if not settings.has_google:
        return WordDurationAnalysis(
            audio_file=audio_path.name,
            corrections=[],
            confidence=0.0,
            provider="none",
        )

    try:
        # First pass: general duration analysis
        analysis = _analyze_durations_with_gemini(audio_path, transcription, settings)

        # Second pass: specifically check words after long pauses
        # These often have wrong start times that Gemini misses in the general pass
        pause_corrections = _analyze_post_pause_words(
            audio_path, transcription, settings
        )

        # Merge corrections (pause corrections take priority for start times)
        if pause_corrections:
            existing_indices = {c.word_index for c in analysis.corrections}
            for pc in pause_corrections:
                if pc.word_index in existing_indices:
                    # Update existing correction with better start time
                    for i, c in enumerate(analysis.corrections):
                        if c.word_index == pc.word_index:
                            analysis.corrections[i] = pc
                            break
                else:
                    analysis.corrections.append(pc)

        return analysis
    except Exception as e:
        if settings.debug:
            print(f"Gemini duration analysis failed: {e}")
        return WordDurationAnalysis(
            audio_file=audio_path.name,
            corrections=[],
            confidence=0.0,
            provider="error",
        )


def _analyze_durations_with_gemini(
    audio_path: Path,
    transcription: TranscriptionResult,
    settings: Settings,
) -> WordDurationAnalysis:
    """Use Gemini to analyze word durations from audio."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)

    # Prepare audio
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    suffix = audio_path.suffix.lower()
    mime_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
    mime_type = mime_types.get(suffix, "audio/mpeg")

    # Build word list with Whisper timestamps
    words_info = [
        {
            "index": i,
            "word": w.word,
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "duration_ms": round((w.end - w.start) * 1000, 1),
        }
        for i, w in enumerate(transcription.words)
    ]

    prompt = f"""Analyze the word timestamps in this Portuguese audio recording.

I have timestamps from Whisper, but many are WRONG. Please listen carefully to the audio
and identify words whose START TIME or END TIME doesn't match what you actually hear.

Whisper's timestamps:
{json.dumps(words_info, indent=2, ensure_ascii=False)}

COMMON PROBLEMS TO FIX:
1. LATE START: Word starts later than the actual speech (causes delayed highlighting)
   - Listen for when the consonant/vowel sound actually begins
   - The corrected_start should be when the word FIRST becomes audible

2. EARLY START: Word starts before the actual speech (causes early highlighting)

3. WRONG DURATION: Duration doesn't match syllable count
   - 1-syllable words (um, lhe, os): typically 100-200ms
   - 2-syllable words (falo, tudo): typically 200-350ms
   - 3-syllable words (amigo, aberto): typically 300-500ms

4. STOLEN TIME: One word "steals" time from adjacent words
   - If word A is too long and word B is too short, redistribute time

Listen to EACH word and check:
- Does the word actually START at the indicated time, or earlier/later?
- Does the word actually END at the indicated time, or earlier/later?

Return JSON with corrections:
{{
    "corrections": [
        {{
            "word_index": 5,
            "word": "AMIGO",
            "original_start": 6.94,
            "original_end": 6.98,
            "corrected_start": 6.50,
            "corrected_end": 6.94,
            "confidence": 0.9,
            "reason": "Word starts at 6.50s in audio, not 6.94s"
        }}
    ],
    "overall_confidence": 0.85
}}

CRITICAL INSTRUCTIONS:
1. Focus especially on START TIMES. A word with a late start time will appear
   to highlight AFTER the user hears it, which is very jarring.

2. For words that come AFTER A PAUSE (gap > 500ms), pay extra attention:
   - These words often have incorrect start times from Whisper
   - Listen for the EXACT moment the first consonant/vowel sound begins
   - The start time should be when you FIRST hear the word, not when it's fully formed

3. Listen to each word multiple times if needed to get accurate timestamps.
"""

    # Create audio part
    audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)

    response = client.models.generate_content(
        model=settings.prosody.model,
        contents=[audio_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    data = json.loads(response.text)

    corrections = []
    for corr in data.get("corrections", []):
        corrections.append(
            WordDurationCorrection(
                word_index=corr["word_index"],
                word=corr["word"],
                original_start=corr["original_start"],
                original_end=corr["original_end"],
                corrected_start=corr["corrected_start"],
                corrected_end=corr["corrected_end"],
                confidence=corr.get("confidence", 0.8),
                reason=corr.get("reason", ""),
            )
        )

    return WordDurationAnalysis(
        audio_file=audio_path.name,
        corrections=corrections,
        confidence=data.get("overall_confidence", 0.8),
        provider="gemini",
    )


def _analyze_post_pause_words(
    audio_path: Path,
    transcription: TranscriptionResult,
    settings: Settings,
    min_pause_ms: float = 500,
) -> list[WordDurationCorrection]:
    """
    Specifically analyze words that come after long pauses.

    These words often have incorrect start times that the general analysis misses.
    We ask Gemini directly: "When does word X start?"
    """
    from google import genai
    from google.genai import types

    audio_path = Path(audio_path)

    # Find words after long pauses
    post_pause_words = []
    for i in range(1, len(transcription.words)):
        prev = transcription.words[i - 1]
        curr = transcription.words[i]
        gap_ms = (curr.start - prev.end) * 1000

        if gap_ms >= min_pause_ms:
            post_pause_words.append({
                "index": i,
                "word": curr.word,
                "whisper_start": curr.start,
                "whisper_end": curr.end,
                "prev_word": prev.word,
                "prev_end": prev.end,
                "gap_ms": gap_ms,
            })

    if not post_pause_words:
        return []

    client = genai.Client(api_key=settings.google_api_key)

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    suffix = audio_path.suffix.lower()
    mime_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
    mime_type = mime_types.get(suffix, "audio/mpeg")
    audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)

    # Build prompt for specific words
    words_to_check = "\n".join([
        f"- Word {w['index']}: \"{w['word']}\" (after \"{w['prev_word']}\", "
        f"Whisper says starts at {w['whisper_start']:.2f}s)"
        for w in post_pause_words
    ])

    prompt = f"""Listen to this Portuguese audio and tell me the EXACT START TIME for these words.

These words come after a pause/silence, and Whisper may have gotten the start time wrong.
I need to know when you FIRST hear each word begin.

Words to check:
{words_to_check}

For each word, listen carefully and tell me:
- The exact second when the word STARTS (when you first hear the sound)
- The exact second when the word ENDS

Return JSON:
{{
    "words": [
        {{"index": 14, "word": "ABERTO", "start": 8.50, "end": 9.10}},
        {{"index": 18, "word": "GUARDO", "start": 11.30, "end": 11.75}}
    ]
}}

IMPORTANT: Focus on the START time. When does the sound actually begin?
Listen multiple times if needed. Be precise to 0.05 seconds."""

    try:
        response = client.models.generate_content(
            model=settings.prosody.model,
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        data = json.loads(response.text)
        corrections = []

        for word_data in data.get("words", []):
            idx = word_data.get("index")
            if idx is None:
                continue

            # Find the original word info
            orig = None
            for w in post_pause_words:
                if w["index"] == idx:
                    orig = w
                    break

            if not orig:
                continue

            corrections.append(WordDurationCorrection(
                word_index=idx,
                word=word_data.get("word", orig["word"]),
                original_start=orig["whisper_start"],
                original_end=orig["whisper_end"],
                corrected_start=word_data.get("start", orig["whisper_start"]),
                corrected_end=word_data.get("end", orig["whisper_end"]),
                confidence=0.85,
                reason=f"Post-pause word, start time verified by focused analysis",
            ))

        return corrections

    except Exception:
        return []


def apply_duration_corrections(
    words: list[Word],
    analysis: WordDurationAnalysis,
    min_confidence: float = 0.7,
) -> list[Word]:
    """
    Apply Gemini's duration corrections to word list.

    Only applies corrections with confidence >= min_confidence.

    Args:
        words: List of words with timestamps
        analysis: WordDurationAnalysis with corrections
        min_confidence: Minimum confidence threshold for applying corrections

    Returns:
        List of words with corrected timestamps
    """
    from copy import deepcopy

    words = [deepcopy(w) for w in words]

    for correction in analysis.corrections:
        if correction.confidence < min_confidence:
            continue

        idx = correction.word_index
        if idx < 0 or idx >= len(words):
            continue

        # Verify it's the same word (strip punctuation for comparison)
        import re
        word_clean = re.sub(r'[^\w]', '', words[idx].word.upper())
        corr_clean = re.sub(r'[^\w]', '', correction.word.upper())
        if word_clean != corr_clean:
            continue

        # Apply correction
        words[idx].start = correction.corrected_start
        words[idx].end = correction.corrected_end

    return words
