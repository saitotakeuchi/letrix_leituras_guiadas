"""
WordSync Gap Classification Module

Multi-factor gap classification to determine whether gaps between words
should be preserved (natural pauses) or filled (transcription artifacts).

Classification factors:
- Duration (ms)
- Punctuation context
- Audio evidence (from prosody analysis)
- Portuguese liaison rules
- Educational pacing considerations

Gap Types:
| Type            | Duration    | Audio Evidence        | Action |
|-----------------|-------------|----------------------|--------|
| Micro-gap       | <50ms       | -                    | FILL   |
| Short gap       | 50-150ms    | No breath/pitch      | FILL   |
| Medium gap      | 150-400ms   | Context-dependent    | Analyze|
| Natural pause   | 400-600ms   | Breath or pitch      | KEEP   |
| Sentence bound  | >600ms      | Punctuation + audio  | KEEP   |
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wordsync.config import Settings, get_settings
from wordsync.transcribe import TranscriptionResult, Word
from wordsync.prosody import ProsodyResult, GapAnalysis


class GapAction(str, Enum):
    """Action to take for a gap."""

    KEEP = "keep"  # Preserve the natural pause
    FILL = "fill"  # Fill the gap (transcription artifact)
    REVIEW = "review"  # Flag for manual review


class GapType(str, Enum):
    """Classification of gap type."""

    MICRO = "micro"  # <50ms - always fill
    SHORT = "short"  # 50-150ms - usually fill
    MEDIUM = "medium"  # 150-400ms - context dependent
    NATURAL = "natural"  # 400-600ms - usually keep
    SENTENCE = "sentence"  # >600ms - always keep


@dataclass
class GapClassification:
    """Classification result for a single gap."""

    gap_index: int  # Index in word list (gap after word at this index)
    word_before: Word
    word_after: Word
    gap_start: float
    gap_end: float
    duration_ms: float

    gap_type: GapType
    action: GapAction
    confidence: float

    # Classification factors
    has_punctuation: bool = False
    has_breath: bool = False
    has_pitch_reset: bool = False
    is_liaison: bool = False
    is_sentence_boundary: bool = False

    # For filling gaps
    fill_start: float | None = None
    fill_end: float | None = None

    # NEW: True if words need duration expansion (overlapping/compressed words)
    needs_expansion: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_index": self.gap_index,
            "word_before": self.word_before.word,
            "word_after": self.word_after.word,
            "gap_start": round(self.gap_start, 3),
            "gap_end": round(self.gap_end, 3),
            "duration_ms": round(self.duration_ms, 1),
            "gap_type": self.gap_type.value,
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "factors": {
                "has_punctuation": self.has_punctuation,
                "has_breath": self.has_breath,
                "has_pitch_reset": self.has_pitch_reset,
                "is_liaison": self.is_liaison,
                "is_sentence_boundary": self.is_sentence_boundary,
            },
            "fill": {
                "start": round(self.fill_start, 3) if self.fill_start else None,
                "end": round(self.fill_end, 3) if self.fill_end else None,
            } if self.action == GapAction.FILL else None,
        }


@dataclass
class ClassificationResult:
    """Result of classifying all gaps in a transcription."""

    classifications: list[GapClassification]
    total_gaps: int
    gaps_kept: int
    gaps_filled: int
    gaps_review: int
    average_confidence: float
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gaps": self.total_gaps,
            "gaps_kept": self.gaps_kept,
            "gaps_filled": self.gaps_filled,
            "gaps_review": self.gaps_review,
            "average_confidence": round(self.average_confidence, 3),
            "classifications": [c.to_dict() for c in self.classifications],
            "summary": self.summary,
        }


def classify_gaps(
    transcription: TranscriptionResult,
    prosody: ProsodyResult | None = None,
    settings: Settings | None = None,
) -> ClassificationResult:
    """
    Classify all gaps in a transcription.

    Analyzes each gap between consecutive words and determines
    whether to keep the natural pause or fill the gap.

    Args:
        transcription: Transcription result with word timestamps
        prosody: Optional prosody analysis result
        settings: Settings instance

    Returns:
        ClassificationResult with all gap classifications
    """
    settings = settings or get_settings()
    classifications = []

    words = transcription.words
    thresholds = settings.gap_thresholds
    rules = settings.portuguese_rules

    for i in range(len(words) - 1):
        word_before = words[i]
        word_after = words[i + 1]

        gap_start = word_before.end
        gap_end = word_after.start
        duration_ms = (gap_end - gap_start) * 1000

        # Handle negative or zero gaps (overlapping/compressed words)
        if duration_ms <= 0:
            # Create FILL classification with needs_expansion flag
            # This signals that the words need duration expansion, not gap filling
            classification = GapClassification(
                gap_index=i,
                word_before=word_before,
                word_after=word_after,
                gap_start=gap_start,
                gap_end=gap_end,
                duration_ms=duration_ms,
                gap_type=GapType.MICRO,
                action=GapAction.FILL,
                confidence=0.99,
                needs_expansion=True,  # Flag for apply_classifications to skip
                has_punctuation=_has_sentence_punctuation(word_before.word),
            )
            classifications.append(classification)
            continue

        # Get prosody analysis for this gap if available
        prosody_gap = None
        if prosody:
            prosody_gap = _find_prosody_gap(prosody, gap_start, gap_end)

        # Classify the gap
        classification = _classify_single_gap(
            gap_index=i,
            word_before=word_before,
            word_after=word_after,
            gap_start=gap_start,
            gap_end=gap_end,
            duration_ms=duration_ms,
            prosody_gap=prosody_gap,
            thresholds=thresholds,
            rules=rules,
        )

        classifications.append(classification)

    # Calculate summary statistics
    gaps_kept = sum(1 for c in classifications if c.action == GapAction.KEEP)
    gaps_filled = sum(1 for c in classifications if c.action == GapAction.FILL)
    gaps_review = sum(1 for c in classifications if c.action == GapAction.REVIEW)

    avg_confidence = (
        sum(c.confidence for c in classifications) / len(classifications)
        if classifications else 0.0
    )

    # Summary by gap type
    summary = {
        "by_type": {},
        "by_action": {
            "keep": gaps_kept,
            "fill": gaps_filled,
            "review": gaps_review,
        },
    }

    for gap_type in GapType:
        type_gaps = [c for c in classifications if c.gap_type == gap_type]
        if type_gaps:
            summary["by_type"][gap_type.value] = {
                "count": len(type_gaps),
                "kept": sum(1 for c in type_gaps if c.action == GapAction.KEEP),
                "filled": sum(1 for c in type_gaps if c.action == GapAction.FILL),
            }

    return ClassificationResult(
        classifications=classifications,
        total_gaps=len(classifications),
        gaps_kept=gaps_kept,
        gaps_filled=gaps_filled,
        gaps_review=gaps_review,
        average_confidence=avg_confidence,
        summary=summary,
    )


def _classify_single_gap(
    gap_index: int,
    word_before: Word,
    word_after: Word,
    gap_start: float,
    gap_end: float,
    duration_ms: float,
    prosody_gap: GapAnalysis | None,
    thresholds: Any,
    rules: Any,
) -> GapClassification:
    """
    Classify a single gap between words.

    Uses multi-factor analysis to determine the appropriate action.
    """
    # Determine gap type based on duration
    if duration_ms < thresholds.micro_gap_max:
        gap_type = GapType.MICRO
    elif duration_ms < thresholds.short_gap_max:
        gap_type = GapType.SHORT
    elif duration_ms < thresholds.medium_gap_max:
        gap_type = GapType.MEDIUM
    elif duration_ms < thresholds.natural_pause_max:
        gap_type = GapType.NATURAL
    else:
        gap_type = GapType.SENTENCE

    # Check punctuation
    has_punctuation = _has_sentence_punctuation(word_before.word)

    # Check prosody factors
    has_breath = prosody_gap.has_breath if prosody_gap else False
    has_pitch_reset = prosody_gap.has_pitch_reset if prosody_gap else has_punctuation
    is_sentence_boundary = prosody_gap.is_sentence_boundary if prosody_gap else (
        has_punctuation and gap_type in (GapType.NATURAL, GapType.SENTENCE)
    )

    # Check liaison rules
    is_liaison = _is_liaison(word_before.word, word_after.word, rules)

    # Determine action based on all factors
    action, confidence = _determine_action(
        gap_type=gap_type,
        has_punctuation=has_punctuation,
        has_breath=has_breath,
        has_pitch_reset=has_pitch_reset,
        is_liaison=is_liaison,
        is_sentence_boundary=is_sentence_boundary,
        prosody_confidence=prosody_gap.confidence if prosody_gap else None,
        prosody_recommendation=prosody_gap.recommendation if prosody_gap else None,
    )

    # Calculate fill timestamps if needed
    fill_start = None
    fill_end = None
    if action == GapAction.FILL:
        # Extend word_before's end to meet word_after's start
        # Split the gap proportionally
        fill_start = gap_start
        fill_end = gap_end

    return GapClassification(
        gap_index=gap_index,
        word_before=word_before,
        word_after=word_after,
        gap_start=gap_start,
        gap_end=gap_end,
        duration_ms=duration_ms,
        gap_type=gap_type,
        action=action,
        confidence=confidence,
        has_punctuation=has_punctuation,
        has_breath=has_breath,
        has_pitch_reset=has_pitch_reset,
        is_liaison=is_liaison,
        is_sentence_boundary=is_sentence_boundary,
        fill_start=fill_start,
        fill_end=fill_end,
    )


def _determine_action(
    gap_type: GapType,
    has_punctuation: bool,
    has_breath: bool,
    has_pitch_reset: bool,
    is_liaison: bool,
    is_sentence_boundary: bool,
    prosody_confidence: float | None,
    prosody_recommendation: str | None,
) -> tuple[GapAction, float]:
    """
    Determine the action for a gap based on all factors.

    Returns tuple of (action, confidence).
    """
    # Use prosody recommendation if available with high confidence
    if prosody_recommendation and prosody_confidence and prosody_confidence > 0.8:
        if prosody_recommendation == "keep":
            return GapAction.KEEP, prosody_confidence
        elif prosody_recommendation == "fill":
            return GapAction.FILL, prosody_confidence

    # Micro gaps - always fill
    if gap_type == GapType.MICRO:
        return GapAction.FILL, 0.95

    # Sentence boundaries - always keep
    if gap_type == GapType.SENTENCE:
        return GapAction.KEEP, 0.95

    # Natural pauses with audio evidence - keep
    if gap_type == GapType.NATURAL:
        if has_breath or has_pitch_reset or has_punctuation:
            return GapAction.KEEP, 0.85
        return GapAction.REVIEW, 0.6

    # Short gaps
    if gap_type == GapType.SHORT:
        if is_liaison:
            return GapAction.FILL, 0.85
        if has_breath or has_pitch_reset:
            return GapAction.KEEP, 0.7
        return GapAction.FILL, 0.75

    # Medium gaps - most complex
    if gap_type == GapType.MEDIUM:
        # Strong evidence to keep
        if is_sentence_boundary:
            return GapAction.KEEP, 0.9
        if has_breath and has_pitch_reset:
            return GapAction.KEEP, 0.85
        if has_punctuation:
            return GapAction.KEEP, 0.8

        # Strong evidence to fill
        if is_liaison:
            return GapAction.FILL, 0.8

        # Weak evidence
        if has_breath or has_pitch_reset:
            return GapAction.KEEP, 0.65
        if prosody_confidence and prosody_confidence > 0.6:
            if prosody_recommendation == "keep":
                return GapAction.KEEP, prosody_confidence
            elif prosody_recommendation == "fill":
                return GapAction.FILL, prosody_confidence

        # Uncertain - flag for review
        return GapAction.REVIEW, 0.5

    # Default
    return GapAction.REVIEW, 0.5


def _has_sentence_punctuation(word: str) -> bool:
    """Check if word ends with sentence-ending punctuation."""
    word = word.strip()
    if not word:
        return False
    return word[-1] in ".!?;:"


def _is_liaison(word_before: str, word_after: str, rules: Any) -> bool:
    """
    Check if words form a Portuguese liaison.

    Liaisons should have gaps filled to maintain natural speech flow.
    """
    word_before_clean = word_before.strip().lower().rstrip(".,!?;:'\"")
    word_after_clean = word_after.strip().lower()

    # Article + noun
    if rules.article_noun_liaison and word_before_clean in rules.articles:
        return True

    # Preposition + object
    if rules.preposition_liaison and word_before_clean in rules.prepositions:
        return True

    return False


def _find_prosody_gap(
    prosody: ProsodyResult,
    gap_start: float,
    gap_end: float,
    tolerance: float = 0.1,
) -> GapAnalysis | None:
    """
    Find matching prosody gap analysis.

    Matches by time with tolerance.
    """
    for gap in prosody.gap_analyses:
        if (abs(gap.gap_start - gap_start) < tolerance and
            abs(gap.gap_end - gap_end) < tolerance):
            return gap

    # Try to match by just start time
    for gap in prosody.gap_analyses:
        if abs(gap.gap_start - gap_start) < tolerance:
            return gap

    return None


def apply_classifications(
    words: list[Word],
    classifications: ClassificationResult,
) -> list[Word]:
    """
    Apply gap classifications to modify word timestamps.

    For FILL actions, extends word end times to close gaps.

    SAFEGUARDS (new):
    - Skip gaps marked with needs_expansion (handled by _enforce_minimum_durations)
    - Only apply midpoint fills if BOTH words keep adequate duration (>= 150ms)

    Args:
        words: Original word list
        classifications: Classification result

    Returns:
        New list of words with adjusted timestamps
    """
    if not classifications.classifications:
        return words

    # Minimum duration to preserve (150ms)
    min_duration_sec = 0.15

    # Create a copy of words
    adjusted_words = []
    for word in words:
        adjusted_words.append(Word(
            word=word.word,
            start=word.start,
            end=word.end,
            confidence=word.confidence,
            line_break_after=word.line_break_after,
            is_title=word.is_title,
        ))

    # Apply FILL classifications
    for classification in classifications.classifications:
        if classification.action == GapAction.FILL:
            # Skip gaps that need expansion (handled by _enforce_minimum_durations)
            if classification.needs_expansion:
                continue

            idx = classification.gap_index
            if idx < len(adjusted_words) - 1:
                word_before = adjusted_words[idx]
                word_after = adjusted_words[idx + 1]

                # Calculate midpoint for gap filling
                gap_midpoint = (classification.gap_start + classification.gap_end) / 2

                # Calculate new durations if we apply this fill
                new_before_dur = gap_midpoint - word_before.start
                new_after_dur = word_after.end - gap_midpoint

                # SAFEGUARD: Only apply if BOTH words keep adequate duration
                if new_before_dur >= min_duration_sec and new_after_dur >= min_duration_sec:
                    word_before.end = gap_midpoint
                    word_after.start = gap_midpoint
                # Otherwise: skip this gap fill to preserve word durations

    return adjusted_words


def get_low_confidence_gaps(
    result: ClassificationResult,
    threshold: float = 0.7,
) -> list[GapClassification]:
    """
    Get gaps with confidence below threshold.

    These should be flagged for manual review.

    Args:
        result: Classification result
        threshold: Confidence threshold

    Returns:
        List of low-confidence classifications
    """
    return [c for c in result.classifications if c.confidence < threshold]


def get_review_gaps(result: ClassificationResult) -> list[GapClassification]:
    """Get all gaps marked for review."""
    return [c for c in result.classifications if c.action == GapAction.REVIEW]
