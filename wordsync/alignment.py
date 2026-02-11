"""
WordSync Robust Alignment Module

Multi-phase alignment system using:
1. Text structure analysis (title detection, anchor words)
2. Needleman-Wunsch global alignment (DP-based)
3. Gemini verification for title and missing words
4. Interpolation for unverified words

This replaces the greedy forward-only approach that fails when Whisper
misses words at the beginning of audio.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from wordsync.config import Settings, get_settings
from wordsync.transcribe import TranscriptionResult, Word


class AlignmentType(str, Enum):
    """Type of alignment for each word."""

    MATCH = "match"  # Reference word matches transcribed word
    DELETION = "deletion"  # Reference word not in transcription (Whisper missed)
    INSERTION = "insertion"  # Transcribed word not in reference (hallucination)


@dataclass
class AlignedWord(Word):
    """A word with alignment information."""

    alignment_type: AlignmentType = AlignmentType.MATCH
    similarity_score: float = 1.0
    verification_status: str = "unverified"  # "verified", "interpolated", "needs_review"
    ref_index: int = -1  # Index in reference text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["alignment_type"] = self.alignment_type.value
        result["similarity_score"] = round(self.similarity_score, 3)
        result["verification_status"] = self.verification_status
        return result


@dataclass
class TextStructure:
    """Structure analysis of reference text."""

    title_words: list[str]
    body_words: list[str]
    section_boundaries: list[int]  # Word indices where sections start
    total_word_count: int
    anchor_word_indices: list[int] = field(default_factory=list)

    @property
    def all_words(self) -> list[str]:
        """Get all words in order (title + body)."""
        return self.title_words + self.body_words


@dataclass
class TitleVerification:
    """Result of Gemini title verification."""

    expected_title: str
    is_spoken: bool
    actual_first_words: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    confidence: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class WordVerification:
    """Verification result for a missing word."""

    word: str
    ref_index: int
    is_spoken: bool
    start_time: float | None = None
    end_time: float | None = None
    confidence: float = 0.0


@dataclass
class AlignmentResult:
    """Result of the alignment process."""

    aligned_words: list[AlignedWord]
    words_matched: int
    words_deleted: int  # Missing from Whisper
    words_inserted: int  # Hallucinated by Whisper
    words_interpolated: int
    words_verified: int
    title_verified: bool
    alignment_method: str  # "dp", "greedy"
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "words_matched": self.words_matched,
            "words_deleted": self.words_deleted,
            "words_inserted": self.words_inserted,
            "words_interpolated": self.words_interpolated,
            "words_verified": self.words_verified,
            "title_verified": self.title_verified,
            "alignment_method": self.alignment_method,
            "confidence": round(self.confidence, 3),
        }


# =============================================================================
# Text Structure Analysis
# =============================================================================


def analyze_text_structure(text: str) -> TextStructure:
    """
    Analyze the structure of reference text.

    Identifies:
    - Title (first line)
    - Body (remaining lines)
    - Section boundaries (where blank lines occur)
    - Anchor words (long/distinctive words)

    Args:
        text: Reference text with title on first line

    Returns:
        TextStructure with parsed information
    """
    if not text or not text.strip():
        return TextStructure(
            title_words=[],
            body_words=[],
            section_boundaries=[],
            total_word_count=0,
        )

    lines = text.split("\n")

    # Find title (first non-empty line)
    title_line = ""
    body_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            title_line = line.strip()
            body_start_idx = i + 1
            break

    # Extract title words
    title_words = re.findall(r"\S+", title_line)

    # Extract body words and track section boundaries
    body_words = []
    section_boundaries = []
    in_blank_section = False

    for i in range(body_start_idx, len(lines)):
        line = lines[i]
        if not line.strip():
            in_blank_section = True
            continue

        if in_blank_section:
            # Mark section boundary at current word position
            section_boundaries.append(len(body_words))
            in_blank_section = False

        line_words = re.findall(r"\S+", line)
        body_words.extend(line_words)

    # Find anchor words (long, distinctive words)
    all_words = title_words + body_words
    anchor_indices = _find_anchor_words(all_words)

    return TextStructure(
        title_words=title_words,
        body_words=body_words,
        section_boundaries=section_boundaries,
        total_word_count=len(all_words),
        anchor_word_indices=anchor_indices,
    )


def _find_anchor_words(words: list[str]) -> list[int]:
    """
    Find indices of anchor words for reliable alignment.

    Anchor words are:
    - Long words (6+ characters after cleaning)
    - Uncommon words (not articles/prepositions)
    - Words with numbers
    - Words after punctuation

    Args:
        words: List of words

    Returns:
        List of indices of anchor words
    """
    # Common short words to exclude
    common_words = {
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
        "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
        "para", "pra", "pro", "por", "com", "sem",
        "que", "se", "e", "ou", "mas", "mais", "menos",
        "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas",
        "meu", "minha", "seu", "sua", "teu", "tua",
        "este", "esta", "esse", "essa", "aquele", "aquela",
        "nao", "sim", "muito", "pouco", "ja", "ainda",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
    }

    anchors = []
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w]", "", word.lower())

        # Skip if common word
        if clean in common_words:
            continue

        # Long words are anchors
        if len(clean) >= 6:
            anchors.append(i)
            continue

        # Words with numbers are anchors
        if re.search(r"\d", clean):
            anchors.append(i)
            continue

        # Words at start (after punctuation in previous word)
        if i > 0:
            prev_word = words[i - 1]
            if prev_word and prev_word[-1] in ".!?;:":
                anchors.append(i)

    return anchors


# =============================================================================
# Word Similarity
# =============================================================================


def word_similarity(ref: str, trans: str) -> float:
    """
    Calculate similarity between reference and transcribed word.

    Uses multiple strategies:
    1. Exact match after normalization
    2. Substring match
    3. Character overlap ratio
    4. Phonetic similarity (basic Portuguese rules)

    Args:
        ref: Reference word
        trans: Transcribed word

    Returns:
        Similarity score 0-1
    """
    # Normalize: lowercase, remove punctuation
    ref_clean = re.sub(r"[^\w]", "", ref.lower())
    trans_clean = re.sub(r"[^\w]", "", trans.lower())

    if not ref_clean or not trans_clean:
        return 0.0

    # Exact match
    if ref_clean == trans_clean:
        return 1.0

    # Substring match (one contains the other)
    if ref_clean in trans_clean or trans_clean in ref_clean:
        shorter = min(len(ref_clean), len(trans_clean))
        longer = max(len(ref_clean), len(trans_clean))
        return shorter / longer

    # Character overlap (Jaccard-like)
    ref_chars = set(ref_clean)
    trans_chars = set(trans_clean)
    intersection = len(ref_chars & trans_chars)
    union = len(ref_chars | trans_chars)
    char_similarity = intersection / union if union > 0 else 0.0

    # Levenshtein-based similarity
    edit_distance = _levenshtein_distance(ref_clean, trans_clean)
    max_len = max(len(ref_clean), len(trans_clean))
    edit_similarity = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

    # Phonetic similarity for Portuguese
    phonetic_similarity = _phonetic_similarity_pt(ref_clean, trans_clean)

    # Weighted combination
    return max(
        char_similarity * 0.8,
        edit_similarity,
        phonetic_similarity,
    )


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _phonetic_similarity_pt(word1: str, word2: str) -> float:
    """
    Calculate phonetic similarity for Portuguese words.

    Handles common confusions:
    - s/z/ç (same sound in some positions)
    - ão/am (nasal endings)
    - x/ch/s (same sound sometimes)
    - lh/li, nh/ni

    Args:
        word1: First word
        word2: Second word

    Returns:
        Phonetic similarity 0-1
    """
    # Phonetic normalization for Portuguese
    replacements = [
        (r"ç", "s"),
        (r"ss", "s"),
        (r"ão", "am"),
        (r"ões", "ams"),
        (r"ães", "ams"),
        (r"ch", "x"),
        (r"lh", "li"),
        (r"nh", "ni"),
        (r"qu", "k"),
        (r"gu(?=[ei])", "g"),
        (r"rr", "r"),
        (r"h", ""),  # Silent h
    ]

    p1 = word1
    p2 = word2
    for pattern, replacement in replacements:
        p1 = re.sub(pattern, replacement, p1)
        p2 = re.sub(pattern, replacement, p2)

    if p1 == p2:
        return 1.0

    # Check similarity after phonetic normalization
    edit_distance = _levenshtein_distance(p1, p2)
    max_len = max(len(p1), len(p2))
    return 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0


# =============================================================================
# Needleman-Wunsch Global Alignment
# =============================================================================


# Scoring constants
SCORE_EXACT_MATCH = 10
SCORE_FUZZY_MATCH = 7  # >80% similar
SCORE_PHONETIC_MATCH = 5
SCORE_GAP_OPEN = -4
SCORE_GAP_EXTEND = -1
SCORE_MISMATCH = -3


def needleman_wunsch_align(
    ref_words: list[str],
    trans_words: list[Word],
) -> list[tuple[int | None, int | None, AlignmentType]]:
    """
    Global alignment using Needleman-Wunsch dynamic programming.

    Finds optimal alignment between reference text and transcription,
    allowing for gaps (deletions/insertions).

    Args:
        ref_words: Reference text words
        trans_words: Transcribed words with timestamps

    Returns:
        List of (ref_idx, trans_idx, alignment_type) tuples
    """
    m = len(ref_words)
    n = len(trans_words)

    if m == 0 or n == 0:
        # Handle empty cases
        if m == 0:
            return [(None, j, AlignmentType.INSERTION) for j in range(n)]
        else:
            return [(i, None, AlignmentType.DELETION) for i in range(m)]

    # Build scoring matrix
    # score[i][j] = best score aligning ref[0:i] with trans[0:j]
    score = [[0.0] * (n + 1) for _ in range(m + 1)]
    traceback = [[None] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column (gap penalties)
    for i in range(1, m + 1):
        score[i][0] = SCORE_GAP_OPEN + (i - 1) * SCORE_GAP_EXTEND
        traceback[i][0] = "up"  # deletion

    for j in range(1, n + 1):
        score[0][j] = SCORE_GAP_OPEN + (j - 1) * SCORE_GAP_EXTEND
        traceback[0][j] = "left"  # insertion

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ref_word = ref_words[i - 1]
            trans_word = trans_words[j - 1].word

            # Calculate match/mismatch score
            similarity = word_similarity(ref_word, trans_word)
            if similarity >= 0.95:
                match_score = SCORE_EXACT_MATCH
            elif similarity >= 0.8:
                match_score = SCORE_FUZZY_MATCH
            elif similarity >= 0.6:
                match_score = SCORE_PHONETIC_MATCH
            else:
                match_score = SCORE_MISMATCH

            # Three options: diagonal (match/mismatch), up (deletion), left (insertion)
            diag = score[i - 1][j - 1] + match_score

            # Gap penalties with affine gap scoring
            up_score = score[i - 1][j]
            if traceback[i - 1][j] == "up":
                up_score += SCORE_GAP_EXTEND
            else:
                up_score += SCORE_GAP_OPEN

            left_score = score[i][j - 1]
            if traceback[i][j - 1] == "left":
                left_score += SCORE_GAP_EXTEND
            else:
                left_score += SCORE_GAP_OPEN

            # Choose best option
            best = max(diag, up_score, left_score)
            score[i][j] = best

            if best == diag:
                traceback[i][j] = "diag"
            elif best == up_score:
                traceback[i][j] = "up"
            else:
                traceback[i][j] = "left"

    # Traceback to find alignment
    alignment = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i][j] == "diag":
            ref_word = ref_words[i - 1]
            trans_word = trans_words[j - 1].word
            similarity = word_similarity(ref_word, trans_word)
            alignment.append((i - 1, j - 1, AlignmentType.MATCH))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or traceback[i][j] == "up"):
            # Deletion: ref word not in transcription
            alignment.append((i - 1, None, AlignmentType.DELETION))
            i -= 1
        else:
            # Insertion: trans word not in reference
            alignment.append((None, j - 1, AlignmentType.INSERTION))
            j -= 1

    # Reverse to get forward order
    alignment.reverse()
    return alignment


# =============================================================================
# Timestamp Interpolation
# =============================================================================


def interpolate_missing_timestamps(
    aligned_words: list[AlignedWord],
    audio_duration: float,
    average_syllables_per_second: float = 4.0,
) -> list[AlignedWord]:
    """
    Interpolate timestamps for words marked as DELETION.

    Uses surrounding context and syllable estimates to assign
    reasonable timestamps to words that Whisper missed.

    Args:
        aligned_words: Words with alignment info (some may have no timestamps)
        audio_duration: Total audio duration in seconds
        average_syllables_per_second: Speech rate estimate

    Returns:
        Words with interpolated timestamps
    """
    if not aligned_words:
        return aligned_words

    # Find words that need interpolation (DELETION type with no timestamps)
    for i, word in enumerate(aligned_words):
        if word.alignment_type == AlignmentType.DELETION:
            # Find surrounding words with timestamps
            prev_end = 0.0
            next_start = audio_duration

            # Look backward for previous timestamp
            for j in range(i - 1, -1, -1):
                if aligned_words[j].end > 0:
                    prev_end = aligned_words[j].end
                    break

            # Look forward for next timestamp
            for j in range(i + 1, len(aligned_words)):
                if aligned_words[j].start > 0:
                    next_start = aligned_words[j].start
                    break

            # Count how many consecutive words need interpolation
            gap_words = [word]
            for j in range(i + 1, len(aligned_words)):
                if aligned_words[j].alignment_type == AlignmentType.DELETION:
                    gap_words.append(aligned_words[j])
                else:
                    break

            # Calculate available time
            available_time = next_start - prev_end
            if available_time <= 0:
                available_time = 0.3 * len(gap_words)  # Minimum estimate

            # Estimate duration based on syllables
            total_syllables = sum(_estimate_syllables(w.word) for w in gap_words)
            if total_syllables == 0:
                total_syllables = len(gap_words)

            # Distribute time proportionally
            time_per_syllable = available_time / total_syllables
            current_time = prev_end

            for gap_word in gap_words:
                syllables = _estimate_syllables(gap_word.word) or 1
                duration = syllables * time_per_syllable

                # Ensure minimum duration
                duration = max(duration, 0.15)

                gap_word.start = current_time
                gap_word.end = current_time + duration
                gap_word.verification_status = "interpolated"
                gap_word.confidence = 0.3  # Low confidence for interpolated

                current_time += duration

    return aligned_words


def _estimate_syllables(word: str) -> int:
    """
    Estimate syllable count for a Portuguese word.

    Simple heuristic based on vowel clusters.

    Args:
        word: Word to analyze

    Returns:
        Estimated syllable count
    """
    clean = re.sub(r"[^\w]", "", word.lower())
    if not clean:
        return 1

    # Count vowel groups (simplified)
    vowels = "aeiouáéíóúâêôãõàü"
    count = 0
    in_vowel = False

    for char in clean:
        if char in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False

    return max(1, count)


# =============================================================================
# Gemini Verification
# =============================================================================


def verify_title_with_gemini(
    audio_path: Path,
    expected_title: str,
    settings: Settings,
) -> TitleVerification:
    """
    Use Gemini to verify if title is spoken at the beginning of audio.

    Args:
        audio_path: Path to audio file
        expected_title: Expected title text
        settings: Settings instance

    Returns:
        TitleVerification with results
    """
    if not settings.has_google:
        return TitleVerification(
            expected_title=expected_title,
            is_spoken=True,  # Assume spoken if can't verify
            confidence=0.0,
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.google_api_key)

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        suffix = audio_path.suffix.lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
        }
        mime_type = mime_types.get(suffix, "audio/mpeg")

        prompt = f"""Listen to the BEGINNING of this Portuguese audio recording.

Expected title: "{expected_title}"

Questions:
1. Is the word/phrase "{expected_title}" spoken at the very beginning?
2. If yes, at what time does it start and end?
3. If no, what is the FIRST word actually spoken?

Return JSON:
{{
    "is_spoken": true/false,
    "actual_first_words": "...",  // What you actually hear first
    "start_time": 0.0,  // When title starts (if spoken)
    "end_time": 0.5,    // When title ends (if spoken)
    "confidence": 0.9
}}

Be precise with timestamps. Focus on the first 2-3 seconds of audio."""

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

        return TitleVerification(
            expected_title=expected_title,
            is_spoken=data.get("is_spoken", True),
            actual_first_words=data.get("actual_first_words"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            confidence=data.get("confidence", 0.8),
            raw_response=data,
        )

    except Exception as e:
        if settings.debug:
            print(f"Gemini title verification failed: {e}")
        return TitleVerification(
            expected_title=expected_title,
            is_spoken=True,  # Assume spoken on error
            confidence=0.0,
        )


def verify_missing_words(
    audio_path: Path,
    missing_words: list[tuple[str, int, float, float]],  # (word, ref_idx, time_before, time_after)
    settings: Settings,
) -> list[WordVerification]:
    """
    Use Gemini to verify if missing words are actually spoken.

    Args:
        audio_path: Path to audio file
        missing_words: List of (word, ref_index, time_range_start, time_range_end)
        settings: Settings instance

    Returns:
        List of WordVerification results
    """
    if not settings.has_google or not missing_words:
        return [
            WordVerification(
                word=word,
                ref_index=ref_idx,
                is_spoken=True,  # Assume spoken
                confidence=0.0,
            )
            for word, ref_idx, _, _ in missing_words
        ]

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.google_api_key)

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        suffix = audio_path.suffix.lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
        }
        mime_type = mime_types.get(suffix, "audio/mpeg")

        # Build word list for prompt
        words_info = "\n".join([
            f"- \"{word}\" (expected around {start:.1f}s - {end:.1f}s)"
            for word, _, start, end in missing_words[:10]  # Limit to 10
        ])

        prompt = f"""Listen to this Portuguese audio and verify if these words are spoken:

{words_info}

For each word, tell me:
1. Is the word actually spoken in the audio?
2. If yes, what are the exact start and end times?

Return JSON:
{{
    "verifications": [
        {{
            "word": "NADA",
            "is_spoken": true,
            "start_time": 0.5,
            "end_time": 0.9,
            "confidence": 0.9
        }},
        {{
            "word": "ALGUMA",
            "is_spoken": false,
            "confidence": 0.85
        }}
    ]
}}

Listen carefully to each time range. Be precise with timestamps."""

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

        results = []
        verifications = data.get("verifications", [])

        for word, ref_idx, _, _ in missing_words:
            # Find matching verification
            found = None
            for v in verifications:
                if v.get("word", "").upper() == word.upper():
                    found = v
                    break

            if found:
                results.append(WordVerification(
                    word=word,
                    ref_index=ref_idx,
                    is_spoken=found.get("is_spoken", True),
                    start_time=found.get("start_time"),
                    end_time=found.get("end_time"),
                    confidence=found.get("confidence", 0.8),
                ))
            else:
                results.append(WordVerification(
                    word=word,
                    ref_index=ref_idx,
                    is_spoken=True,  # Assume spoken if not in response
                    confidence=0.0,
                ))

        return results

    except Exception as e:
        if settings.debug:
            print(f"Gemini word verification failed: {e}")
        return [
            WordVerification(
                word=word,
                ref_index=ref_idx,
                is_spoken=True,
                confidence=0.0,
            )
            for word, ref_idx, _, _ in missing_words
        ]


# =============================================================================
# Robust Alignment Orchestrator
# =============================================================================


def align_transcription_robust(
    transcription: TranscriptionResult,
    reference_text: str,
    audio_path: Path | str,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """
    Multi-phase robust alignment of transcription to reference text.

    Phases:
    1. Analyze text structure (title, body, anchors)
    2. Needleman-Wunsch DP alignment
    3. Gemini verification for title and missing words (if enabled)
    4. Interpolate timestamps for unverified missing words
    5. Flag low-confidence regions for review

    Args:
        transcription: Original transcription result from Whisper
        reference_text: Expected/reference text
        audio_path: Path to audio file
        settings: Settings instance

    Returns:
        New TranscriptionResult with aligned words
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    # Get alignment settings
    alignment_settings = getattr(settings, "alignment", None)
    algorithm = getattr(alignment_settings, "algorithm", "robust") if alignment_settings else "robust"
    verify_title = getattr(alignment_settings, "verify_title", True) if alignment_settings else True
    verify_missing = getattr(alignment_settings, "verify_missing_words", True) if alignment_settings else True
    low_confidence_threshold = getattr(alignment_settings, "low_confidence_threshold", 0.6) if alignment_settings else 0.6

    # Fall back to greedy if requested
    if algorithm == "greedy":
        from wordsync.transcribe import align_transcription_to_text
        return align_transcription_to_text(transcription, reference_text)

    # Phase 1: Analyze text structure
    structure = analyze_text_structure(reference_text)

    if structure.total_word_count == 0:
        return transcription

    # Phase 2: Needleman-Wunsch alignment
    ref_words = structure.all_words
    alignment = needleman_wunsch_align(ref_words, transcription.words)

    # Build aligned words list
    aligned_words: list[AlignedWord] = []
    words_matched = 0
    words_deleted = 0
    words_inserted = 0

    for ref_idx, trans_idx, align_type in alignment:
        if align_type == AlignmentType.MATCH and trans_idx is not None and ref_idx is not None:
            trans_word = transcription.words[trans_idx]
            ref_word = ref_words[ref_idx]
            similarity = word_similarity(ref_word, trans_word.word)

            aligned_words.append(AlignedWord(
                word=ref_word,  # Use reference word (preserves punctuation)
                start=trans_word.start,
                end=trans_word.end,
                confidence=trans_word.confidence * similarity,
                line_break_after=trans_word.line_break_after,
                is_title=trans_word.is_title,
                alignment_type=AlignmentType.MATCH,
                similarity_score=similarity,
                verification_status="matched",
                ref_index=ref_idx,
            ))
            words_matched += 1

        elif align_type == AlignmentType.DELETION and ref_idx is not None:
            # Word in reference but not in transcription (Whisper missed it)
            ref_word = ref_words[ref_idx]
            aligned_words.append(AlignedWord(
                word=ref_word,
                start=0.0,  # Will be interpolated
                end=0.0,
                confidence=0.3,
                alignment_type=AlignmentType.DELETION,
                similarity_score=0.0,
                verification_status="needs_verification",
                ref_index=ref_idx,
            ))
            words_deleted += 1

        elif align_type == AlignmentType.INSERTION and trans_idx is not None:
            # Word in transcription but not in reference (Whisper hallucination)
            # Skip these - don't include hallucinated words
            words_inserted += 1

    # Phase 3: Gemini verification (if enabled)
    title_verified = False
    words_verified = 0

    if settings.has_google:
        # Verify title if requested and there are title words
        if verify_title and structure.title_words:
            title_text = " ".join(structure.title_words)
            title_verification = verify_title_with_gemini(audio_path, title_text, settings)

            if title_verification.confidence > 0.7:
                title_verified = True

                # Update title word timestamps if verified
                if title_verification.is_spoken and title_verification.start_time is not None:
                    title_len = len(structure.title_words)
                    title_duration = (title_verification.end_time or 0.5) - title_verification.start_time
                    time_per_word = title_duration / title_len

                    for i, word in enumerate(aligned_words[:title_len]):
                        if word.alignment_type == AlignmentType.DELETION:
                            word.start = title_verification.start_time + i * time_per_word
                            word.end = word.start + time_per_word
                            word.verification_status = "verified"
                            word.confidence = title_verification.confidence
                            words_verified += 1

        # Verify missing words if requested
        if verify_missing:
            missing_words = []
            for i, word in enumerate(aligned_words):
                if word.alignment_type == AlignmentType.DELETION and word.verification_status != "verified":
                    # Find time context
                    time_before = 0.0
                    time_after = transcription.duration

                    for j in range(i - 1, -1, -1):
                        if aligned_words[j].end > 0:
                            time_before = aligned_words[j].end
                            break

                    for j in range(i + 1, len(aligned_words)):
                        if aligned_words[j].start > 0:
                            time_after = aligned_words[j].start
                            break

                    missing_words.append((word.word, word.ref_index, time_before, time_after))

            if missing_words:
                verifications = verify_missing_words(audio_path, missing_words, settings)

                for v in verifications:
                    if v.confidence > 0.7 and v.is_spoken and v.start_time is not None:
                        # Find and update the word
                        for word in aligned_words:
                            if word.ref_index == v.ref_index:
                                word.start = v.start_time
                                word.end = v.end_time or (v.start_time + 0.3)
                                word.verification_status = "verified"
                                word.confidence = v.confidence
                                words_verified += 1
                                break

    # Phase 4: Interpolate remaining missing timestamps
    aligned_words = interpolate_missing_timestamps(
        aligned_words,
        transcription.duration,
    )

    # Phase 5: Flag low-confidence words
    words_interpolated = sum(
        1 for w in aligned_words
        if w.verification_status == "interpolated"
    )

    for word in aligned_words:
        if word.confidence < low_confidence_threshold:
            word.verification_status = "needs_review"

    # Calculate overall confidence
    if aligned_words:
        avg_confidence = sum(w.confidence for w in aligned_words) / len(aligned_words)
    else:
        avg_confidence = 0.0

    if settings.debug:
        print(f"Robust alignment: {words_matched} matched, {words_deleted} deleted, "
              f"{words_inserted} inserted, {words_verified} verified, "
              f"{words_interpolated} interpolated")

    # Build result
    return TranscriptionResult(
        audio_file=transcription.audio_file,
        language=transcription.language,
        model=transcription.model,
        duration=transcription.duration,
        words=[Word(
            word=w.word,
            start=w.start,
            end=w.end,
            confidence=w.confidence,
            line_break_after=w.line_break_after,
            is_title=w.is_title,
        ) for w in aligned_words],
        full_text=reference_text,
        provider=transcription.provider,
        raw_response=transcription.raw_response,
    )
