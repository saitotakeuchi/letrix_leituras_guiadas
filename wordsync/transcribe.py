"""
WordSync Transcription Module

Handles audio transcription with word-level timestamps using:
1. OpenAI Whisper API (primary, cloud-based)
2. Local whisper-timestamped (fallback, offline)

The transcription result provides the foundation for the word-sync pipeline.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wordsync.config import Settings, get_settings


@dataclass
class Word:
    """A single word with timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0  # Transcription confidence (0-1)
    line_break_after: bool = False  # Whether there's a line break after this word
    is_title: bool = False  # Whether this word is part of the title

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
        }
        if self.line_break_after:
            result["line_break_after"] = True
        if self.is_title:
            result["is_title"] = True
        return result


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    audio_file: str
    language: str
    model: str
    duration: float  # Total audio duration in seconds
    words: list[Word]
    full_text: str
    provider: str  # "openai" or "local"
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Number of words transcribed."""
        return len(self.words)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all words."""
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "audio_file": self.audio_file,
            "language": self.language,
            "model": self.model,
            "duration": round(self.duration, 3),
            "word_count": self.word_count,
            "average_confidence": round(self.average_confidence, 3),
            "full_text": self.full_text,
            "provider": self.provider,
            "words": [w.to_dict() for w in self.words],
        }


def transcribe_audio(
    audio_path: str | Path,
    language: str | None = None,
    use_local: bool | None = None,
    provider: str | None = None,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """
    Transcribe audio file to get word-level timestamps.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        language: Language code (default: from settings, typically "pt")
        use_local: Force local transcription (default: from settings)
        provider: Transcription provider ("openai", "local", "whisperx")
                  If None, uses settings.transcription.provider
        settings: Settings instance (default: global settings)

    Returns:
        TranscriptionResult with word-level timestamps

    Raises:
        ValueError: If audio file not found or API key missing
        RuntimeError: If transcription fails
    """
    settings = settings or get_settings()
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")

    language = language or settings.language

    # Determine provider
    if provider is None:
        provider = getattr(settings.transcription, "provider", None)

    # Legacy support: use_local flag overrides provider
    if use_local is None:
        use_local = settings.use_local_whisper

    if use_local and provider is None:
        provider = "local"
    elif provider is None:
        provider = "openai"

    # Route to appropriate transcription method
    if provider == "whisperx":
        return _transcribe_whisperx(audio_path, language, settings)
    elif provider == "local":
        return _transcribe_local(audio_path, language, settings)
    else:
        return _transcribe_openai(audio_path, language, settings)


def _transcribe_openai(
    audio_path: Path,
    language: str,
    settings: Settings,
) -> TranscriptionResult:
    """
    Transcribe using OpenAI Whisper API.

    Uses the verbose_json format to get word-level timestamps.
    """
    from openai import OpenAI

    if not settings.has_openai:
        raise ValueError(
            "OpenAI API key not configured. "
            "Set OPENAI_API_KEY in .env or use --local flag."
        )

    client = OpenAI(api_key=settings.openai_api_key)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=settings.transcription.whisper_model,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            temperature=settings.transcription.temperature,
        )

    # Parse response
    response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    words = []
    for word_data in response_dict.get("words", []):
        words.append(Word(
            word=word_data["word"].strip(),
            start=word_data["start"],
            end=word_data["end"],
            confidence=1.0,  # Whisper API doesn't provide per-word confidence
        ))

    return TranscriptionResult(
        audio_file=audio_path.name,
        language=language,
        model=settings.transcription.whisper_model,
        duration=response_dict.get("duration", 0.0),
        words=words,
        full_text=response_dict.get("text", "").strip(),
        provider="openai",
        raw_response=response_dict,
    )


def _transcribe_local(
    audio_path: Path,
    language: str,
    settings: Settings,
) -> TranscriptionResult:
    """
    Transcribe using local whisper-timestamped.

    Requires: pip install wordsync[local]
    """
    try:
        import whisper_timestamped as whisper
    except ImportError:
        raise RuntimeError(
            "Local Whisper not installed. "
            "Install with: pip install wordsync[local]"
        )

    model_name = settings.transcription.local_model
    model = whisper.load_model(model_name)

    result = whisper.transcribe(
        model,
        str(audio_path),
        language=language,
    )

    words = []
    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            confidence = word_data.get("confidence", 1.0)
            words.append(Word(
                word=word_data["text"].strip(),
                start=word_data["start"],
                end=word_data["end"],
                confidence=confidence,
            ))

    # whisper_timestamped doesn't return a duration field;
    # calculate from last segment or last word
    duration = result.get("duration", 0.0)
    if not duration:
        segments = result.get("segments", [])
        if segments:
            duration = segments[-1].get("end", 0.0)
        elif words:
            duration = words[-1].end

    return TranscriptionResult(
        audio_file=audio_path.name,
        language=language,
        model=f"local:{model_name}",
        duration=duration,
        words=words,
        full_text=result.get("text", "").strip(),
        provider="local",
        raw_response=result,
    )


def _transcribe_whisperx(
    audio_path: Path,
    language: str,
    settings: Settings,
) -> TranscriptionResult:
    """
    Transcribe using WhisperX for accurate word-level timestamps.

    WhisperX uses forced phoneme alignment (wav2vec 2.0) for precision
    that standard Whisper cannot achieve. This is the recommended method
    for karaoke applications where timing accuracy is critical.

    Requires: pip install whisperx

    Benefits over standard Whisper:
    - Phoneme-level alignment using wav2vec 2.0
    - No more integer-rounded timestamps
    - No more 0.0 start times for initial words
    - Proper handling of word boundaries
    """
    import torch

    # WORKAROUND: PyTorch 2.6+ requires weights_only=True by default,
    # but pyannote models use pickle which isn't compatible.
    # Patch torch.load to force weights_only=False for WhisperX.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    try:
        import whisperx
    except ImportError:
        torch.load = _original_torch_load  # Restore on error
        raise RuntimeError(
            "WhisperX not installed. "
            "Install with: pip install whisperx"
        )

    model_name = settings.transcription.whisperx_model
    device = settings.transcription.whisperx_device

    # Determine compute type based on device
    # float16 is faster on GPU, but CPU requires int8 or float32
    compute_type = "float16" if device == "cuda" else "int8"

    try:
        # Load Whisper model for initial transcription
        model = whisperx.load_model(
            model_name,
            device=device,
            language=language,
            compute_type=compute_type,
        )

        # Load and transcribe audio
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio)

        # Load alignment model and align for precise word timestamps
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )

        # Extract words from aligned result
        words = []
        full_text_parts = []

        for segment in result.get("segments", []):
            segment_text = segment.get("text", "")
            if segment_text:
                full_text_parts.append(segment_text.strip())

            for word_data in segment.get("words", []):
                # WhisperX uses 'word' key, and provides 'score' for confidence
                word_text = word_data.get("word", "").strip()
                if not word_text:
                    continue

                # Handle cases where alignment failed (missing start/end)
                start = word_data.get("start")
                end = word_data.get("end")

                if start is None or end is None:
                    # Skip words that couldn't be aligned
                    continue

                words.append(Word(
                    word=word_text,
                    start=start,
                    end=end,
                    confidence=word_data.get("score", 0.9),
                ))

        # Calculate duration from last word or audio length
        duration = 0.0
        if words:
            duration = words[-1].end
        elif result.get("segments"):
            last_segment = result["segments"][-1]
            duration = last_segment.get("end", 0.0)

        return TranscriptionResult(
            audio_file=audio_path.name,
            language=language,
            model=f"whisperx:{model_name}",
            duration=duration,
            words=words,
            full_text=" ".join(full_text_parts).strip(),
            provider="whisperx",
            raw_response=result,
        )

    finally:
        # Restore original torch.load
        torch.load = _original_torch_load


def transcribe_with_fallback(
    audio_path: str | Path,
    language: str | None = None,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """
    Transcribe with automatic fallback chain.

    Fallback order:
    1. WhisperX (if configured as provider) - best word-level accuracy
    2. OpenAI Whisper API (if API key available)
    3. Local whisper-timestamped

    Args:
        audio_path: Path to audio file
        language: Language code (default: from settings)
        settings: Settings instance

    Returns:
        TranscriptionResult from whichever method succeeds
    """
    settings = settings or get_settings()
    language = language or settings.language
    audio_path = Path(audio_path)

    provider = getattr(settings.transcription, "provider", None)

    # Try WhisperX first if configured
    if provider == "whisperx":
        try:
            return _transcribe_whisperx(audio_path, language, settings)
        except Exception as e:
            if settings.debug:
                print(f"WhisperX transcription failed: {e}, falling back to OpenAI")

    # Try OpenAI if available
    if settings.has_openai:
        try:
            return _transcribe_openai(audio_path, language, settings)
        except Exception as e:
            if settings.debug:
                print(f"OpenAI transcription failed: {e}, falling back to local")

    # Fall back to local
    try:
        return _transcribe_local(audio_path, language, settings)
    except ImportError:
        raise RuntimeError(
            "No transcription method available. "
            "Either set OPENAI_API_KEY, install WhisperX, or install local Whisper: pip install wordsync[local]"
        )


def get_audio_mime_type(audio_path: str | Path) -> str:
    """
    Get MIME type for audio file based on extension.

    Args:
        audio_path: Path to audio file

    Returns:
        MIME type string (e.g., "audio/mpeg", "audio/wav")
    """
    audio_path = Path(audio_path)
    extension = audio_path.suffix.lower()

    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }

    return mime_types.get(extension, "audio/mpeg")


