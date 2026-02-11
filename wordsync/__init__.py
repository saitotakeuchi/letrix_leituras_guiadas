"""
WordSync Engine - Karaoke Word-Sync for Letrix

A multi-modal LLM-powered engine for generating accurate word-level
timestamps for karaoke-style guided reading applications.

Pipeline:
    1. Whisper API -> Raw word-level timestamps
    2. Gemini 2.5 Flash -> Duration correction (listens to audio)
    3. Gemini 2.5 Flash -> Prosodic analysis (breath pauses, pitch resets)
    4. Gap Classifier -> Decide per-gap: keep natural pause vs fill artifact
    5. Gemini/GPT-4o -> Cross-validate timestamps (Gemini preferred)
    6. Heuristic Validation -> Flag suspicious timestamps (zero cost)
    7. Build -> Generate HTML with embedded timestamps

Usage:
    from wordsync import sync_audio, build_page

    # Generate timestamps
    result = sync_audio("audio.mp3", "text.txt")

    # Build HTML page
    build_page(result, "output/page.html")

CLI:
    wordsync sync audio.mp3 text.txt -o output/
    wordsync build page-001
    wordsync batch
    wordsync preview page-001
    wordsync validate timestamps.json
"""

__version__ = "1.0.0"
__author__ = "MokLabs"

from wordsync.config import Settings, get_settings
from wordsync.transcribe import transcribe_audio, TranscriptionResult
from wordsync.prosody import analyze_prosody, ProsodyResult
from wordsync.classify import classify_gaps, GapClassification
from wordsync.validate import (
    validate_timestamps,
    validate_timestamps_heuristic,
    ValidationResult,
    HeuristicValidationResult,
    TimestampIssue,
    TimestampIssueType,
)
from wordsync.alignment import (
    align_transcription_robust,
    AlignedWord,
    AlignmentType,
    TextStructure,
    needleman_wunsch_align,
)
from wordsync.process import process_sync, SyncResult
from wordsync.build import build_page, build_batch

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Transcription
    "transcribe_audio",
    "TranscriptionResult",
    # Prosody
    "analyze_prosody",
    "ProsodyResult",
    # Classification
    "classify_gaps",
    "GapClassification",
    # Alignment
    "align_transcription_robust",
    "AlignedWord",
    "AlignmentType",
    "TextStructure",
    "needleman_wunsch_align",
    # Validation
    "validate_timestamps",
    "validate_timestamps_heuristic",
    "ValidationResult",
    "HeuristicValidationResult",
    "TimestampIssue",
    "TimestampIssueType",
    # Processing
    "process_sync",
    "SyncResult",
    # Building
    "build_page",
    "build_batch",
]
