"""
WordSync Configuration Module

Handles configuration loading from:
1. Environment variables (.env file)
2. YAML configuration file (config.yaml)
3. Default values

Uses Pydantic for validation and type safety.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GapThresholds(BaseSettings):
    """Gap classification thresholds in milliseconds."""

    micro_gap_max: int = 50
    short_gap_max: int = 150
    medium_gap_max: int = 400
    natural_pause_max: int = 600
    sentence_boundary_min: int = 600


class PortugueseRules(BaseSettings):
    """Portuguese-specific liaison and pause rules."""

    article_noun_liaison: bool = True
    preposition_liaison: bool = True
    preserve_educational_pauses: bool = True
    articles: list[str] = Field(
        default_factory=lambda: ["o", "a", "os", "as", "um", "uma", "uns", "umas"]
    )
    prepositions: list[str] = Field(
        default_factory=lambda: [
            "de", "da", "do", "das", "dos",
            "em", "na", "no", "nas", "nos",
            "para", "pra", "pro"
        ]
    )


class TranscriptionSettings(BaseSettings):
    """Transcription (Whisper) settings."""

    # Provider selection: "openai" | "local" | "whisperx"
    provider: str = "local"

    # OpenAI Whisper API settings
    whisper_model: str = "whisper-1"
    response_format: str = "verbose_json"
    temperature: float = 0.0

    # Local whisper-timestamped settings
    local_model: str = "base"

    # WhisperX settings (recommended for best word-level accuracy)
    whisperx_model: str = "base"  # "tiny", "base", "small", "medium", "large-v2"
    whisperx_device: str = "cpu"  # "cpu" or "cuda"


class ProsodySettings(BaseSettings):
    """Prosody analysis (Gemini) settings."""

    enabled: bool = True
    model: str = "gemini-2.5-flash"
    confidence_threshold: float = 0.7
    features: list[str] = Field(
        default_factory=lambda: ["breath_pauses", "pitch_resets", "emphasis", "syllable_stress"]
    )


class ValidationSettings(BaseSettings):
    """Validation (GPT-4o) settings."""

    enabled: bool = False  # Disabled by default - focus on better timestamps instead
    model: str = "gpt-audio"  # Updated model name
    confidence_threshold: float = 0.85
    use_median_timestamps: bool = True


class QualitySettings(BaseSettings):
    """Quality metrics thresholds."""

    min_confidence: float = 0.90
    review_threshold: float = 0.80
    max_timing_deviation: int = 100
    min_word_duration_ms: int = 200  # Minimum word duration for visible highlighting
    min_gap_between_words_ms: int = 10  # Minimum gap between consecutive words


class AlignmentSettings(BaseSettings):
    """Alignment algorithm settings."""

    algorithm: str = "robust"  # "greedy" | "robust"
    verify_title: bool = True  # Use Gemini to verify title is spoken
    verify_missing_words: bool = True  # Use Gemini to verify missing words
    low_confidence_threshold: float = 0.6  # Mark words below this for review


class OutputSettings(BaseSettings):
    """Output generation settings."""

    include_json: bool = True
    include_metrics: bool = True
    embed_audio: bool = False
    bundle_assets: bool = False


class TemplateSettings(BaseSettings):
    """Template configuration."""

    page_template: str = "page.html.jinja2"
    styles: str = "styles.css"
    player: str = "player.js"


class Settings(BaseSettings):
    """
    Main settings class for WordSync Engine.

    Loads configuration from:
    1. Environment variables (highest priority)
    2. .env file
    3. config.yaml
    4. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="WORDSYNC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # General settings
    project_name: str = "Letrix Leituras Guiadas"
    language: str = "pt"
    debug: bool = False
    use_local_whisper: bool = False

    # Component settings
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    prosody: ProsodySettings = Field(default_factory=ProsodySettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    gap_thresholds: GapThresholds = Field(default_factory=GapThresholds)
    portuguese_rules: PortugueseRules = Field(default_factory=PortugueseRules)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    templates: TemplateSettings = Field(default_factory=TemplateSettings)

    # Page configurations (loaded from config.yaml)
    pages: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    content_dir: Path | None = None
    output_dir: Path | None = None
    templates_dir: Path | None = None

    @field_validator("content_dir", "output_dir", "templates_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path | None, info: Any) -> Path | None:
        if v is None:
            return None
        return Path(v).resolve()

    def __init__(self, **kwargs: Any) -> None:
        # Load from config.yaml if not provided in kwargs
        config_yaml = kwargs.pop("_config_yaml", None)
        if config_yaml is None:
            config_yaml = load_yaml_config()

        # Merge yaml config with kwargs (kwargs take precedence)
        merged = merge_configs(config_yaml, kwargs)
        super().__init__(**merged)

        # Set default paths relative to project root
        if self.content_dir is None:
            self.content_dir = self.project_root / "content"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.templates_dir is None:
            self.templates_dir = self.project_root / "templates"

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return bool(self.google_api_key)

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    def get_available_providers(self) -> list[str]:
        """Get list of configured API providers."""
        providers = []
        if self.has_openai:
            providers.append("openai")
        if self.has_google:
            providers.append("google")
        if self.has_anthropic:
            providers.append("anthropic")
        return providers

    def validate_required_keys(self) -> list[str]:
        """Validate required API keys are present. Returns list of missing keys."""
        missing = []
        if not self.has_openai:
            missing.append("OPENAI_API_KEY")
        return missing


def load_yaml_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, searches in current directory.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return flatten_yaml_config(data)


def flatten_yaml_config(data: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested YAML structure to match Settings model.

    Converts:
        project:
          name: "X"
          language: "pt"
        transcription:
          whisper_model: "whisper-1"

    To:
        project_name: "X"
        language: "pt"
        transcription: TranscriptionSettings(whisper_model="whisper-1")
    """
    result: dict[str, Any] = {}

    # Project settings
    if "project" in data:
        project = data["project"]
        if "name" in project:
            result["project_name"] = project["name"]
        if "language" in project:
            result["language"] = project["language"]

    # Nested settings objects
    nested_keys = [
        "transcription",
        "prosody",
        "validation",
        "quality",
        "alignment",
        "output",
        "templates",
    ]
    for key in nested_keys:
        if key in data:
            result[key] = data[key]

    # Gap classification -> gap_thresholds
    if "gap_classification" in data:
        result["gap_thresholds"] = data["gap_classification"]

    # Portuguese rules
    if "portuguese_rules" in data:
        result["portuguese_rules"] = data["portuguese_rules"]

    # Pages
    if "pages" in data:
        result["pages"] = data["pages"]

    return result


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Override takes precedence over base.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


@lru_cache
def get_settings() -> Settings:
    """
    Get cached Settings instance.

    Call this function to get the global settings object.
    Settings are loaded once and cached for the lifetime of the process.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings from files, clearing the cache.

    Use this when config files have changed.
    """
    get_settings.cache_clear()
    return get_settings()
