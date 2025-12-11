"""
STT Factory Module

Provides factory functions for creating Speech-to-Text (STT) instances
based on configuration. Supports multiple providers: Soniox, Deepgram, and Speechmatics.

Usage:
    from src.utils.stt_factory import create_stt
    
    stt_config = {
        "provider": "soniox",
        "language_hints": ["en", "ar"]
    }
    stt_instance = create_stt(stt_config)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import stt
from livekit.plugins import deepgram, silero, soniox

try:
    from livekit.plugins import speechmatics
except ImportError:
    speechmatics = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_stt(stt_config: dict[str, Any]) -> stt.STT:
    """
    Create an STT instance based on the provided configuration.
    
    Args:
        stt_config: Configuration dictionary containing provider and settings.
                   Must include 'provider' key with value: 'soniox', 'deepgram', or 'speechmatics'
    
    Returns:
        Configured STT instance ready for use with LiveKit agents
    
    Raises:
        ValueError: If provider is unknown or configuration is invalid
        RuntimeError: If required API keys are missing
    
    Example:
        >>> config = {
        ...     "provider": "soniox",
        ...     "language_hints": ["en", "ar"],
        ...     "vad_enabled": True
        ... }
        >>> stt_instance = create_stt(config)
    """
    provider = stt_config.get("provider", "soniox").lower()
    
    logger.info(f"Creating STT instance for provider: {provider}")
    
    # Load VAD (Voice Activity Detection) if enabled
    vad = None
    if stt_config.get("vad_enabled", True):
        min_speech_duration = stt_config.get("min_speech_duration", 0.1)
        vad = silero.VAD.load(min_speech_duration=min_speech_duration)
        logger.debug(f"VAD loaded with min_speech_duration={min_speech_duration}")
    
    # Route to appropriate provider
    if provider == "soniox":
        return _create_soniox_stt(stt_config, vad)
    
    elif provider == "deepgram":
        return _create_deepgram_stt(stt_config)
    
    elif provider == "speechmatics":
        return _create_speechmatics_stt(stt_config)
    
    else:
        available_providers = ["soniox", "deepgram", "speechmatics"]
        raise ValueError(
            f"Unknown STT provider: '{provider}'. "
            f"Available providers: {', '.join(available_providers)}"
        )


def _create_soniox_stt(config: dict[str, Any], vad: silero.VAD | None) -> soniox.STT:
    """
    Create a Soniox STT instance.
    
    Soniox provides multilingual speech recognition with automatic language detection.
    Ideal for conversations that may switch between languages.
    
    Args:
        config: Configuration dictionary with Soniox-specific settings
        vad: Optional Voice Activity Detection instance
    
    Returns:
        Configured Soniox STT instance
    """
    language_hints = config.get("language_hints", ["en", "ar"])
    
    logger.info(f"Initializing Soniox STT with language hints: {language_hints}")
    
    return soniox.STT(
        params=soniox.STTOptions(language_hints=language_hints),
        vad=vad,
    )


def _create_deepgram_stt(config: dict[str, Any]) -> deepgram.STT:
    """
    Create a Deepgram STT instance.
    
    Deepgram provides fast, accurate speech recognition with smart formatting.
    Recommended for English-primary conversations.
    
    Args:
        config: Configuration dictionary with Deepgram-specific settings
    
    Returns:
        Configured Deepgram STT instance
    """
    model = config.get("model", "nova-3")
    language = config.get("language_code", "en")
    endpointing_ms = config.get("endpointing_ms", 100)
    
    logger.info(
        f"Initializing Deepgram STT with model={model}, "
        f"language={language}, endpointing={endpointing_ms}ms"
    )
    
    return deepgram.STT(
        model=model,
        language=language,
        endpointing_ms=endpointing_ms,
        filler_words=config.get("filler_words", True),
        smart_format=config.get("smart_format", True),
    )


def _create_speechmatics_stt(config: dict[str, Any]) -> speechmatics.STT:
    """
    Create a Speechmatics STT instance using LiveKit plugin.
    
    Speechmatics provides enterprise-grade speech recognition.
    
    Args:
        config: Configuration dictionary with Speechmatics-specific settings
    
    Returns:
        Configured Speechmatics STT instance
    
    Raises:
        RuntimeError: If SPEECHMATICS_API_KEY is not found in environment or plugin not available
    """
    if speechmatics is None:
        raise RuntimeError(
            "Speechmatics plugin not available. "
            "Install with: pip install livekit-plugins-speechmatics"
        )
    
    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SPEECHMATICS_API_KEY not found in environment. "
            "Please add it to your .env file:\n"
            "SPEECHMATICS_API_KEY=your_api_key_here"
        )
    
    language = config.get("language", "en")
    
    logger.info(f"Initializing Speechmatics STT with language={language}")
    
    return speechmatics.STT(
        api_key=api_key,
        language=language,
    )
