# src/utils/audio_recorder.py
"""
Audio recording utility for capturing user audio from LiveKit rooms.
Captures audio frames from the STT pipeline and saves to WAV files.
"""
from __future__ import annotations

import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from livekit import rtc

logger = logging.getLogger("audio-recorder")


class UserAudioRecorder:
    """
    Records user audio by capturing frames from the STT pipeline.

    Usage:
        recorder = UserAudioRecorder(output_dir="recordings")
        recorder.start_session(session_id="room_123")

        # In stt_node, for each frame:
        recorder.add_frame(frame)

        # When session ends:
        filepath = recorder.save()
    """

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate: Optional[int] = None
        self.num_channels: Optional[int] = None
        self._frames: list[bytes] = []
        self._is_recording = False
        self._session_id: Optional[str] = None

    def start_session(self, session_id: Optional[str] = None) -> None:
        """Start a new recording session."""
        if self._is_recording:
            logger.warning("Recording already in progress")
            return

        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._frames = []
        self.sample_rate = None
        self.num_channels = None
        self._is_recording = True

        logger.info("Started audio recording session: %s", self._session_id)

    def add_frame(self, frame: rtc.AudioFrame) -> None:
        """Add an audio frame to the recording buffer."""
        if not self._is_recording:
            return

        # Auto-detect sample rate and channels from first frame
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
            self.num_channels = frame.num_channels
            logger.info(
                "Audio format: %dHz, %d channel(s)",
                frame.sample_rate,
                frame.num_channels,
            )

        self._frames.append(bytes(frame.data))

    def save(self) -> Optional[str]:
        """Stop recording and save the audio to a WAV file."""
        if not self._is_recording:
            return None

        self._is_recording = False

        if not self._frames:
            logger.warning("No audio frames captured")
            return None

        filepath = self._save_wav()
        return filepath

    def _save_wav(self) -> str:
        """Save captured frames to a WAV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_audio_{self._session_id}_{timestamp}.wav"
        filepath = self.output_dir / filename

        audio_data = b"".join(self._frames)
        sample_rate = self.sample_rate or 48000
        num_channels = self.num_channels or 1

        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        duration = len(audio_data) / (sample_rate * num_channels * 2)
        logger.info("Saved %.2fs of audio to %s", duration, filepath)

        return str(filepath)

    @property
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._is_recording
