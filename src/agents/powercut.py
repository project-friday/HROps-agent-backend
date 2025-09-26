# src/agents/powercut_agent.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import elevenlabs, openai, silero, soniox

from src.config.loader import get_cfg, render

# ---- Import powercut tools ----
from src.tools.powercut_agent import (
    get_area_mapping,
    get_customer_details,
    get_outage_details,
    schedule_service,
)

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


logger = logging.getLogger("hr-eve-powercut-agent")
logger.setLevel(logging.INFO)


class PowercutAgent(Agent):
    """
    Powercut assistant:
    - Identifies customers (via UAC number or manual address)
    - Retrieves outage details using area code
    - Escalates to service scheduling if no outage detected
    """

    def __init__(self, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()
        prompt = load_prompt("src/prompts/powercut.txt")
        self.prompt = render(prompt, self.cfg)
        self.eleven_tts = elevenlabs.TTS(
            voice_id="H8bdWZHK2OgZwTN7ponr",
            model="eleven_turbo_v2_5",
        )
        self.openai_tts = openai.TTS(voice="shimmer")
        self._user_language = "te"

        super().__init__(
            instructions=self.prompt,
            stt=soniox.STT(params=soniox.STTOptions(language_hints=["en", "hi", "te"])),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            tts=self.openai_tts,
            tools=[
                get_customer_details,
                get_outage_details,
                get_area_mapping,
                schedule_service,
            ],
        )

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Custom STT node that intercepts only finalized transcripts."""
        print("🎤 Starting custom STT node...")
        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node is available"

        wrapped_stt = activity.stt
        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError(
                    f"The STT ({activity.stt.label}) does not support streaming, add a VAD"
                )
            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        conn_options = activity.session.conn_options.stt_conn_options
        async with wrapped_stt.stream(conn_options=conn_options) as stream:

            @utils.log_exceptions()
            async def _forward_input() -> None:
                async for frame in audio:
                    stream.push_frame(frame)

            print("🎤 Launched audio forwarding task")
            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:

                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        if event.alternatives:
                            self._user_language = event.alternatives[0].language

                    # Always yield back into agent pipeline
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Dynamically choose TTS engine based on last STT language.
        Telugu -> OpenAI shimmer; others -> ElevenLabs.
        """

        # ---- Select TTS Engine ----
        if self._user_language == "te":
            chosen_tts = self.openai_tts
        else:
            chosen_tts = self.eleven_tts

        # ---- Handle non-streaming TTS engines ----
        wrapped_tts = chosen_tts
        if not chosen_tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=chosen_tts,
                sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
                    retain_format=True
                ),
            )

        conn_options = self.session.conn_options.tts_conn_options
        async with wrapped_tts.stream(conn_options=conn_options) as stream:

            async def _forward_input() -> None:
                async for chunk in text:
                    stream.push_text(chunk)
                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def on_enter(self):
        cfg = get_cfg()
        await self.session.say(cfg["greeting"])
