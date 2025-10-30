# src/agents/survey_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from livekit import rtc
from livekit.agents import llm, stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings

# Plugins
from livekit.plugins import elevenlabs, silero, soniox

from src.config.loader import get_cfg
from src.tools.survey_tools import record_feedback

logger = logging.getLogger("dubai-mall-survey-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


EVE_SURVEY_PROMPT = load_prompt("src/prompts/survey.txt")


class SurveyAgent(Agent):
    """
    Dubai Mall Survey Agent:
    - Conducts customer satisfaction surveys
    - Handles multilingual (English/Arabic) interactions
    - Switches TTS dynamically based on detected speech language
    """

    def __init__(self, cfg: dict, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()

        # --- Voice Setup (Arabic & English) ---
        tts_cfg = cfg.get("tts", {})
        voices = tts_cfg.get("voices", {})

        arabic_voice = voices.get("arabic") or self.cfg.get("voices", {}).get("arabic")
        english_voice = voices.get("english") or self.cfg.get("voices", {}).get(
            "english"
        )

        if not arabic_voice or not english_voice:
            raise RuntimeError("❌ Missing one or both voices for SurveyAgent")

        self.arabic_tts = elevenlabs.TTS(
            voice_id=arabic_voice,
            model=tts_cfg.get("model", "eleven_turbo_v2_5"),
        )
        self.english_tts = elevenlabs.TTS(
            voice_id=english_voice,
            model=tts_cfg.get("model", "eleven_turbo_v2_5"),
        )

        super().__init__(
            instructions=EVE_SURVEY_PROMPT,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            tools=[record_feedback],
            chat_ctx=chat_ctx,
        )

        # --- Tool actions ---
        self.actions = {
            "Record Feedback": record_feedback,
            # "Submit Survey Response": submit_survey_response,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        self.visible_tools = {record_feedback}

    async def _send_websocket_message(self, action: str, result: Dict[str, Any] = None):
        """Send WebSocket update when tool actions are triggered."""
        message = {"action": action, "result": result or {}}
        try:
            await self.room.local_participant.send_text(
                json.dumps(message), topic="lk.transcription"
            )
            logger.info("Sent WebSocket message: %s", message)
        except Exception as e:
            logger.error("Failed to send WebSocket message: %s", e)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        """Capture LLM outputs and trigger tool calls."""
        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node available"

        buffer = []
        async with activity.llm.chat(chat_ctx=chat_ctx, tools=tools) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    buffer.append(chunk)
                elif isinstance(chunk, llm.ChatChunk) and chunk.delta:
                    if chunk.delta.content:
                        buffer.append(chunk.delta.content)
                yield chunk

        self.last_llm_response = "".join(buffer).strip()

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Detect language (en/ar) for dynamic TTS switching."""
        print("🎤 Starting STT node for survey agent...")
        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node available"

        wrapped_stt = activity.stt
        if not wrapped_stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError("STT requires a VAD for non-streaming mode")
            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        async with wrapped_stt.stream() as stream:

            @utils.log_exceptions()
            async def _forward_input():
                async for frame in audio:
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    if (
                        event.type == SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                    ):
                        self._user_language = event.alternatives[0].language
                        print("🌐 Detected survey language:", self._user_language)
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """Switch TTS voice dynamically based on detected language."""
        lang = getattr(self, "_user_language", "en")

        if not lang.startswith("en"):
            print("🗣️ Using Arabic survey TTS")
            chosen_tts = self.arabic_tts
        else:
            print("🗣️ Using English survey TTS")
            chosen_tts = self.english_tts

        wrapped_tts = chosen_tts
        if not chosen_tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=chosen_tts,
                sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
                    retain_format=True
                ),
            )

        async with wrapped_tts.stream() as stream:

            async def _forward_input():
                async for chunk in text:
                    stream.push_text(chunk)
                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await asyncio.wait([forward_task])

    async def on_enter(self):
        """Speak greeting when the survey starts."""
        cfg = get_cfg()
        greeting = cfg["agents"]["SurveyAgent"].get("greeting")
        if greeting:
            await self.session.say(greeting)
