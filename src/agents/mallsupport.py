# src/agents/mallsupport_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from livekit import rtc
from livekit.agents import llm, stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)
from livekit.agents.voice.io import TimedString  # ← Missing one

# Plugins
from livekit.plugins import azure, cartesia, elevenlabs, openai, silero, soniox

from src.config.loader import get_cfg
from src.tools.handover import handover_to_survey

# ---- Import only the knowledge base tool ----
from src.tools.mallsupport_tools import (
    create_ticket,
    feedback_sms_tool,
    query_knowledge_base,
)
from src.utils.audio_recorder import UserAudioRecorder
from src.utils.transcription_utils import is_wrong_script, repair_text

logger = logging.getLogger("dubai-mall-support-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


EVE_MALL_PROMPT = load_prompt("src/prompts/mallsupport.txt")


class MallSupportAgent(Agent):
    """
    Dubai Mall Customer Support Agent:
    - Handles queries from customers
    - Single tool: query_knowledge_base
    - RAG-based answers for knowledge base
    - Keeps responses concise and polite
    """

    def __init__(self, cfg: dict, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()
        # ---------- TTS setup (multi-language) ----------
        tts_cfg = cfg.get("tts", {})
        voices = tts_cfg.get("voices", {})
        providers = tts_cfg.get("providers", {})

        # ---- English (ElevenLabs) ----
        english_voice = voices.get("english")
        self.english_tts = None
        if english_voice and providers.get("english", "elevenlabs") == "elevenlabs":
            self.english_tts = elevenlabs.TTS(
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_turbo_v2_5",
            )
        else:
            raise RuntimeError("❌ No English ElevenLabs voice configured")

        # # ---- English (Cartesia) ----
        # english_voice = voices.get("english")
        # self.english_tts = None
        # if english_voice and providers.get("english", "cartesia") == "cartesia":
        #     self.english_tts = cartesia.TTS(
        #         model=tts_cfg.get("model", "sonic-3"),
        #         voice=english_voice,
        #         language="en",
        #         speed=0.8,
        #     )

        # ---- Arabic (Cartesia) ----
        arabic_voice = voices.get("arabic")
        if arabic_voice and providers.get("arabic", "cartesia") == "cartesia":
            self.arabic_tts = cartesia.TTS(
                model=tts_cfg.get("model", "sonic-3"),
                voice=arabic_voice,
                language="ar",
                volume=2,
            )
        else:
            raise RuntimeError("❌ No Arabic Cartesia voice configured")

        # # ---- Mandarin (Inworld) ----
        # mandarin_voice = voices.get("mandarin")
        # self.mandarin_tts = None
        # if mandarin_voice:
        #     if providers.get("mandarin") == "inworld":
        #         from livekit.plugins import inworld
        #         self.mandarin_tts = inworld.TTS(voice=mandarin_voice)

        # self.azure_tts_en = azure.TTS(voice="en-US-JennyNeural")
        super().__init__(
            instructions=EVE_MALL_PROMPT,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            tts=self.english_tts,
            # tts=elevenlabs.TTS(
            #     voice_id="H8bdWZHK2OgZwTN7ponr",
            #     model="eleven_turbo_v2_5",
            # ),
            tools=[
                query_knowledge_base,
                create_ticket,
                handover_to_survey,
                feedback_sms_tool,
            ],
            chat_ctx=chat_ctx,
        )

        self.actions = {
            "Query Knowledge Base": query_knowledge_base,
            "Creating ticket": create_ticket,
            "Handover to Survey": handover_to_survey,
            "Send Feedback SMS": feedback_sms_tool,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optional: keys to filter from tool results before sending
        self.tool_result_filters = {
            query_knowledge_base: ["internal_id", "metadata"],
            create_ticket: ["ticket_id"],
            feedback_sms_tool: ["phone_number"],
        }

        self.tool_cards = {
            query_knowledge_base: "knowledge_base_result",
            create_ticket: "ticket_creation_result",
            handover_to_survey: "survey_handover",
            feedback_sms_tool: "feedback_sms_sent",
        }

        self.visible_tools = {create_ticket, feedback_sms_tool}
        self.manual_language = None  # tracks manually switched language

        self.audio_recorder = UserAudioRecorder(output_dir="recordings/user_audio")

    async def _send_websocket_message(
        self, action: str, result: Dict[str, Any] = None, tool_func=None
    ):
        """Send WebSocket message with action, filtered result, and card_name."""
        message = {"action": action}

        if result is not None:
            if tool_func in self.tool_result_filters:
                for key in self.tool_result_filters[tool_func]:
                    result.pop(key, None)

            message["result"] = result
            message["card_name"] = self.tool_cards.get(tool_func, "generic")

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
        """Custom LLM node that captures full response text and executes tools."""

        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM)

        tool_choice = model_settings.tool_choice if model_settings else llm.NOT_GIVEN
        activity_llm = activity.llm
        conn_options = activity.session.conn_options.llm_conn_options

        buffer: list[str] = []
        pending_tools: list[tuple[str, callable, dict]] = []

        async with activity_llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            conn_options=conn_options,
        ) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    buffer.append(chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Invalid JSON for %s: %s", tool_name, tool_args
                                    )
                                    tool_args = {}

                            tool_function = None
                            action_name = None
                            for name, func in self.actions.items():
                                if func.__name__ == tool_name:
                                    tool_function = func
                                    action_name = name
                                    break

                            if tool_function and action_name:
                                await self._send_websocket_message(action_name)
                                pending_tools.append(
                                    (action_name, tool_function, tool_args)
                                )

                yield chunk

        # Capture final LLM response
        self.last_llm_response = "".join(buffer).strip()

        # Execute queued tools and send results
        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                logger.info("Skipping execution of %s (not visible)", action_name)
                continue

            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)

                await self._send_websocket_message(
                    action_name, result, tool_func=tool_function
                )
                logger.info("Sent result for %s: %s", action_name, result)

            except Exception as e:
                await self._send_websocket_message(
                    action_name, {"error": str(e)}, tool_func=tool_function
                )
                logger.error("Tool execution failed for %s: %s", action_name, e)

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Track detected language (en/ar)."""
        print("🎤 Starting STT node for mall support...")
        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node available"

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
                    self.audio_recorder.add_frame(frame)
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    if (
                        event.type
                        in [
                            # SpeechEventType.INTERIM_TRANSCRIPT,
                            # SpeechEventType.PREFLIGHT_TRANSCRIPT,
                            SpeechEventType.FINAL_TRANSCRIPT,
                        ]
                        and event.alternatives
                    ):
                        detected_lang = event.alternatives[0].language

                        # Only update _user_language from STT if no
                        # manual language override is active.  When the
                        # user has explicitly chosen a language (e.g.
                        # Arabic), stray English numbers / code-switch
                        # fragments must NOT flip the session language.
                        if not self.manual_language:
                            self._user_language = detected_lang
                        else:
                            self._user_language = self.manual_language

                        # -------------------------------
                        # FIX: Repair wrong-language STT output here
                        # -------------------------------
                        last_alt = event.alternatives[0]
                        raw_text = last_alt.text.strip()
                        target_lang = self.manual_language or self._user_language

                        if is_wrong_script(raw_text):
                            print(f"⚠️ Repairing last alt in {event.type}: {raw_text}")
                            repaired = await repair_text(self, raw_text, target_lang)
                            last_alt.text = repaired
                        # When manual language is set and STT detected a
                        # different language, the transcription text may
                        # be garbled (e.g. Arabic speech rendered as
                        # English phonemes).  Repair it so the LLM
                        # receives correct text.
                        elif (
                            self.manual_language
                            and detected_lang != self.manual_language
                            and raw_text
                        ):
                            print(
                                f"⚠️ Language mismatch: manual={self.manual_language}, "
                                f"detected={detected_lang}. Repairing: {raw_text}"
                            )
                            repaired = await repair_text(
                                self, raw_text, self.manual_language
                            )
                            last_alt.text = repaired

                        # 👂 Capture language from user speech
                        text = event.alternatives[0].text.lower().strip()
                        print("🌐 Detected language:", self._user_language)
                        cfg = get_cfg()
                        switch_cfg = cfg["agents"]["MallSupportAgent"][
                            "language_switch"
                        ]

                        lang_map = {
                            "arabic": ("ar", switch_cfg.get("arabic", {})),
                        }

                        for keyword, (lang_code, cfg) in lang_map.items():
                            if keyword in text:
                                await self.session.say(cfg.get("notice"))
                                await asyncio.sleep(0.8)
                                self.manual_language = self._user_language = lang_code
                                self.background_audio = BackgroundAudioPlayer(
                                    ambient_sound=AudioConfig(
                                        BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8
                                    ),
                                )
                                await self.background_audio.start(
                                    room=self.room, agent_session=self.session
                                )
                                await asyncio.sleep(2.0)
                                await self.background_audio.aclose()
                                await self.session.say(cfg.get("greeting", ""))
                                break
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Dynamically choose  TTS voice based on detected STT language.

        """
        lang = self.manual_language or getattr(self, "_user_language", "en")
        if self.manual_language == "en":
            self._user_language = "en"

        if lang.startswith("ar"):
            print("🗣️ Using Arabic TTS voice")
            chosen_tts = self.arabic_tts
        # elif lang.startswith("zh"):
        #     print("🗣️ Using Mandarin TTS voice")
        #     chosen_tts = self.mandarin_tts
        else:
            activity = self._get_activity_or_raise()
            assert (
                activity.tts is not None
            ), "tts_node called but no TTS node is available"
            chosen_tts = activity.tts

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
        """Speaks immediately after the agent becomes active."""
        cfg = get_cfg()
        self.manual_language = "en"
        self.audio_recorder.start_session(session_id=self.room.name)
        await self.session.say(cfg["greeting"])

    async def on_exit(self):
        """Save recorded audio when session ends."""
        self.audio_recorder.save()
