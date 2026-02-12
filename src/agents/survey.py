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
from livekit.plugins import cartesia, elevenlabs, inworld, silero, soniox

from src.config.loader import get_cfg
from src.tools.survey_tools import feedback_call_tool, feedback_sms_tool
from src.utils.transcription_utils import is_wrong_script, repair_text

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
        # ---------- TTS setup (multi-language) ----------
        tts_cfg = cfg.get("tts", {})
        voices = tts_cfg.get("voices", {})
        providers = tts_cfg.get("providers", {})

        # ---- English (Inworld Sarah) ----
        english_voice = voices.get("english")
        self.english_tts = None
        if english_voice and providers.get("english", "inworld") == "inworld":
            self.english_tts = inworld.TTS(voice=english_voice)
        else:
            raise RuntimeError("❌ No English Inworld voice configured")

        # ---- Arabic (Elevenlabs) ----
        arabic_voice = voices.get("arabic")
        self.arabic_tts = None
        if arabic_voice and providers.get("arabic", "elevenlabs") == "elevenlabs":
            self.arabic_tts = elevenlabs.TTS(
                voice_id=arabic_voice,
                model="eleven_turbo_v2_5",
            )
        else:
            raise RuntimeError("❌ No Arabic ElevenLabs voice configured")

        super().__init__(
            instructions=EVE_SURVEY_PROMPT,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            tools=[feedback_call_tool, feedback_sms_tool],
            chat_ctx=chat_ctx,
        )

        # --- Tool actions ---
        self.actions = {
            "Feedback Rating": feedback_call_tool,
            "Feedback SMS": feedback_sms_tool,
            # "Submit Survey Response": submit_survey_response,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        self.visible_tools = {feedback_call_tool, feedback_sms_tool}

        self.tool_result_filters = {
            feedback_call_tool: [],
            feedback_sms_tool: [],
        }

        self.tool_cards = {
            feedback_call_tool: "feedback_call_result",
            feedback_sms_tool: "feedback_sms_result",
        }

        self.manual_language = None  # tracks manually switched language
        self._user_language = "en"  # default to English

    async def _send_websocket_message(
        self, action: str, result: Dict[str, Any] = None, tool_func=None
    ):
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
        """Capture LLM outputs and execute tools with arguments."""
        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node available"
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

                    # 🟢 Detect when the LLM triggers a tool call
                    if chunk.delta and chunk.delta.tool_calls:
                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"

                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    tool_args = {}

                            # Match the function by name
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

        self.last_llm_response = "".join(buffer).strip()

        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                continue
            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)

                await self._send_websocket_message(
                    action_name, result, tool_func=tool_function
                )
            except Exception as e:
                await self._send_websocket_message(
                    action_name, {"error": str(e)}, tool_func=tool_function
                )

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
                        detected_lang = event.alternatives[0].language
                        last_alt = event.alternatives[0]
                        raw_text = last_alt.text.strip()

                        # Preserve manually / session-set language;
                        # only update from STT when no override exists.
                        session_lang = getattr(self.session, "state", {}).get(
                            "language"
                        ) or getattr(self, "manual_language", None)
                        if not session_lang:
                            self._user_language = detected_lang
                        else:
                            self._user_language = session_lang

                        target_lang = session_lang or self._user_language

                        if is_wrong_script(raw_text):
                            print(f"⚠️ Repairing last alt in {event.type}: {raw_text}")
                            repaired = await repair_text(self, raw_text, target_lang)
                            last_alt.text = repaired
                        elif (
                            session_lang and detected_lang != session_lang and raw_text
                        ):
                            print(
                                f"⚠️ Language mismatch: session={session_lang}, "
                                f"detected={detected_lang}. Repairing: {raw_text}"
                            )
                            repaired = await repair_text(self, raw_text, session_lang)
                            last_alt.text = repaired

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
        lang = getattr(self.session, "state", {}).get("language") or getattr(
            self, "_user_language", "en"
        )

        if lang.startswith("ar"):
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
        lang = getattr(self.session, "state", {}).get("language", "en")
        survey_cfg = cfg["agents"]["SurveyAgent"]["greeting"]
        # Default to English if language not found
        greeting = survey_cfg.get(lang[:2], survey_cfg.get("en"))
        await self.session.say(greeting)
