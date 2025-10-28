# src/agents/mallsupport_agent.py
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
from livekit.plugins import azure, elevenlabs, openai, silero, soniox, deepgram

from src.config.loader import get_cfg
from src.tools.mallsupport_tools import create_ticket, query_knowledge_base

logger = logging.getLogger("dubai-mall-support-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


EVE_MALL_PROMPT = load_prompt("src/prompts/mallsupport.txt")

_LANG_POLICY = (
    "STRICT LANGUAGE POLICY:\n"
    "- If the user speaks English, reply ONLY in English.\n"
    "- If the user speaks Arabic, reply ONLY in Arabic.\n"
    "- If the user speaks any other language, reply in English with: "
    "'Sorry—I can only assist you in English or Arabic at Dubai Mall.'\n"
    "- Keep replies concise and polite."
)


class MallSupportAgent(Agent):
    """
    Dubai Mall Customer Support Agent:
    - English → Deepgram STT, reply in English
    - Arabic → Soniox STT, reply in Arabic
    - Other → Soniox STT (for detection), reply with English apology
    - Switch engines at END_OF_SPEECH
    """

    def __init__(self, cfg: dict, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()

        # Arabic TTS voice
        arabic_voice = cfg["voices"]["arabic"]
        self.arabic_tts = elevenlabs.TTS(
            voice_id=arabic_voice, model=cfg["tts"].get("model")
        )

        super().__init__(
            instructions=f"{EVE_MALL_PROMPT}\n\n{_LANG_POLICY}",
            # Default STT: Soniox (good multilingual accuracy + hints)
            # Silero VAD handles turn-taking and emits END_OF_SPEECH. :contentReference[oaicite:2]{index=2}
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.10, min_silence_duration=0.30),
            ),
            tools=[query_knowledge_base, create_ticket],
            chat_ctx=chat_ctx,
        )

        # Deepgram STT: enable language detection so we can switch away when user speaks Arabic. :contentReference[oaicite:3]{index=3}
        self._dg_stt = deepgram.STT(
            detect_language=True,   # IMPORTANT for switching away from Deepgram on Arabic
            # leave language unset so detection is effective (plugin default is en-US). :contentReference[oaicite:4]{index=4}
        )

        self._stt_provider = "soniox"        # 'soniox' | 'deepgram' (current engine)
        self._switch_after_eos: str | None = None
        self._user_language = "en"           # last reported language code
        self._reply_mode = "en"              # 'en' | 'ar' | 'apology'

        self.actions = {
            "Query Knowledge Base": query_knowledge_base,
            "Creating ticket": create_ticket,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        self.tool_result_filters = {
            query_knowledge_base: ["internal_id", "metadata"],
            create_ticket: ["ticket_id"],
        }
        self.tool_cards = {
            query_knowledge_base: "knowledge_base_result",
            create_ticket: "ticket_creation_result",
        }
        self.visible_tools = {create_ticket}

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
        # If unsupported language was detected, bypass tools/model and apologize immediately.
        if getattr(self, "_reply_mode", "en") == "apology":
            apology = "Sorry—I can only assist you in English or Arabic at Dubai Mall."
            self.last_llm_response = apology
            yield apology
            return

        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM)

        tool_choice = model_settings.tool_choice if model_settings else llm.NOT_GIVEN
        activity_llm = activity.llm
        conn_options = activity.session.conn_options.llm_conn_options

        buffer: list[str] = []
        pending_tools: list[tuple[str, callable, dict]] = []

        async with activity_llm.chat(
            chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice, conn_options=conn_options
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
                                    logger.warning("Invalid JSON for %s: %s", tool_name, tool_args)
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
                                pending_tools.append((action_name, tool_function, tool_args))
                yield chunk

        self.last_llm_response = "".join(buffer).strip()

        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                logger.info("Skipping execution of %s (not visible)", action_name)
                continue
            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)
                await self._send_websocket_message(action_name, result, tool_func=tool_function)
                logger.info("Sent result for %s: %s", action_name, result)
            except Exception as e:
                await self._send_websocket_message(action_name, {"error": str(e)}, tool_func=tool_function)
                logger.error("Tool execution failed for %s: %s", action_name, e)

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """
        Auto-switch STT at END_OF_SPEECH:
        - English → Deepgram
        - Arabic → Soniox
        - Other → Soniox; reply_mode='apology'
        """
        print("🎤 Starting STT node for mall support...")

        while True:
            # Announce which engine is ACTIVE for this turn
            if self._stt_provider == "deepgram":
                chosen_stt = self._dg_stt
                print("🟢 Active STT engine: DEEPGRAM")
                print("🎧 Using Deepgram STT (language detection enabled)")
            else:
                activity = self._get_activity_or_raise()
                chosen_stt = activity.stt  # Soniox wired in __init__
                print("🟢 Active STT engine: SONIOX")
                print("🎧 Using Soniox STT (auto/multilingual)")

            conn_options = self.session.conn_options.stt_conn_options

            async with chosen_stt.stream(conn_options=conn_options) as stream:

                @utils.log_exceptions()
                async def _forward_input() -> None:
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(_forward_input())
                switched = False
                try:
                    async for event in stream:
                        if event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                            alt = event.alternatives[0]
                            # Prefer provider’s language code; Deepgram sets it when detect_language=True. :contentReference[oaicite:5]{index=5}
                            lang_code = (getattr(alt, "language", None) or "").lower() or "und"
                            self._user_language = lang_code
                            print("🌐 Detected language:", lang_code)

                            # Decide next engine + reply mode (actual switch happens on EOS)
                            if lang_code.startswith("en"):
                                self._switch_after_eos = "deepgram"
                                self._reply_mode = "en"
                                print("📎 Decision: route next turn to DEEPGRAM, reply in EN")
                            elif lang_code.startswith("ar"):
                                self._switch_after_eos = "soniox"
                                self._reply_mode = "ar"
                                print("📎 Decision: route next turn to SONIOX, reply in AR")
                            else:
                                self._switch_after_eos = "soniox"
                                self._reply_mode = "apology"
                                print("📎 Decision: keep SONIOX for detection; reply with EN apology")

                        elif event.type == SpeechEventType.END_OF_SPEECH:
                            target = self._switch_after_eos
                            self._switch_after_eos = None
                            if target and target != self._stt_provider:
                                if target == "deepgram":
                                    print("🔁 Switching STT provider → Deepgram (at END_OF_SPEECH)")
                                    self._stt_provider = "deepgram"
                                else:
                                    print("🔁 Switching STT provider → Soniox (at END_OF_SPEECH)")
                                    self._stt_provider = "soniox"
                                switched = True
                                break  # reopen with the new engine

                        # forward all events
                        yield event
                finally:
                    await utils.aio.cancel_and_wait(forward_task)

            if switched:
                continue  # reopen with chosen engine for the next turn
            break

    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        TTS based on reply_mode:
        - 'en' → default (English) TTS
        - 'ar' → Arabic TTS
        - 'apology' → English TTS
        """
        mode = getattr(self, "_reply_mode", "en")
        if mode == "ar":
            print("🗣️ Using Arabic TTS voice")
            chosen_tts = self.arabic_tts
        else:
            activity = self._get_activity_or_raise()
            assert activity.tts is not None, "tts_node called but no TTS node is available"
            chosen_tts = activity.tts

        wrapped_tts = chosen_tts
        if not chosen_tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=chosen_tts,
                sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True),
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
        cfg = get_cfg()
        await self.session.say(cfg["greeting"])
