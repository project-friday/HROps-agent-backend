# src/agents/mallsupport_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from livekit import rtc
from livekit.agents import llm, stt, utils, tts, tokenize
from livekit.agents.voice import Agent, ModelSettings
from livekit.agents.stt import SpeechEventType

# Plugins
from livekit.plugins import elevenlabs, openai, silero, soniox, azure

# ---- Import only the knowledge base tool ----
from src.tools.mallsupport_tools import query_knowledge_base
from src.config.loader import get_cfg

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
        self.azure_tts_en = azure.TTS(voice="en-US-JennyNeural")
        super().__init__(
            instructions=EVE_MALL_PROMPT,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.2),
            vad=silero.VAD.load(min_speech_duration=0.1),
            # tts=elevenlabs.TTS(
            #     voice_id="H8bdWZHK2OgZwTN7ponr",
            #     model="eleven_multilingual_v2",
            # ),
            tts=self.azure_tts_en,
            tools=[query_knowledge_base],
            chat_ctx=chat_ctx,
        )


        self.actions = {
            "Query Knowledge Base": query_knowledge_base,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optional: keys to filter from tool results before sending
        self.tool_result_filters = {
            query_knowledge_base: ["internal_id", "metadata"],
        }

        self.tool_cards = {
            query_knowledge_base: "knowledge_base_result",
        }

        self.visible_tools = {query_knowledge_base}

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
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    if (
                        event.type == SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                    ):
                        # 👂 Capture language from user speech
                        self._user_language = event.alternatives[0].language
                        print("🌐 Detected language:", self._user_language)
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Dynamically choose Azure TTS voice based on detected STT language.
        Arabic -> FatimaNeural
        English -> JennyNeural
        """
        # 👂 Detect the last user language (default English)
        lang = getattr(self, "_user_language", "en")
        if lang.startswith("ar"):
            chosen_tts = azure.TTS(voice="ar-AE-FatimaNeural")
        else:
            chosen_tts = self.azure_tts_en

        # Handle non-streaming engines (mirror Powercut)
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
        cfg = get_cfg()
        """Speaks immediately after the agent becomes active."""
        await self.session.say(cfg["greeting"])
