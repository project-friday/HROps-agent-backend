# src/agents/powercut_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import llm
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import elevenlabs, openai, silero

# ---- Import powercut tools ----
from src.tools.powercut_agent import (
    get_area_mapping,
    get_customer_details,
    get_outage_details,
    schedule_service,
)
from src.utils.stt_config import make_deepgram_stt

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


logger = logging.getLogger("hr-eve-powercut-agent")
logger.setLevel(logging.INFO)

EVE_POWERCUT_PROMPT = load_prompt("src/prompts/powercut.txt")


class PowercutAgent(Agent):
    """
    Powercut assistant:
    - Identifies customers (via UAC number or manual address)
    - Retrieves outage details using area code
    - Escalates to service scheduling if no outage detected
    """

    def __init__(self, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        super().__init__(
            instructions=EVE_POWERCUT_PROMPT,
            stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            tts=elevenlabs.TTS(
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_turbo_v2_5",
            ),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            tools=[
                get_customer_details,
                get_outage_details,
                get_area_mapping,
                schedule_service,
            ],
        )

        self.actions = {
            "Getting Customer Details": get_customer_details,
            "Getting Outage Details": get_outage_details,
            "Getting Area Mapping": get_area_mapping,
            "Scheduling Service": schedule_service,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        self.tool_result_filters = {
            get_customer_details: ["uac_number"],  # hide sensitive info
            schedule_service: ["success"],
        }

        self.tool_cards = {
            get_customer_details: "customer_details",
            get_outage_details: "outage_details",
            get_area_mapping: "area_mapping",
            schedule_service: "service_schedule",
        }

        self.visible_tools = {
            get_customer_details,
            get_outage_details,
            get_area_mapping,
            schedule_service,
        }

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
            print(f"✅ Sent WebSocket message: {message}")
        except Exception as e:
            print(f"❌ Failed to send WebSocket message: {e}")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        """Custom LLM node that captures full response text."""

        activity = self._get_activity_or_raise()
        assert activity.llm is not None
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
                    print("⚡ LLM str chunk:", chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        print("🔧 Tool calls:", chunk.delta.tool_calls)

                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"

                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    print(
                                        f"⚠️ Invalid JSON for {tool_name}: {tool_args}"
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

        self.last_llm_response = "".join(buffer).strip()

        for action_name, tool_function, tool_args in pending_tools:
            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)

                await self._send_websocket_message(
                    action_name, result, tool_func=tool_function
                )
                print(f"✅ Sent result for {action_name}: {result}")

            except Exception as e:
                await self._send_websocket_message(
                    action_name, {"error": str(e)}, tool_func=tool_function
                )
                print(f"❌ Tool execution failed for {action_name}: {e}")

    async def on_enter(self):
        await self.session.generate_reply()
