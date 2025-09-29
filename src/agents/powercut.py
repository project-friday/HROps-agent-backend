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
from livekit.plugins import elevenlabs, openai, silero, soniox

from src.config.loader import get_cfg, render

# ---- Import powercut tools ----
from src.tools.powercut_agent import (
    get_area_mapping,
    get_customer_details,
    get_outage_details,
    schedule_service,
    send_sms,
)

load_dotenv()

logger = logging.getLogger("hr-eve-powercut-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


class PowercutAgent(Agent):
    """
    Powercut assistant:
    - Identifies customers (via USC number or manual address)
    - Retrieves outage details using area code
    - Escalates to service scheduling if no outage detected
    - Streams action JSONs via WebSocket
    """

    def __init__(self, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()
        prompt = load_prompt("src/prompts/powercut.txt")
        self.prompt = render(prompt, self.cfg)

        super().__init__(
            instructions=self.prompt,
            stt=soniox.STT(params=soniox.STTOptions(language_hints=["en", "hi", "te"])),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            tts=elevenlabs.TTS(
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_multilingual_v2",
            ),
            tools=[
                get_customer_details,
                get_outage_details,
                get_area_mapping,
                schedule_service,
            ],
        )

        # Map action names to functions (used in websocket messages)
        self.actions = {
            "Fetching Customer Details": get_customer_details,
            "Fetching Outage Details": get_outage_details,
            "Fetching Area Mapping": get_area_mapping,
            "Scheduling Service": schedule_service,
            "Sending SMS": send_sms,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optionally filter sensitive fields before sending over WS
        self.tool_result_filters = {
            get_customer_details: ["phone", "email"],  # hide PII
        }

        # 🎴 Map functions → card names for frontend rendering
        self.tool_cards = {
            get_customer_details: "customer_details",
            get_outage_details: "outage_status",
            get_area_mapping: "area_mapping",
            schedule_service: "service_schedule",
            send_sms: "sms_notification",
        }

        # Mark which tools should send JSON updates
        self.visible_tools = {
            get_customer_details,
            get_outage_details,
            get_area_mapping,
            schedule_service,
            send_sms,
        }

    async def _send_websocket_message(
        self, action: str, result: Dict[str, Any] = None, tool_func=None
    ):
        """Send structured action JSONs via WebSocket."""
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
        """Custom LLM node that captures response and streams actions."""

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
                    print("🤖 LLM str chunk:", chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        print("🛠️ Tool calls:", chunk.delta.tool_calls)

                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"

                            # Parse args safely
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

        # Capture final response
        self.last_llm_response = "".join(buffer).strip()
        print("✅ Full LLM response captured:", self.last_llm_response)

        # Execute queued tools + send results
        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                print(f"🚫 Skipping execution of {action_name} (not visible)")
                continue

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
        cfg = get_cfg()
        await self.session.say(cfg["greeting"])
