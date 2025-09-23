# src/agents/assessment_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import llm, stt, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import elevenlabs, openai, silero

# ---- Import assessment tools ----
from src.tools.assessment_agent import (
    check_assessment_status,
    escalate_to_assessment_team,
    get_assessment_details,
    get_assessment_result,
    reschedule_assessment,
    send_assessment_reminder,
)
from src.utils.stt_config import make_deepgram_stt
from src.utils.translator import translate_to_english

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


logger = logging.getLogger("hr-eve-assessment-agent")
logger.setLevel(logging.INFO)

EVE_ASSESSMENT_PROMPT = load_prompt("src/prompts/assessment.txt")


class AssessmentAgent(Agent):
    """
    Assessment assistant:
    - Handles technical assessments and evaluations
    - Provides status, reminders, rescheduling
    - RAG-based answers for knowledge base
    - Titles-only list for multiple assessments
    - Keeps responses concise
    """

    def __init__(self, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        super().__init__(
            instructions=EVE_ASSESSMENT_PROMPT,
            stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            tts=elevenlabs.TTS(
                voice_id="kjHz50TasdqbpbfK4uaN",
                model="eleven_turbo_v2_5",
                language="es",
            ),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            tools=[
                get_assessment_details,
                check_assessment_status,
                reschedule_assessment,
                send_assessment_reminder,
                escalate_to_assessment_team,
                get_assessment_result,
            ],
        )

        # Map actions to functions
        self.actions = {
            "Getting Assessment Details": get_assessment_details,
            "Checking Assessment Status": check_assessment_status,
            "Rescheduling Assessment": reschedule_assessment,
            "Sending Assessment Reminder": send_assessment_reminder,
            "Escalating to Assessment Team": escalate_to_assessment_team,
            "Getting Assessment Result": get_assessment_result,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # keys that should be filtered out from tool results before sending to frontend
        self.tool_result_filters = {
            # check_assessment_status: ["status", "deadline", "submitted_at", "score"],
            # reschedule_assessment: ["new_deadline"],
            get_assessment_details: ["instructions", "duration"],
            get_assessment_result: ["submitted_at"],
        }

        self.tool_cards = {
            get_assessment_details: "assessment_details",
            check_assessment_status: "assessment_status",
            reschedule_assessment: "assessment_reschedule",
            send_assessment_reminder: "assessment_reminder",
            escalate_to_assessment_team: "assessment_escalation",
            get_assessment_result: "assessment_result",
        }

        self.visible_tools = {
            get_assessment_details,
            check_assessment_status,
            reschedule_assessment,
            send_assessment_reminder,
            escalate_to_assessment_team,
            get_assessment_result,
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

    async def _translate_and_send_llm_response(
        self, raw_response: str, role: str
    ) -> None:
        """
        Translate the final LLM response to English and send it to the WebSocket.
        Falls back to the raw response if translation fails.
        """
        try:
            translated_text = await translate_to_english(text=raw_response)
            print("🌍 Translated to English:", translated_text)
        except Exception as e:
            print(f"⚠️ Translation failed, sending raw text: {e}")
            translated_text = raw_response

        # 📤 Forward translated response to WebSocket
        try:
            await self.room.local_participant.send_text(
                json.dumps({"role": role, "translation": translated_text}),
                topic="lk.transcription",
            )
            print("📤 Translated response sent to WebSocket")
        except Exception as e:
            print(f"⚠️ Failed to forward translated response: {e}")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        """Custom LLM node that captures full response text."""

        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM)

        tool_choice = model_settings.tool_choice if model_settings else llm.NOT_GIVEN
        activity_llm = activity.llm
        conn_options = activity.session.conn_options.llm_conn_options

        buffer: list[str] = []
        pending_tools: list[tuple[str, callable, dict]] = []
        last_user_msg = next(
            (
                item
                for item in reversed(chat_ctx.items)
                if item.type == "message" and item.role == "user"
            ),
            None,
        )
        # print("🧑‍💻 Last user message:", last_user_msg.text_content if last_user_msg else "None")
        if last_user_msg and last_user_msg.text_content:
            asyncio.create_task(
                self._translate_and_send_llm_response(
                    last_user_msg.text_content, "user"
                )
            )

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

                            # 🔑 Parse args safely
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

        # Capture final LLM response
        raw_response = "".join(buffer).strip()
        print("✅ Full LLM response captured:", raw_response)
        if raw_response:
            asyncio.create_task(
                self._translate_and_send_llm_response(raw_response, "bot")
            )
        # print("✅ Full LLM response captured:", self.last_llm_response)

        # Execute queued tools and send results
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
        """Speaks immediately after the agent becomes active (e.g., after handover)."""
        await self.session.generate_reply()
