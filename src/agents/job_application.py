# ______________________________________________________________________________________________#
# src/agents/job_application.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import aiofiles

# from custom.livekit.plugins import murfai
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import llm
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import assemblyai, elevenlabs, openai, silero

from src.tools.handover import go_assessment, handover_to_onboarding

# ---- import the REAL tools directly ----
from src.tools.job_application_agent import (
    check_application_status,
    check_interview_availability,
    email_conversation_summary,
    escalate_to_hiring_team,
    get_upcoming_interview,
    list_applications_by_email,
    query_knowledge_base,
    reschedule_interview,
    select_application_by_choice,
)
from src.utils.stt_config import make_deepgram_stt

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


logger = logging.getLogger("hr-eve-agent")
logger.setLevel(logging.INFO)

EVE_SYSTEM_PROMPT = load_prompt("src/prompts/job_application.txt")


class JobApplicationAgent(Agent):
    """
    Job application assistant:
    - Status + RAG only
    - Tiny fillers (prompt-driven)
    - Repeat-back confirmation for name / phone / email
    - Titles-only list for multi-application
    - Status-only answer; keep details for follow-ups
    """

    def __init__(self, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        super().__init__(
            instructions=EVE_SYSTEM_PROMPT,
            # stt=assemblyai.STT(),
            stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
            # # stt = openai.STT(model="gpt-4o-transcribe"),
            # stt=openai.STT(
            #     model="gpt-4o-transcribe",
            #     language="en",          # force English
            #     detect_language=False   # disable auto language detection
            # ),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),
            tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                voice_id="H8bdWZHK2OgZwTN7ponr",
                # model="eleven_multilingual_v2",
                model="eleven_turbo_v2_5",
            ),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            tools=[
                list_applications_by_email,
                select_application_by_choice,
                check_application_status,
                query_knowledge_base,
                get_upcoming_interview,
                check_interview_availability,
                reschedule_interview,
                handover_to_onboarding,
                escalate_to_hiring_team,
                email_conversation_summary,
                go_assessment,
            ],
        )

        # Map action names to functions (used in websocket messages)
        self.actions = {
            "Fetching Applications": list_applications_by_email,
            "Selecting Application": select_application_by_choice,
            "Checking Application Status": check_application_status,
            "Querying Knowledge Base": query_knowledge_base,
            "Fetching Interview Details": get_upcoming_interview,
            "Checking Interview Availability": check_interview_availability,
            "Rescheduling Interview": reschedule_interview,
            "Processing Request": handover_to_onboarding,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}
        self.tool_result_filters = {
            list_applications_by_email: ["email"],
            check_application_status: [
                "email",
                "phone",
                "human_status",
                "updated_at_human",
                "reschedules",
                "found",
            ],
            get_upcoming_interview: ["has_interview", "application_id"],
            reschedule_interview: ["reschedule_args"],
        }

        # 🎴 Card mapping for frontend
        self.tool_cards = {
            list_applications_by_email: "applications_list",
            check_application_status: "application_status",
            get_upcoming_interview: "upcoming_interview",
            reschedule_interview: "interview_reschedule",
            query_knowledge_base: "knowledge_base",
            check_interview_availability: "interview_availability",
            select_application_by_choice: "application_selection",
            handover_to_onboarding: "handover",
        }

        self.visible_tools = {
            list_applications_by_email,
            check_application_status,
            get_upcoming_interview,
            reschedule_interview,
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
                    print("🤖 LLM str chunk:", chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        print("🛠️ Tool calls:", chunk.delta.tool_calls)

                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"

                            # 🔑 Parse args safely (JSON string → dict)
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
                                # Send "action started"
                                await self._send_websocket_message(action_name)

                                # Queue tool execution after LLM finishes
                                pending_tools.append(
                                    (action_name, tool_function, tool_args)
                                )

                yield chunk

        # Capture final LLM response
        self.last_llm_response = "".join(buffer).strip()
        print("✅ Full LLM response captured:", self.last_llm_response)

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

    # --- speaks immediately after the agent becomes active (e.g., after handover) ---
    async def on_enter(self):
        await self.session.generate_reply()


# ______________________________________________________________________________________________#
