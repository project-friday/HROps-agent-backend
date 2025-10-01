import asyncio
import json
from cmath import log
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import aiofiles
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import ChatContext, llm
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import assemblyai, elevenlabs, openai, silero

from src.tools.handover import get_handover_tools
from src.tools.onboarding_agent import (
    check_offer_status,
    confirm_joining_date,
    email_documents_checklist,
    escalate_to_onboarding_team,
    get_background_verification_status,
    get_day1_agenda,
    get_documents_checklist,
    get_it_assets,
    get_offer_details,
    get_offer_summary,
    get_preboarding_tasks,
    get_reporting_manager,
    get_work_location,
    log_negotiation,
    mark_deferral,
    schedule_intro_call,
    send_onboarding_summary,
    update_shipping_address,
)
from src.utils.stt_config import make_deepgram_stt

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


ONBOARDING_PROMPT = load_prompt("src/prompts/onboarding.txt")


class OnboardingAgent(Agent):
    def __init__(self, room: rtc.Room, chat_ctx=None):
        self.room = room
        handover_tools = get_handover_tools("onboarding")

        super().__init__(
            instructions=ONBOARDING_PROMPT,
            stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
            tts=elevenlabs.TTS(
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_turbo_v2_5",
            ),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(),
            chat_ctx=chat_ctx,
            tools=[
                check_offer_status,
                get_offer_summary,
                confirm_joining_date,
                get_reporting_manager,
                get_work_location,
                get_preboarding_tasks,
                get_documents_checklist,
                get_offer_details,
                update_shipping_address,
                schedule_intro_call,
                mark_deferral,
                email_documents_checklist,
                send_onboarding_summary,
                get_it_assets,
                get_day1_agenda,
                get_background_verification_status,
                log_negotiation,
                escalate_to_onboarding_team,
            ]
            + handover_tools,
        )

        # --- Action mapping for websocket updates ---
        self.actions = {
            "Checking Offer Status": check_offer_status,
            "Fetching Offer Summary": get_offer_summary,
            "Getting Documents Checklist": get_documents_checklist,
            "Fetching Offer Details": get_offer_details,
            "Getting Manager Details": get_reporting_manager,
            "Sending Summary Mail": send_onboarding_summary,
            "Sending Checklist on Email": email_documents_checklist,
            "Getting Day1 Agenda": get_day1_agenda,
            "Getting Work Location": get_work_location,
            "Getting IT Assets": get_it_assets,
            "Getting Background Verification Status": get_background_verification_status,
            "Logging Negotiation": log_negotiation,
            "Notifying Onboarding Team": escalate_to_onboarding_team,
            "Submitting Deferral Request": mark_deferral,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # --- Filters (hide internal/sensitive keys) ---
        self.tool_result_filters = {
            get_offer_details: ["status", "loacation", "payroll", "benefits"],
            get_offer_summary: ["benefits"],
            get_documents_checklist: ["internal_ref"],
            log_negotiation: ["raw_email"],
            get_background_verification_status: ["remarks", "link", "dispute_info"],
            # avoid exposing internals
        }
        self.email_tools = {
            log_negotiation,
            escalate_to_onboarding_team,
            send_onboarding_summary,
            mark_deferral,
        }

        # --- Card mapping for frontend UI ---
        self.tool_cards = {
            check_offer_status: "offer_status",
            get_offer_summary: "offer_summary",
            get_offer_details: "offer_details",
            get_reporting_manager: "manager_details",
            get_work_location: "work_location",
            get_preboarding_tasks: "preboarding_tasks",
            get_documents_checklist: "documents_checklist",
            update_shipping_address: "shipping_address",
            schedule_intro_call: "intro_call",
            mark_deferral: "deferral",
            email_documents_checklist: "documents_email",
            send_onboarding_summary: "summary_email",
            get_it_assets: "it_assets",
            get_day1_agenda: "day1_agenda",
            get_background_verification_status: "bgv_status",
            log_negotiation: "negotiation",
            escalate_to_onboarding_team: "escalation",
        }

        # --- Only some tools visible in UI ---
        self.visible_tools = {
            check_offer_status,
            get_offer_summary,
            get_offer_details,
            get_reporting_manager,
            get_work_location,
            get_documents_checklist,
            get_background_verification_status,
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
                json.dumps(message),
                topic="lk.transcription",
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
        """Custom LLM node that captures response and tool usage like JobApplicationAgent."""
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

        # Final LLM response
        self.last_llm_response = "".join(buffer).strip()
        print("✅ Full LLM response captured:", self.last_llm_response)

        # Execute queued tools (only visible ones)
        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                print(f"🚫 Skipping execution of {action_name} (not visible)")

                continue

            if tool_function in self.email_tools:
                tool_args["send"] = False

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
