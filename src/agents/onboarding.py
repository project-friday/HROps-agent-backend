from cmath import log
from livekit.agents import ChatContext
from livekit import rtc
from livekit.agents.voice import Agent,ModelSettings
from livekit.plugins import openai, silero, assemblyai
from livekit.plugins import elevenlabs
from livekit.agents import llm
import asyncio
import aiofiles
import json
from pathlib import Path
from typing import AsyncGenerator,Dict, Any
from src.tools.handover import handover_to_applications
from src.tools.onboarding_agent import (
    check_offer_status,
    get_offer_summary,
    confirm_joining_date,
    get_reporting_manager,
    get_work_location,
    get_preboarding_tasks,
    get_day1_agenda,
    get_it_assets,
    get_documents_checklist,
    get_offer_details,
    update_shipping_address,
    schedule_intro_call,
    mark_deferral,
    email_documents_checklist,
    send_onboarding_summary,
    get_background_verification_status,
    log_negotiation,
    escalate_to_onboarding_team

)
from dotenv import load_dotenv  
load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()

ONBOARDING_PROMPT =load_prompt("src/prompts/onboarding.txt")

class OnboardingAgent(Agent):
    def __init__(self, room: rtc.Room, chat_ctx=None):
        self.room = room
        # print("room:", self.room)
        super().__init__(
        instructions=ONBOARDING_PROMPT,
        stt=assemblyai.STT(),
        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),
        tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_multilingual_v2",
            ),
        llm=openai.LLM(model="gpt-4.1"),
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
                handover_to_applications,
                get_background_verification_status,
                log_negotiation,
                escalate_to_onboarding_team
            ],)

        # Mapping of actions to tool functions
        self.actions = {
            "checking offer status": check_offer_status,
            "getting info": get_documents_checklist,
            "getting offer": get_offer_details,
            "getting manager details": get_reporting_manager,
            "sending summary mail": send_onboarding_summary,
            "getting orientation details": get_day1_agenda,
            "getting work location details":get_work_location,
            "getting assest info": get_it_assets,
            "getting bgv status": get_background_verification_status,
            "logging negotiation": log_negotiation

        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

    async def _send_websocket_message(self, action: str, result: Dict[str, Any] = None):
        """Send WebSocket message with action and optional result"""
        message = {"action": action}
        if result is not None:
            message["result"] = result

        try:
            await self.room.local_participant.send_text(
                json.dumps(message),
                topic="lk.transcription"
            )
            # print(f"✅ Sent WebSocket message: {message}")
        except Exception as e:
            print(f"❌ Failed to send WebSocket message: {e}")

    async def on_enter(self):
        # candidate = self.chat_ctx.session.userdata.get("candidate", {})
        # query = self.session.userdata.get("handover_query")

        # name = candidate.get("name", "there")
        # email = candidate.get("email", "unknown")

        await self.session.generate_reply(
            # instructions=( )
        )

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
        pending_tools: list[tuple[str, callable, dict]] = []  # (action_name, tool_fn, tool_args)

        async with activity_llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            conn_options=conn_options,
        ) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    buffer.append(chunk)
                    # print("🤖 LLM str chunk:", chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        # print("🛠️ Tool calls:", chunk.delta.tool_calls)

                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"

                            # 🔑 Always parse arguments safely
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    print(f"⚠️ Invalid JSON for {tool_name}: {tool_args}")
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

                                # Queue for execution after LLM completes
                                pending_tools.append((action_name, tool_function, tool_args))

                yield chunk

        # Capture final LLM response
        self.last_llm_response = "".join(buffer).strip()
        # print("✅ Full LLM response captured:", self.last_llm_response)

        # Now execute queued tools and send results
        # for action_name, tool_function, tool_args in pending_tools:
        #     try:
        #         if asyncio.iscoroutinefunction(tool_function):
        #             result = await tool_function(**(tool_args or {}))
        #         else:
        #             result = tool_function(**(tool_args or {}))

        #         await self._send_websocket_message(action_name, result)
        #         # print(f"✅ Sent result for {action_name}: {result}")

        #     except Exception as e:
        #         await self._send_websocket_message(action_name, {"error": str(e)})
        #         print(f"❌ Tool execution failed for {action_name}: {e}")


    async def on_enter(self):
        self.session.generate_reply()
