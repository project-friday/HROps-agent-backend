# src/agents/powercut_agent.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterable, Dict, Optional

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    JobContext,
    RunContext,
    function_tool,
    get_job_context,
    llm,
    stt,
    tokenize,
    tts,
    utils,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import azure, elevenlabs, google, openai, sarvam, silero, soniox

from src.config.loader import get_cfg, render
from src.models.data import UserData

load_dotenv()


RunContext_T = RunContext[UserData]


# --- Hangup helper (LiveKit Telephony) ---
async def hangup_call():
    """
    Cleanly ends the LiveKit room and disconnects the call.
    """
    ctx = get_job_context()
    if ctx is None:
        return  # not running in job context
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))


# ---- Import powercut tools ----
from src.tools.powercut_agent import (
    create_ticket,
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
        self._usc_buffer = ""  # buffer to collect digits for USC number
        self._user_language = "te"  # default until detected
        self.openai_tts = openai.TTS(voice="shimmer", model="gpt-4o-mini-tts")
        self.eleven_tts = elevenlabs.TTS(
            voice_id="H8bdWZHK2OgZwTN7ponr",
            model="eleven_multilingual_v2",
        )
        self.azure_tts = azure.TTS(
            voice="te-IN-ShrutiNeural",
        )
        self.gemini_tts = google.beta.GeminiTTS(
            model="gemini-2.5-pro-preview-tts",
            voice_name="Zephyr",
            instructions="Speak in a ploite, professional and clear manner always be to the point, take natural pauses and try to use simple words in telugu",
        )
        creds_path = Path("credentials.json")
        with creds_path.open("r", encoding="utf-8") as f:
            google_creds = json.load(f)

        self.google_tts = google.TTS(
            voice_name="te-IN-Chirp3-HD-Achernar", credentials_info=google_creds
        )
        self.sarvam_tts = sarvam.TTS(
            target_language_code="te-IN",
            speaker="manisha",
            pace=0.9,
            enable_preprocessing=True,
            pitch=0.1,
        )
        super().__init__(
            instructions=self.prompt,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "hi", "te"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(min_speech_duration=0.1),
            chat_ctx=chat_ctx,
            # tts=elevenlabs.TTS(
            #     voice_id="H8bdWZHK2OgZwTN7ponr",
            #     model="eleven_multilingual_v2",
            # ),
            tts=self.azure_tts,
            tools=[
                get_customer_details,
                get_outage_details,
                get_area_mapping,
                schedule_service,
                send_sms,
                create_ticket,
            ],
        )

        # Map action names to functions (used in websocket messages)
        self.actions = {
            "Fetching Customer Details": get_customer_details,
            "Fetching Outage Details": get_outage_details,
            "Fetching Area Mapping": get_area_mapping,
            "Scheduling Service": schedule_service,
            "Sending SMS": send_sms,
            "Creating Ticket": create_ticket,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optionally filter sensitive fields before sending over WS
        self.tool_result_filters = {
            get_customer_details: ["phone", "email"],
            create_ticket: ["address", "issue_description"],
            schedule_service: ["address", "issue_description"],  # hide PII
        }

        # 🎴 Map functions → card names for frontend rendering
        self.tool_cards = {
            get_customer_details: "customer_details",
            get_outage_details: "outage_status",
            get_area_mapping: "area_mapping",
            schedule_service: "service_schedule",
            send_sms: "sms_notification",
            create_ticket: "ticket_creation",
        }

        # Mark which tools should send JSON updates
        self.visible_tools = {
            get_customer_details,
            get_outage_details,
            get_area_mapping,
            schedule_service,
            send_sms,
            create_ticket,
        }

    def _log_to_file(self, label: str, text: str):
        """Write user and LLM messages to logs/powercut_agent.log"""
        try:
            log_path = Path("logs/powercut_agent.log")
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- {label} ---\n{text.strip()}\n")
        except Exception as e:
            print(f"⚠️ Failed to write to log file: {e}")

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

                    # 👀 Detect if USC is being requested
                    if re.search(r"\bUSC\b", chunk, re.IGNORECASE):
                        print("🔔 USC requested by agent")
                        self._usc_requested = True

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                        # 👀 Detect if USC is being requested
                        if re.search(r"\bUSC\b", chunk.delta.content, re.IGNORECASE):
                            print("🔔 USC requested by agent")
                            self._usc_requested = True

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
        self._log_to_file("LLM_OUTPUT", self.last_llm_response)

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

    def _process_transcript(self, transcript: str) -> str:
        """
        Collect digits across transcripts.
        Once a complete USC number is detected, trim trailing 0 if needed and clear buffer.
        """
        digits = re.findall(r"\d+", transcript)
        if digits:
            # Append to buffer
            self._usc_buffer += "".join(digits)

            # Limit buffer size to avoid runaway growth
            if len(self._usc_buffer) > 12:
                self._usc_buffer = self._usc_buffer[-12:]

            # Check if we have enough digits for USC
            if len(self._usc_buffer) >= 9:
                candidate = self._usc_buffer[:10]  # take first 9-10 digits
                if len(candidate) == 10 and candidate.endswith("0"):
                    fixed = candidate[:-1]
                else:
                    fixed = candidate

                # ⚡ Clear buffer after processing a USC number
                self._usc_buffer = ""

                return fixed

        # No complete USC yet → return original transcript
        return transcript

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        print("🎤 Starting custom STT node...")
        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node is available"

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
                        self._user_language = event.alternatives[0].language
                        transcript = event.alternatives[0].text

                        processed_text = self._process_transcript(transcript)
                        print("processed_text:", processed_text)

                        # Only alter the event if processing changed it
                        if processed_text != transcript:
                            print(f"🔎 Raw transcript: {transcript}")
                            print(f"✅ Fixed transcript: {processed_text}")

                            # Replace the alternatives with a new SpeechData
                            event = stt.SpeechEvent(
                                type=event.type,
                                request_id=event.request_id,
                                alternatives=[
                                    stt.SpeechData(
                                        text=processed_text,
                                        language=event.alternatives[0].language,
                                        start_time=event.alternatives[0].start_time,
                                        end_time=event.alternatives[0].end_time,
                                        confidence=event.alternatives[0].confidence,
                                        speaker_id=event.alternatives[0].speaker_id,
                                        is_primary_speaker=event.alternatives[
                                            0
                                        ].is_primary_speaker,
                                    )
                                ],
                                recognition_usage=event.recognition_usage,
                            )
                        self._log_to_file("USER_INPUT", processed_text)

                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Dynamically choose TTS engine based on last STT language.
        Telugu -> Sarvam Manisha
        Hindi -> ElevenLabs
        Others -> ElevenLabs
        """

        if self._user_language == "te":
            chosen_tts = self.sarvam_tts
        else:
            chosen_tts = self.eleven_tts

        # ---- Handle non-streaming TTS engines ----
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

            async def _forward_input() -> None:
                async for chunk in text:
                    stream.push_text(chunk)
                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def on_enter(self):
        cfg = get_cfg()
        await self.session.say(cfg["greeting"])

    @function_tool(
        description="Transfer the current SIP call to a human agent via phone number."
    )
    async def transfer_call(self, context: RunContext_T):
        """Transfers the active SIP participant to a specified phone number."""

        await self.session.generate_reply(
            user_input="Please hold while I connect you to a human agent."
        )

        try:
            # Get LiveKit credentials
            livekit_url = os.getenv("LIVEKIT_URL")
            api_key = os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_API_SECRET")

            # Use the verified number - ensure proper E.164 format
            transfer_number = "tel:+917994820760"

            # Initialize LiveKit API
            livekit_api = api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret,
            )

            room = context.userdata.ctx.room

            # ✅ FIXED: Use remote_participants instead of participants
            logger.info("=== Room Participants Debug ===")
            for participant in room.remote_participants.values():
                participant_info = {
                    "identity": participant.identity,
                    "sid": participant.sid,
                    "kind": getattr(participant, "kind", "N/A"),
                    "type": type(participant).__name__,
                    "is_local": participant.is_local,
                }
                logger.info(f"Participant: {participant_info}")

            # Find SIP participant
            sip_participant = None
            for participant in room.remote_participants.values():
                identity = participant.identity
                if identity.startswith(("sip_", "+", "tel:")):
                    sip_participant = participant
                    logger.info(f"🎯 Found SIP participant: {identity}")
                    break

            if not sip_participant:
                logger.error("❌ No SIP participant found")
                await self.session.generate_reply(
                    user_input="I'm sorry, I couldn't find a suitable connection to transfer your call."
                )
                await livekit_api.aclose()
                return

            # ✅ CORRECT: Create TransferSIPParticipantRequest with proper structure
            transfer_req = api.TransferSIPParticipantRequest(
                room_name=room.name,
                participant_identity=sip_participant.identity,
                transfer_to=transfer_number,
                play_dialtone=True,
                # Optional: Add headers if needed by Twilio
                headers={
                    "User-Agent": "LiveKit-Agent",
                    # "X-Twilio-Caller-ID": "+917994820760"  # If Twilio needs specific headers
                },
                # Optional: Set ringing timeout (default is usually fine)
                # ringing_timeout=duration_pb2.Duration(seconds=30)
            )

            logger.info(
                f"🔄 Transferring {sip_participant.identity} to {transfer_number}"
            )

            # Execute the transfer
            await livekit_api.sip.transfer_sip_participant(transfer_req)
            logger.info("✅ Call transfer initiated successfully")

            # Close API connection properly
            await livekit_api.aclose()

        except Exception as e:
            logger.error(f"❌ Call transfer failed: {e}")

            # More specific error handling based on SIP status codes
            error_msg = str(e)

            if "403" in error_msg and "Caller ID verification rejected" in error_msg:
                # This is a Twilio-specific restriction
                logger.error(
                    "🔒 Twilio Caller ID verification failed - number may need additional verification"
                )
                await self.session.generate_reply(
                    user_input="I'm unable to transfer due to security restrictions. Please contact our support team directly at +917994820760."
                )

            elif "408" in error_msg or "timeout" in error_msg.lower():
                # Timeout error
                await self.session.generate_reply(
                    user_input="The transfer timed out. The line may be busy. Let me help you with your issue directly."
                )

            elif "403" in error_msg:
                # General forbidden error
                await self.session.generate_reply(
                    user_input="I don't have permission to transfer calls at the moment. Please stay on the line for assistance."
                )

            else:
                # Generic error
                await self.session.generate_reply(
                    user_input="I encountered a technical issue transferring your call. Let me continue helping you with your power outage issue."
                )
