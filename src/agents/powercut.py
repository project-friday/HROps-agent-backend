# src/agents/powercut_agent.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterable, Dict

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import llm, stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import azure, elevenlabs, google, openai, silero, soniox

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
        creds_path = Path("credentials.json")
        with creds_path.open("r", encoding="utf-8") as f:
            google_creds = json.load(f)

        self.google_tts = google.TTS(
            voice_name="te-IN-Chirp3-HD-Achernar", credentials_info=google_creds
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
        Telugu -> OpenAI shimmer; others -> ElevenLabs.
        """

        # ---- Select TTS Engine ----
        if self._user_language == "te":
            chosen_tts = self.azure_tts
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
