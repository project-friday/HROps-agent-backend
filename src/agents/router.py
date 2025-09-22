# src/agents/router.py
import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

# from custom.livekit.plugins import murfai
from livekit import rtc
from livekit.agents import Agent, RunContext, function_tool, llm, stt, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import deepgram, elevenlabs, openai, silero

from src.agents.assessment import AssessmentAgent
from src.agents.job_application import JobApplicationAgent
from src.agents.onboarding import OnboardingAgent
from src.utils.translator import translate_to_english


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


ROUTER_INSTRUCTIONS = load_prompt("src/prompts/router.txt")


class RouterAgent(Agent):
    def __init__(self, room: rtc.Room):
        self.room = room
        super().__init__(
            instructions=ROUTER_INSTRUCTIONS,
            #  stt=assemblyai.STT(language),
            stt=deepgram.STT(language="es"),
            llm=openai.LLM(model="gpt-4.1"),
            vad=silero.VAD.load(),
            tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                # voice_id="H8bdWZHK2OgZwTN7ponr",
                # voice_id="hHjbwzYZW17oh0p05AKv",
                voice_id="kjHz50TasdqbpbfK4uaN",
                model="eleven_turbo_v2_5",
                language="es",
            ),
            # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),
        )

    # ... existing init
    async def _translate_and_send_llm_response(
        self, raw_response: str, role: str, translate=True
    ) -> None:
        """
        Translate the final LLM response to English and send it to the WebSocket.
        Falls back to the raw response if translation fails.
        """
        if translate:
            try:
                translated_text = await translate_to_english(text=raw_response)
                print("🌍 Translated to English:", translated_text)
            except Exception as e:
                print(f"⚠️ Translation failed, sending raw text: {e}")
                translated_text = raw_response
        else:
            translated_text = raw_response
            print("🌍 Translation skipped, using raw text.")

        # 📤 Forward translated response to WebSocket
        try:
            await self.room.local_participant.send_text(
                json.dumps({"role": role, "translation": translated_text}),
                topic="lk.transcription",
            )
            print("📤 Translated response sent to WebSocket")
        except Exception as e:
            print(f"⚠️ Failed to forward translated response: {e}")

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Custom STT node that intercepts only finalized transcripts."""
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

            print("🎤 Launched audio forwarding task")
            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:

                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        if event.alternatives:
                            transcript = event.alternatives[0].text
                            print(f"📝 Final transcript: {transcript}")
                            if transcript or transcript != "":
                                # 📤 Forward finalized transcript to WebSocket
                                await self._translate_and_send_llm_response(
                                    transcript, "user"
                                )

                    # Always yield back into agent pipeline
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

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
                yield chunk

        # Capture final LLM response
        raw_response = "".join(buffer).strip()
        if raw_response:
            await self._translate_and_send_llm_response(raw_response, "bot")

    @function_tool
    async def go_onboarding(self, context: RunContext[dict]):
        agent = context.session.current_agent
        # Generic, smooth transition
        return (OnboardingAgent(room=agent.room, chat_ctx=context.session._chat_ctx),)

    @function_tool
    async def go_applications(self, context: RunContext[dict]):
        agent = context.session.current_agent
        return (
            JobApplicationAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
        )

    @function_tool
    async def go_assessment(self, context: RunContext[dict]):
        agent = context.session.current_agent
        return (AssessmentAgent(room=agent.room, chat_ctx=context.session._chat_ctx),)

    # --- speaks immediately after the router becomes active ---
    async def on_enter(self):
        await self.session.say(
            "Hola, soy Eve, del equipo de Adquisición de Talento de Walmart. ¿En qué puedo ayudarte?"
        )
        await self._translate_and_send_llm_response(
            "Hello, I am Eve from Walmart's Talent Acquisition team. How can I help you?",
            "bot",
            translate=False,
        )


# ______________________________________________________________________________________________________________#
