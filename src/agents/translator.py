import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterable, Dict

from livekit import rtc
from livekit.agents import Agent, llm, stt, tokenize, tts, utils
from livekit.agents.llm import ChatContext
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins import deepgram, elevenlabs, openai, silero

from src.config.loader import get_cfg, render
from src.utils.stt_config import make_deepgram_stt


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


class TranslatorAgent(Agent):
    def __init__(self, cfg: dict, room: rtc.Room) -> None:
        """
        cfg: loaded from get_cfg() with keys: tts_voice, placeholder, etc.
        """
        self.cfg = cfg
        self.room = room
        self._target_language = cfg.get("language_code")
        if not self._target_language:
            raise ValueError(
                f"Missing 'language_code' in translator flow config for flow '{cfg['flow']}'. "
                "Add it to tenants.yaml under the flow."
            )

        # Load system prompt and replace placeholder with target language
        prompt_text = load_prompt("src/prompts/translator.txt")
        prompt_text = render(prompt_text, ctx=cfg)
        print(f"Translator prompt:\n{prompt_text}\n")
        self._user_language = "en"  # default user language
        self.tts_english = elevenlabs.TTS(
            voice_id=cfg["english_voice"],  # hardcoded English voice
            model="eleven_turbo_v2_5",
        )
        self.tts_target = elevenlabs.TTS(
            voice_id=cfg["target_voice"],  # target language voice from YAML
            model="eleven_turbo_v2_5",
        )

        # Initialize Agent
        super().__init__(
            instructions=prompt_text,
            stt=make_deepgram_stt(
                language="multi", endpointing_ms=400, use_keyterms=False
            ),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(),
            tts=self.tts_english,  # default TTS, overridden in tts_node
        )

    def _get_response_language(self) -> str:
        user_lang = self._user_language
        target_lang = self._target_language

        if user_lang == target_lang:
            return "en"
        elif user_lang == "en":
            return target_lang
        else:
            # Unexpected language — default to translating to English
            print(
                f"⚠️ Unexpected user language: {user_lang}. Assuming it's {target_lang}."
            )
            return "en"

    async def _send_message(self, text: str, role: str, language: str):
        """Send message to WebSocket with language"""
        try:
            await self.room.local_participant.send_text(
                json.dumps(
                    {"role": role, "text": text, "language": language},
                    ensure_ascii=False,
                ),
                topic="lk.transcription",
            )
            print(f"📤 Sent {role} message [{language}]: {text[:50]}...")
        except Exception as e:
            print(f"⚠️ Failed to send message: {e}")

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
                            self._user_language = event.alternatives[0].language
                            transcript = event.alternatives[0].text
                            print(f"📝 Final transcript: {transcript}")
                            if transcript or transcript != "":
                                # 📤 Forward finalized transcript to WebSocket
                                asyncio.create_task(
                                    self._send_message(
                                        transcript, "user", self._user_language
                                    )
                                )

                    # Always yield back into agent pipeline
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Choose TTS instance based on the language determined by _get_response_language().
        """
        response_lang = self._get_response_language()
        chosen_tts = self.tts_english if response_lang == "en" else self.tts_target

        # Wrap non-streaming TTS if necessary
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
            bot_language = self._get_response_language()
            asyncio.create_task(self._send_message(raw_response, "bot", bot_language))
