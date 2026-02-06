"""
Finance Collection Agent for Emaar Properties.
Simplified version - works like other agents with inbound calls.
Customer data loaded from JSON file, randomly selected each session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Dict, Optional

from livekit import rtc
from livekit.agents import stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings

from livekit.plugins import cartesia, elevenlabs, silero, soniox

from src.config.loader import get_cfg

logger = logging.getLogger("finance-collection-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    """Load prompt template from file."""
    return Path(file_path).read_text(encoding="utf-8").strip()


def load_customers(file_path: str = "data/finance_customers.json") -> dict:
    """Load customer data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


FINANCE_PROMPT_TEMPLATE = load_prompt("src/prompts/finance_collection.txt")


def get_customer_by_id(customer_id: int) -> Optional[dict]:
    """Get customer data by ID from JSON file."""
    data = load_customers()
    for customer in data["customers"]:
        if customer["id"] == customer_id:
            return customer
    return None


def get_random_customer() -> dict:
    """Get a random customer from the JSON file for demo purposes."""
    data = load_customers()
    customer = random.choice(data["customers"])
    logger.info(f"Randomly selected customer: {customer['name']} (ID: {customer['id']})")
    return customer


def get_default_customer() -> dict:
    """Get the default customer for demo purposes."""
    data = load_customers()
    default_id = data.get("default_customer_id", 1)
    return get_customer_by_id(default_id) or data["customers"][0]


def format_prompt_with_customer(template: str, customer: dict) -> str:
    """Format the prompt template with customer data."""
    return template.format(
        customer_name=customer.get("name", "Sir/Madam"),
        phone_number=customer.get("phone", ""),
        property_name=customer.get("property_name", "your property"),
        unit_number=customer.get("unit_number", ""),
        amount_due=f"{customer.get('amount_due', 0):,.2f}",
        due_date=customer.get("due_date", "the due date"),
        grace_period_end=customer.get("grace_period_end", "30 days from due date"),
        installment_type=customer.get("installment_type", "installment"),
        commitment_date="{commitment_date}",
        payment_method="{payment_method}",
    )


class FinanceCollectionAgent(Agent):
    """
    Emaar Properties Finance Collection Agent.
    Simplified version for demo - inbound calls with JSON data.
    Randomly selects a customer from JSON each session.
    """

    def __init__(
        self,
        cfg: dict,
        room: rtc.Room,
        customer_id: Optional[int] = None,
        random_customer: bool = True,
        chat_ctx=None
    ) -> None:
        self.room = room
        self.cfg = cfg

        # Load customer data - random by default for demo
        if customer_id:
            self.customer = get_customer_by_id(customer_id) or get_random_customer()
        elif random_customer:
            self.customer = get_random_customer()
        else:
            self.customer = get_default_customer()

        logger.info("Loaded customer: %s (ID: %s)", self.customer['name'], self.customer['id'])

        # Format prompt with customer data
        formatted_prompt = format_prompt_with_customer(
            FINANCE_PROMPT_TEMPLATE,
            self.customer
        )

        # TTS setup
        tts_cfg = cfg.get("tts", {})
        voices = tts_cfg.get("voices", {})
        providers = tts_cfg.get("providers", {})

        # English TTS
        english_voice = voices.get("english", "H8bdWZHK2OgZwTN7ponr")
        if providers.get("english", "elevenlabs") == "elevenlabs":
            self.english_tts = elevenlabs.TTS(
                voice_id=english_voice,
                model="eleven_turbo_v2_5",
            )
        else:
            self.english_tts = cartesia.TTS(
                model=tts_cfg.get("model", "sonic-3"),
                voice=english_voice,
                language="en",
            )

        # Arabic TTS
        arabic_voice = voices.get("arabic", "6304c635-6681-4f9e-85b6-a97f4d26461a")
        self.arabic_tts = cartesia.TTS(
            model=tts_cfg.get("model", "sonic-3"),
            voice=arabic_voice,
            language="ar",
            volume=2,
        )

        super().__init__(
            instructions=formatted_prompt,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            tts=self.english_tts,
            chat_ctx=chat_ctx,
        )

        self.manual_language = None
        self._user_language = self.customer.get("preferred_language", "en")

    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Custom STT node with language detection."""
        activity = self._get_activity_or_raise()
        assert activity.stt is not None

        wrapped_stt = activity.stt
        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError("STT does not support streaming, add a VAD")
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
                        detected_lang = event.alternatives[0].language
                        if not self.manual_language:
                            self._user_language = detected_lang
                        else:
                            self._user_language = self.manual_language

                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """Dynamic TTS selection based on detected language."""
        lang = self.manual_language or getattr(self, "_user_language", "en")

        if lang.startswith("ar"):
            chosen_tts = self.arabic_tts
        else:
            activity = self._get_activity_or_raise()
            assert activity.tts is not None
            chosen_tts = activity.tts

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

            async def _forward_input():
                async for chunk in text:
                    stream.push_text(chunk)
                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await asyncio.wait([forward_task])

    async def on_enter(self):
        """Called when agent becomes active - greet the customer."""
        cfg = get_cfg()
        greeting = f"Hello? Hi, am I speaking with {self.customer['name']}?"
        await self.session.say(greeting)

    async def on_exit(self):
        """Called when session ends."""
        logger.info("Finance collection call ended")
