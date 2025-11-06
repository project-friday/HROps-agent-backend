# src/agents/mallsupport_agent.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from livekit import rtc
from livekit.agents import llm, stt, tokenize, tts, utils
from livekit.agents.stt import SpeechEventType
from livekit.agents.voice import Agent, ModelSettings

# Plugins
from livekit.plugins import azure, elevenlabs, openai, silero, soniox, cartesia  # noqa: F401

from src.config.loader import get_cfg
from src.tools.handover import handover_to_survey

# ---- Import only the knowledge base tool ----
from src.tools.mallsupport_tools import create_ticket, query_knowledge_base, feedback_sms_tool

logger = logging.getLogger("dubai-mall-support-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


EVE_MALL_PROMPT = load_prompt("src/prompts/mallsupport.txt")

# --- Language inference from assistant output (silent) ---
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")

# --- Lightweight guardrail patterns ---
RE_OTP = re.compile(r"\b(otp|one[-\s]?time\s?password|verification code|passcode)\b", re.I)
RE_PASSWORD = re.compile(r"\b(password|pin|cvv|security code)\b", re.I)
RE_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # crude; never echo full numbers
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"\+?\d[\d\s-]{6,}\d")
RE_SELF_HARM = re.compile(r"\b(kill myself|suicide|end my life|self[-\s]?harm)\b", re.I)
RE_MEDICAL = re.compile(r"\b(diagnose|prescribe|medicine|treat|treatment|dose|dosage)\b", re.I)
RE_LEGAL = re.compile(r"\b(legal advice|lawsuit|sue|attorney|contract)\b", re.I)
RE_FINANCIAL = re.compile(r"\b(invest|loan advice|stock tip|financial advice)\b", re.I)
RE_HARASS = re.compile(r"\b(stupid|idiot|shut up|hate|racist|sexist)\b", re.I)


def _redact(text: str) -> str:
    """Redact obvious PII before logging or emitting over WS."""
    if not text:
        return text
    text = RE_EMAIL.sub("[redacted-email]", text)
    text = RE_PHONE.sub("[redacted-phone]", text)
    text = RE_CARD.sub("[redacted-card]", text)
    return text


class MallSupportAgent(Agent):
    """
    Dubai Mall Customer Support Agent:
    - Silent LLM-based language selection (en/ar/zh) inferred from assistant text.
    - Guardrails with localized short replies.
    - PII redaction in websocket/tool payloads.
    - RAG tools + survey handover + SMS feedback preserved.
    """

    def __init__(self, cfg: dict, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = get_cfg()
        # ---------- TTS setup (multi-language) ----------
        tts_cfg = cfg.get("tts", {})
        voices = tts_cfg.get("voices", {})
        providers = tts_cfg.get("providers", {})

        # ---- English (Cartesia) ----
        english_voice = voices.get("english")
        self.english_tts = None
        if english_voice and providers.get("english", "cartesia") == "cartesia":
            self.english_tts = cartesia.TTS(
                model=tts_cfg.get("model", "sonic-3"),
                voice=english_voice,
                language="en",
                speed=0.9,
            )

        # ---- Arabic (Cartesia) ----
        arabic_voice = voices.get("arabic")
        if arabic_voice and providers.get("arabic", "cartesia") == "cartesia":
            self.arabic_tts = cartesia.TTS(
                model=tts_cfg.get("model", "sonic-3"),
                voice=arabic_voice,
                language="ar",
            )
        else:
            raise RuntimeError("❌ No Arabic Cartesia voice configured")

        # ---- Mandarin (Inworld) ----
        mandarin_voice = voices.get("mandarin")
        self.mandarin_tts = None
        if mandarin_voice:
            if providers.get("mandarin") == "inworld":
                from livekit.plugins import inworld
                self.mandarin_tts = inworld.TTS(voice=mandarin_voice)

        # super(): multilingual STT + VAD; no explicit TTS here (we route dynamically)
        super().__init__(
            instructions=EVE_MALL_PROMPT,
            stt=soniox.STT(
                params=soniox.STTOptions(language_hints=["en", "ar", "zh"]),
                vad=silero.VAD.load(min_speech_duration=0.1),
            ),
            tools=[query_knowledge_base, create_ticket, handover_to_survey, feedback_sms_tool],
            chat_ctx=chat_ctx,
        )

        # conversation language chosen implicitly by LLM (default English)
        self._current_lang: str = "en"
        self._harass_warnings = 0

        self.actions = {
            "Query Knowledge Base": query_knowledge_base,
            "Creating ticket": create_ticket,
            "Handover to Survey": handover_to_survey,
            "Send Feedback SMS": feedback_sms_tool,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optional: keys to filter from tool results before sending
        self.tool_result_filters = {
            query_knowledge_base: ["internal_id", "metadata"],
            create_ticket: ["ticket_id"],
            feedback_sms_tool: [],
        }

        self.tool_cards = {
            query_knowledge_base: "knowledge_base_result",
            create_ticket: "ticket_creation_result",
            handover_to_survey: "survey_handover",
            feedback_sms_tool: "feedback_sms_sent",
        }

        self.visible_tools = {create_ticket, feedback_sms_tool}

    # -------------- Guardrails helpers --------------
    def _last_user_text(self, chat_ctx: llm.ChatContext) -> str:
        try:
            msgs = getattr(chat_ctx, "messages", None) or []
            for m in reversed(msgs):
                if getattr(m, "role", "") == "user":
                    return str(getattr(m, "content", "")).strip()
        except Exception:
            pass
        return ""

    def _safe_template(self, category: str) -> str:
        lang = self._current_lang or "en"
        if category == "self_harm":
            return (
                "I’m really sorry you’re feeling this way. Please seek immediate help from a professional or local emergency services. Inside the mall, the nearest Concierge Desk can assist right away."
                if lang == "en" else
                "أنا آسف لسماع ذلك. يُفضّل التواصل فوراً مع جهة مختصة أو الطوارئ. داخل المول، توجه لأقرب مكتب كونسييرج للمساعدة."
                if lang == "ar" else
                "很抱歉听到你的情况。请尽快联系专业人员或本地急救。商场内可就近咨询礼宾台获得帮助。"
            )
        if category == "risky_advice":
            return (
                "I can’t advise on that, but I can help with Dubai Mall information. Would you like directions or store details?"
                if lang == "en" else
                "لا أستطيع تقديم نصائح طبية أو قانونية أو مالية. أستطيع مساعدتك بمعلومات دبي مول. هل ترغب بتوجيه داخل المول؟"
                if lang == "ar" else
                "我不能提供医疗、法律或财务建议，但可以帮助你了解迪拜购物中心的信息。需要我指引吗？"
            )
        if category == "credentials":
            return (
                "For your safety, I can’t handle codes or passwords. I can still help with directions or services at the mall."
                if lang == "en" else
                "لأمانك، لا أتعامل مع كلمات المرور أو الرموز. يسعدني مساعدتك بخدمات المول."
                if lang == "ar" else
                "为保障安全，我不能处理验证码或密码。我可以协助你在商场的服务与指引。"
            )
        if category == "harass":
            base = (
                "Let’s keep the conversation respectful. I’m here to help with Dubai Mall information."
                if lang == "en" else
                "دعنا نحافظ على الاحترام. أنا هنا لمساعدتك بمعلومات دبي مول."
                if lang == "ar" else
                "请保持礼貌交流。我在这里为你提供商场相关信息。"
            )
            if self._harass_warnings >= 1:
                base += (
                    " If this continues, I may end the conversation."
                    if lang == "en" else
                    " إذا استمر ذلك، قد أنهي المحادثة."
                    if lang == "ar" else
                    " 如果继续这样，我可能会结束对话。"
                )
            return base
        return (
            "I can help with Dubai Mall information. What would you like to know?"
            if lang == "en" else
            "أستطيع مساعدتك بمعلومات دبي مول. ما الذي ترغب بمعرفته؟"
            if lang == "ar" else
            "我可以帮助你了解迪拜购物中心的信息。你想了解什么？"
        )

    def _moderate(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        if RE_SELF_HARM.search(t):
            return "self_harm"
        if RE_OTP.search(t) or RE_PASSWORD.search(t) or RE_CARD.search(t):
            return "credentials"
        if RE_MEDICAL.search(t) or RE_LEGAL.search(t) or RE_FINANCIAL.search(t):
            return "risky_advice"
        if RE_HARASS.search(t):
            return "harass"
        return None

    # -------------- WebSocket message with redaction --------------
    async def _send_websocket_message(
        self, action: str, result: Dict[str, Any] = None, tool_func=None
    ):
        """Send WebSocket message with action, filtered result, and card_name (with redaction)."""
        message = {"action": action}

        if result is not None:
            safe = json.loads(json.dumps(result))
            if isinstance(safe, dict):
                for k, v in list(safe.items()):
                    if isinstance(v, str):
                        safe[k] = _redact(v)

            if tool_func in self.tool_result_filters:
                for key in self.tool_result_filters[tool_func]:
                    safe.pop(key, None)

            message["result"] = safe
            message["card_name"] = self.tool_cards.get(tool_func, "generic")

        try:
            await self.room.local_participant.send_text(
                json.dumps(message), topic="lk.transcription"
            )
            logger.info("Sent WebSocket message: %s", message)
        except Exception as e:
            logger.error("Failed to send WebSocket message: %s", e)

    # -------------- LLM node (silent lang + guardrails) --------------
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        """Custom LLM node that captures full response text, applies guardrails, infers language, and executes tools."""

        # Guardrails: intercept unsafe requests with localized short reply
        try:
            last_user = self._last_user_text(chat_ctx)
            violation = self._moderate(last_user)
        except Exception:
            violation = None

        if violation:
            if violation == "harass":
                self._harass_warnings += 1
            msg = self._safe_template(violation)
            self.last_llm_response = msg
            yield msg
            return

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

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Invalid JSON for %s: %s", tool_name, tool_args
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

        # Capture final LLM response (as spoken to the user)
        full_out = "".join(buffer).strip()
        self.last_llm_response = full_out

        # --- Silent language inference from assistant text ---
        detected = "en"
        if ARABIC_RE.search(full_out):
            detected = "ar"
        elif CJK_RE.search(full_out):
            detected = "zh"

        if detected != self._current_lang:
            self._current_lang = detected
            logger.info("Silent LLM language selection → %s", self._current_lang)

        # Execute queued tools and send results
        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                logger.info("Skipping execution of %s (not visible)", action_name)
                continue

            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)

                await self._send_websocket_message(
                    action_name, result, tool_func=tool_function
                )
                logger.info("Sent result for %s: %s", action_name, result)

            except Exception as e:
                await self._send_websocket_message(
                    action_name, {"error": str(e)}, tool_func=tool_function
                )
                logger.error("Tool execution failed for %s: %s", action_name, e)

    # -------------- STT node (no switching; hints only) --------------
    async def stt_node(
        self,
        audio: AsyncGenerator[rtc.AudioFrame, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Keep STT multilingual; do NOT change language based on STT. Optionally pass a hint from LLM-selected lang."""
        print("🎤 Starting STT node for mall support... (silent LLM lang switch)")

        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node available"

        wrapped_stt = activity.stt
        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError(
                    f"The STT ({activity.stt.label}) does not support streaming, add a VAD"
                )
            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        # Best-effort hint (if the STT implementation supports it)
        try:
            if hasattr(wrapped_stt, "set_language_hints"):
                wrapped_stt.set_language_hints([self._current_lang])
        except Exception:
            pass

        conn_options = activity.session.conn_options.stt_conn_options
        async with wrapped_stt.stream(conn_options=conn_options) as stream:

            @utils.log_exceptions()
            async def _forward_input() -> None:
                async for frame in audio:
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    # We no longer change language based on STT detection.
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    # -------------- TTS node (routes by LLM-selected lang) --------------
    async def tts_node(
        self,
        text: AsyncGenerator[str, None],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """
        Dynamically choose TTS voice based on LLM-selected language:
        - 'ar' → Arabic Cartesia TTS
        - 'zh' → Mandarin Inworld TTS
        - default → English Cartesia TTS (fallback to activity.tts if not configured)
        """
        lang = getattr(self, "_current_lang", "en")
        if lang.startswith("ar"):
            print("🗣️ Using Arabic TTS voice")
            chosen_tts = self.arabic_tts
        elif lang.startswith("zh"):
            print("🗣️ Using Mandarin TTS voice")
            chosen_tts = self.mandarin_tts
        else:
            chosen_tts = self.english_tts
            if chosen_tts is None:
                activity = self._get_activity_or_raise()
                assert activity.tts is not None, "tts_node called but no TTS node is available"
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
        cfg = get_cfg()
        """Speaks immediately after the agent becomes active."""
        await self.session.say(cfg["greeting"])
