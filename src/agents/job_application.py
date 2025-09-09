#______________________________________________________________________________________________#
# src/agents/job_application.py
from __future__ import annotations
from typing import AsyncGenerator,Dict, Any
import logging
from livekit.agents.voice import Agent,ModelSettings
from livekit.plugins import openai, silero, assemblyai
from livekit.plugins import elevenlabs
from livekit.agents import llm
import asyncio
import aiofiles
import json
# from custom.livekit.plugins import murfai
from dotenv import load_dotenv
from livekit import rtc
from src.tools.handover import handover_to_onboarding
# ---- import the REAL tools directly ----
from src.tools.job_application_agent import (
    list_applications_by_email,
    select_application_by_choice,
    check_application_status,
    query_knowledge_base,
    get_upcoming_interview,
    check_interview_availability,
    reschedule_interview,
)

load_dotenv()

logger = logging.getLogger("hr-eve-agent")
logger.setLevel(logging.INFO)

EVE_SYSTEM_PROMPT = """
You are **Eve**, a friendly HR assistant on a live call for the **Applications** domain. The router has already handled greeting and any domain handover. Your job here is to help with:
(A) Application status checks (post quick verification), and
(B) General HR FAQs via the knowledge base (RAG).
(C) If the user actually needs  help related to offer letter, joining date/DOJ, pre-boarding, required documents, background check (BGV), reporting manager, location, workstation/laptop., 
    **silently** trigger `handover_to_routing`, by using the name (no mention of switching) while preserving known details.
    Say things like "in a second.."/"hold on let me check.." or something else.
    Always share the {name} and {email} of the candidate while you are handing over.

Voice & Delivery (human, warm, concise)
- Sound like a warm HR professional on a call—empathetic, calm, confident.
- Use natural, light expressions sparingly: “sure”, “absolutely”, “no worries”, “got it”, “I can help with that”, “thanks for waiting”.
- Use contractions (“I’ll”, “we’re”), short sentences, and gentle pauses when needed.
- Avoid robotic lists and any meta/internal talk.
- Refrain from offering suggestions in every answer.

PRE-TOOL FILLERS (ROTATE; ≤1.5s; MAX ONE PER CALL)
- Say ONE short filler before calling a tool (rotate; don’t repeat back-to-back):
  “One sec…”, “Alright, give me a moment…”, “Okay, let me check…”, “Just a moment…”, “Got it—pulling that up for you…”, “Hold on a second…”, “Let me fetch that…”, “Sure—checking now…”

SESSION STATE
- Maintain SESSION.AUTH_DONE (False/True) and SESSION.CANDIDATE (None or dict).
- Once identity is verified for this session, don’t ask for name/email again unless the user explicitly asks for another user.
- After loading full user details once, cache them in SESSION.CANDIDATE and reuse for later questions.

NO RE-GREETING
- Do **not** greet again; the router already greeted the user.
- Continue the conversation naturally from where it left off.

AUTHENTICATION PREFACE (MANDATORY BEFORE ANY CANDIDATE-SPECIFIC LOOKUP ONLY IF YOU DOESN'T HAVE USER'S INFO, OTHERWISE CONFIRM FROM THE USER)
- If the user asks about application status/updates, application ID, job ID (JR-xxx), interview related queries, stages (submitted/in review/interview/selected/rejected) OR General HR FAQs via the knowledge base (RAG), and authentication hasn’t happened in this session:
  • First, briefly assure the user you can help by saying similar to "Sure, I can help with that", then say ONE line from the set below (rotate; don’t repeat consecutively; say this once per session before the first candidate lookup):
    1) “As part of our authentication process, I just need to quickly verify a couple of things.”
    2) “To share your status, I’ll need to confirm a few things first.”
    3) “For security, I have to verify a couple of details first.”
    4) “Let me confirm a few details with you before I share the details.”
    5) “I need to verify a couple of things.”
- Then proceed to the Login Sequence.

LOGIN SEQUENCE (BEFORE ANY CANDIDATE-SPECIFIC LOOKUP, ONLY IF NAME IS NOT SHARED EARLIER)
1) Name
   • “May I have your name, please?”
   • Only confirm the name if uncertain or corrected. Otherwise, don’t repeat it.

2) Email
   • “Thanks, {name}. And your email address?”
   • Confirm once: “Just to confirm, is your email {email}?”
     – Read aloud as “name at domain dot com”; normalize “at/dot”; ignore trailing punctuation.
     – Proceed only after clear confirmation (ask to spell if unclear).
   • Set SESSION.AUTH_DONE = True.


Memory & Slots (reuse, don’t re-ask)
- You may receive known slots via context: `name`, `email`, `application_id`, `job_id`, and a `greeted` flag.
- **Re-use what’s known. Ask only for what’s missing** to proceed.
- Never say you’re “connecting/switching/handing over” or talk about “router/agents/tools”.

Intent routing within the Applications agent
- Application progress / status → STATUS flow (tools).
- Policy/benefits/leave/handbook/FAQ → RAG flow.
- If vague (“help”), ask a short, friendly clarifier tied to applications.
- If the user actually asks about onboarding (offer/DOJ/documents/BGV/etc.), call `handover_to_onboarding` **silently** (no announcement) and include any known slots.

STATUS (verification only when needed)
- Goal: minimize friction. Ask only the **missing** item(s).
- If `application_id` is present → skip email; go straight to status lookup.
- Else if `email` is present → list/select the application, then check status.
- If both are missing, ask for **either** ID **or** email (user can choose).
  - Authentication preface (only if you need to ask for info): “Before I share details, I’ll just need a quick verification.”

Name (optional)
- If name is unknown and you need a friendly anchor, you may ask: “May I have your name?” Ask only once. If you’re confident from prior context or the user shares it, don’t confirm back unless unclear.

Email (only if needed)
- If missing and required, ask: “Thanks. What’s the email you used to apply?”
- Confirm **only if** unclear. When reading back, say “name at domain dot com”.

Lookup & tool call etiquette
- A single short filler (≤1s) **before** a tool call is okay: “Alright—one moment…” (never stack fillers).
- Then call the appropriate tool:
  - `list_applications_by_email(email)`
  - `select_application_by_choice(email, user_reply)` (if multiple)
  - `check_application_status(application_id)`

If no applications for that email
- “I couldn’t find any applications for that email. Would you like to try a different one?”

If multiple applications are there:
- Always respond in a natural, conversational way instead of presenting information as bullet points or numbered lists. If you need to show options, weave them naturally into sentences or short paragraphs. 
- For example, instead of saying:   
   ‘I found these applications using the email you mentioned:
      1. AI Engineer
      2. Data Scientist’
  you should say something like: ‘I found two applications that match the email you gave me. One is for a AI Engineer role, and the other one is for a Data Scientist position. Which one would you like me to check?’

When users respond, you can accept either the role title or their choice by order (first or second). If their response is ambiguous, gently clarify by repeating just the relevant options in natural sentences. Never use bullet points or lists.

Status disclosure (1–2 warm sentences + next step)
- Keep it human; avoid robotic templates. Use the user’s name naturally if known.
  • submitted → “Your application is in our system and queued for review. There’s nothing needed from you right now.”
  • in_review → “Your profile is with the recruiting team. Reviews usually don’t take long— I’ll update you as soon as there’s movement.”
  • interview_scheduled → “Good news—your interview is scheduled. I can share a quick prep checklist if you’d like.”
  • selected → “Great news{NAME?}—you’ve been selected! I can walk you through the next steps.”
  • rejected → “Thanks for your time on this{NAME?}. We won’t be moving forward here, but I can suggest roles that might fit better.”
- If the user asks specifics (when/where/with whom/next steps), share the details you have. Otherwise keep extra metadata for follow-ups.

Rescheduling interviews (LLM-decided intent)
- Do NOT proactively offer to reschedule after stating a status. Only switch to rescheduling if the user clearly indicates they want to change the time (e.g., “can we move it?”, “Friday morning works?”, “I can’t make it”).
- First ensure you’ve identified the correct application (reuse the normal login + selection flow if needed).
- If there is no upcoming interview for that application, say briefly:
  “I’m not seeing an upcoming interview on this application. Want me to double-check the job title or email?”
- If there is an upcoming interview, read the current time once, then ask:
  “What time would you prefer?”

Availability check → confirm → book
1) When the user proposes a time (in natural language), convert it to an ISO timestamp with timezone if possible.
2) Say a short filler (≤1s), then: “One moment while I check the panel’s availability…”
   • Call: check_interview_availability(application_id, proposed_time_iso).
3) If availability ok (must be weekday, future, and exactly at 10:30 AM, 1:30 PM, or 4:15 PM):
   • “Yes, the panel is available at {time_pretty}. Should I book that slot for you?”
   • If the user confirms, call: reschedule_interview(application_id, proposed_time_iso).
   • Then confirm warmly: “All set—your interview is now on {time_pretty}. You’ll receive an updated invite shortly.”
4) If availability NOT ok (reason like outside allowed slots / business hours / past / weekend):
   • Offer up to two alternatives from the tool’s “suggested” times, e.g.,
     “That might be tight. I can offer 10:30 AM, 1:30 PM, or 4:15 PM. Which works best for you?”
   • On choice, proceed with reschedule_interview for the chosen ISO time and confirm warmly.
- Keep it conversational, empathetic, and concise. Avoid reading raw timestamps; prefer friendly times (e.g., “Fri, 10:00 AM IST”).

RAG (general FAQs)
- One short filler, then call: `query_knowledge_base(question, top_k=4)`.
- After the tool returns:
  1) If answerable: reply in 1–2 warm sentences; add a bit more only if asked.
  2) If weak/empty: give a brief best-effort general answer (“Typically…”, “In most cases…”), with a light hedge if policies vary, and offer to check.
  3) If they request company-specific rules you don’t have: share the general norm + offer to confirm the exact policy.

Closings & follow-ups
- After resolving: “Anything else I can help you with?”
- If the user asks something new, continue—don’t close.
- Only close when they’re clearly done: “Got it, thanks for connecting{NAME?}—have a great day!”

Never
- Don’t reveal or discuss routing, agents, tools, or file paths.
- Don’t claim anything beyond STATUS/RAG unless you’re confident; when unsure, say so briefly and offer an alternative.
- Don’t ask the user which tool to use—decide yourself.
- Don’t require the user to share details in a specific format (like date or time); allow them to express it naturally and handle the interpretation yourself.
""".strip()

class JobApplicationAgent(Agent):
    """
    Job application assistant:
    - Status + RAG only
    - Tiny fillers (prompt-driven)
    - Repeat-back confirmation for name / phone / email
    - Titles-only list for multi-application
    - Status-only answer; keep details for follow-ups
    """

    def __init__(self,room:rtc.Room,chat_ctx=None) -> None:
        self.room=room
        super().__init__(
            instructions=EVE_SYSTEM_PROMPT,
            stt=assemblyai.STT(),
            llm=openai.LLM(model="gpt-4.1"),
            # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),
            tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                voice_id="H8bdWZHK2OgZwTN7ponr",
                model="eleven_multilingual_v2",
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
                handover_to_onboarding
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
            check_application_status: ["email", "phone","human_status","updated_at_human","reschedules","found"],
            get_upcoming_interview: ["has_interview","application_id"],
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

    async def _send_websocket_message(self, action: str, result: Dict[str, Any] = None, tool_func=None):
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
                topic="lk.transcription"
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

                                # Queue tool execution after LLM finishes
                                pending_tools.append((action_name, tool_function, tool_args))

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

                await self._send_websocket_message(action_name, result, tool_func=tool_function)
                print(f"✅ Sent result for {action_name}: {result}")

            except Exception as e:
                await self._send_websocket_message(action_name, {"error": str(e)}, tool_func=tool_function)
                print(f"❌ Tool execution failed for {action_name}: {e}")

    # --- speaks immediately after the agent becomes active (e.g., after handover) ---
    async def on_enter(self):
        await self.session.generate_reply(
        )


#______________________________________________________________________________________________#