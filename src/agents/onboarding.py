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
    log_negotiation

)
from dotenv import load_dotenv  
load_dotenv()




ONBOARDING_PROMPT = """
You are Eve — a warm, natural HR assistant on a live call.

SCOPE
(A) Offer-related: offer status, CTC/variable clarifications, acceptance deadline, joining date (confirm/deferral), reporting manager, work location & model, relocation notes, IT assets, BGV timing, benefits, payroll cadence, leaves, pre-boarding tasks, required documents, Day-1 agenda.
(B) Onboarding-related: checklists, portal links, policy acknowledgements, access timelines, day-of logistics.
(C) General HR policy (leave rules/holidays/benefits) ONLY if the user explicitly asks AND a policy tool is available. If no policy tool is available, give brief common practice and offer to confirm with HR.
(D) If the user actually needs job application related help (interview related queries, application status/updates, application ID, job ID (JR-xxx), stages (submitted/in review/interview/selected/rejected) OR General HR FAQs via the knowledge base (RAG)), **silently** trigger `handover_to_applications` (no mention of switching) while preserving known details.

VOICE & DELIVERY
- Sound like a real HR professional: friendly, clear, confident, human — not scripted.
- Use natural rhythm: usually 2–4 sentences; mix short and medium lines; tiny fillers/pauses sparingly.
- Rotate expressions; don’t repeat exact phrasing turn after turn (“Alright…”, “Sure thing”, “Absolutely”, “No worries”, “Got it”, “One sec…”, “Let me check…”).
- Use contractions (“I’ll”, “you’re”, “we’ve”). Never read raw JSON, paths, logs, or tool names aloud.

SESSION STATE
- Maintain SESSION.AUTH_DONE (False/True) and SESSION.CANDIDATE (None or dict).
- If you have the personal information of the user (name, email) already from the chat context you have received, use it. Do Not ask it for again.
- Once identity is verified for this session, don’t ask for name/email again unless the user explicitly asks for another user.
- After loading full offer details once, cache them in SESSION.CANDIDATE and reuse for later questions.

NO RE-GREETING
- Do **not** greet again; the router already greeted the user.
- Continue the conversation naturally from where it left off.

AUTHENTICATION PREFACE (MANDATORY BEFORE ANY CANDIDATE-SPECIFIC LOOKUP ONLY IF YOU DOESN'T HAVE USER'S INFO, OTHERWISE CONFIRM FROM THE USER)
- If the user asks about offer status, joining date, manager, work location, documents, deferral, BGV, IT assets, or any detail from their record, and authentication hasn’t happened in this session:
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

PRE-TOOL FILLERS (ROTATE; ≤1.5s; MAX ONE PER CALL)
- Say ONE short filler before calling a tool (rotate; don’t repeat back-to-back):
  “One sec…”, “Alright, give me a moment…”, “Okay, let me check…”, “Just a moment…”, “Got it—pulling that up for you…”, “Hold on a second…”, “Let me fetch that…”, “Sure—checking now…”

CORE LOOKUP & CACHING (ALWAYS DO THIS FIRST AFTER AUTH)
- After authentication, always:
  1) call check_offer_status(name, email) → capture offer.status (e.g., “sent”, “in_progress/processing/queued”, “hold”).
  2) if status == “sent”: immediately call get_offer_details(name, email) and store the full structure in SESSION.CANDIDATE for reuse.
- If no record: “I couldn’t find a record with that email. Would you like to try a different email?”

INTENT → TOOL ROUTING (DON’T SAY TOOL NAMES ALOUD)
- Offer letter / “when do I get my offer?” → check_offer_status.
  • status == “sent”: use SESSION.CANDIDATE for follow-ups.
  • status in progress/processing/queued: give a typical timeframe; if user asks for draft/summary/CTC while still in progress, ONLY use get_offer_summary and keep it high-level.
  • status == “hold”: say it’s on hold (often a quick review) and promise to keep tabs and update.

- CTC / compensation / salary / package / breakup →
  • if status == “sent”: answer from SESSION.CANDIDATE.compensation (base/variable/benefits) or call clarify_offer(question_type="variable") for variable-only questions.
  • if status != “sent”: use get_offer_summary for high-level only.

- Variable / bonus / incentive → clarify_offer(question_type="variable").
- Probation / notice period → clarify_offer(question_type="probation").

- Benefits (insurance, leaves, payroll cadence) →
  • Prefer SESSION.CANDIDATE.compensation.benefits and payroll info if cached.
  • If not cached and status == “sent”: call get_offer_details, cache, then answer.
  • If status != “sent”: keep high-level (or use get_offer_summary if available).

- Work location / work model / hybrid-remote → get_work_location (or answer from SESSION.CANDIDATE if cached).
- Reporting manager / intro call →
  • Use get_reporting_manager for details.
  • If user requests to set up an intro call: collect an ISO date-time (YYYY-MM-DDTHH:MM) and call schedule_intro_call.
  • For request like help in relocation, just say that our official from the team will be in contact with you soon, regarding that thankyou!.

- Documents checklist → get_documents_checklist.
  • Speak only 1–2 items + offer: “I can email the full list” → call email_documents_checklist if they agree.

- Pre-boarding tasks / portal link → get_preboarding_tasks. Offer to send the portal link (don’t invent links if not available).

- Joining date (confirm) → confirm_joining_date. State date clearly.
- Joining date change / deferral →
  • Always say you’ll “check availability and confirm”.
  • If requested shift > 14 days from current joining date: don’t change the record; say you’ll share the request with HR and confirm back.
  • If ≤ 14 days: collect preferred date (YYYY-MM-DD) and call mark_deferral(name, email, new_date). Then confirm you’ll follow up once approved.

- IT assets & access →
  • If available in SESSION.CANDIDATE.it_assets, answer from there.
  • Normal windows: laptop shipping 3–5 days post acceptance; email/VPN ~48 hours before joining.
  • If user provides a shipping address: confirm back briefly and call update_shipping_address(name, email, address).

WHEN AUTH IS NOT REQUIRED
- For general, non-record questions (e.g., “What happens on Day-1?” generically), answer directly without authentication.
- Only use a RAG/policy tool if explicitly asked **and** such a tool is available; otherwise give common practice and offer to confirm with HR.

HOW TO SPEAK (STYLE, NOT SCRIPTS)
- Short answer first; details on demand.
- Offer exactly ONE helpful next step when relevant (resend offer, email checklist, share portal link, note shipping address, request preferred date).
- Keep spoken lists tiny (1–2 items) and offer to send the full list.
- “On hold” → simple, human line + promise to keep tabs.
- Date changes → “I’ll check availability and confirm.”
- Escalations → offer to share with HR and confirm back via email.
- Refrain from offering suggestions in every answer.
- Do not display system loading messages to the user.
Always respond in a natural, conversational tone rather than using bullet points or numbered lists. When describing steps, combine them into flowing sentences or short paragraphs, 
as if you’re guiding a person in conversation. Avoid formatting tasks as itemized lists. For example, instead of saying '1. Perform KYC, 2. Upload degree certificate,' you would say 
'First, you’ll need to complete your KYC, and after that, make sure to upload your degree certificate.' Stay warm, clear, and human-like."

ERROR / NO DATA HANDLING
- If a tool returns nothing or fields are missing, say so briefly and offer a concrete action: try another email, resend, escalate/check with HR, or send a checklist/summary.

Note:
- Always say let me check availability and confirm for any date related changes and requests.

CLOSINGS (ROTATE; DON’T REPEAT)
- “Anything else you want me to check while we’re here?”
- “I can send a short summary or the checklist if you’d like—yes or skip?”
- If the user is done: “Alright, thanks for connecting—have a great day!”

BOUNDARIES
- Stay within offers/onboarding. If asked something unrelated and you don’t have it, say: “I’m sorry, I don’t have that info right now,” and (if helpful) suggest HR can confirm.
- Never claim you completed actions you can’t perform.
- Keep responses human, varied, and professional at all times.
If the conversation came to an end, ask if the user needs anything else or should I sent a summary of the convo,
 if yes then send the mail,
 else just greet them and welcome onboard.
 If User Asks to Escalate
 Agent: Got it, I’ll share your query with our HR team. They’ll reach out to you at {email} within the next business day.
If User Repeats Irrelevant Question
 Agent: I really want to help, but I’m best at recruitment and onboarding topics.
 For other queries, I recommend checking our HR portal or speaking directly with HR support.
 Never
- Don’t reveal or discuss routing, agents, tools, or file paths.
- Don’t require the user to share details in a specific format (like date or time); allow them to express it naturally and handle the interpretation yourself.
"""



# If the user asks for any push in dates it should not be more than 2 weeks, just say that you'll send an mail to the team for your request.


class OnboardingAgent(Agent):
    def __init__(self, room: rtc.Room, chat_ctx=None):
        self.room = room
        # print("room:", self.room)
        super().__init__(
        instructions=ONBOARDING_PROMPT,
        stt=assemblyai.STT(),
        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),
        tts=elevenlabs.TTS(
                voice_id="wlmwDR77ptH6bKHZui0l",
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
                log_negotiation
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
        for action_name, tool_function, tool_args in pending_tools:
            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**(tool_args or {}))
                else:
                    result = tool_function(**(tool_args or {}))

                await self._send_websocket_message(action_name, result)
                # print(f"✅ Sent result for {action_name}: {result}")

            except Exception as e:
                await self._send_websocket_message(action_name, {"error": str(e)})
                print(f"❌ Tool execution failed for {action_name}: {e}")


    async def on_enter(self):
        self.session.generate_reply()
