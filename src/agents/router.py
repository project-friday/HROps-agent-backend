# src/agents/router.py
from livekit.agents import Agent, function_tool, RunContext
from livekit import rtc
from src.agents.job_application import JobApplicationAgent
from src.agents.onboarding import OnboardingAgent
from livekit.plugins import assemblyai, elevenlabs,openai, silero
from src.utils.stt_config import make_deepgram_stt
from dotenv import load_dotenv
from pathlib import Path
from src.config.loader import get_cfg, render

load_dotenv()

CFG = get_cfg()
ENABLED = set(CFG["enabled_agents"])


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


ROUTER_INSTRUCTIONS = render(load_prompt("src/prompts/router.txt"))

# Add a runtime snippet so the LLM only sees brand-allowed tools
tools_lines = []
if "Applications" in ENABLED:
    tools_lines.append("- `go_applications` → application status/updates, application ID, job ID (JR-xxx), stages (submitted/in review/interview/selected/rejected) OR General HR FAQs via the knowledge base (RAG).")
if "Onboarding" in ENABLED:
    tools_lines.append("- `go_onboarding`  → offer letter, joining date/DOJ, pre-boarding, required documents, background check (BGV), reporting manager, location, workstation/laptop.")

TOOLS_SNIPPET = (
    "You are **Eve** from the {{company_name}} Talent Acquisition Team as the **router**. Decide which domain should handle the user’s request and call exactly one transfer tool:\n"
    "Available transfer tools:\n" + ("\n".join(tools_lines) if tools_lines else "- (none)\n")
)

ROUTER_INSTRUCTIONS = TOOLS_SNIPPET + ROUTER_INSTRUCTIONS


class RouterAgent(Agent):
    def __init__(self,room:rtc.Room):
        self.room=room
        super().__init__(instructions=ROUTER_INSTRUCTIONS,
                        #  stt=assemblyai.STT(),
                        stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
                        # stt=openai.STT(
                        #    model="gpt-4o-transcribe",
                        #    language="en",          # force English
                        #    detect_language=False   # disable auto language detection
                        # ),
                        llm=openai.LLM(model="gpt-4.1",temperature=0.1),
                        vad=silero.VAD.load(),
                         tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                voice_id="H8bdWZHK2OgZwTN7ponr",
                # model="eleven_multilingual_v2",
                model="eleven_turbo_v2_5",
            )

                         )

    # expose Onboarding only if enabled
    if "Onboarding" in ENABLED:
        @function_tool
        async def go_onboarding(self, context: RunContext[dict]):
            agent = context.session.current_agent
            # Generic, smooth transition
            return (
                OnboardingAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
            )

    # expose Applications only if enabled
    if "Applications" in ENABLED:
        @function_tool
        async def go_applications(self, context: RunContext[dict]):
            agent = context.session.current_agent
            return (
                JobApplicationAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
            )
     
    # # Walmart-only: Assessment
    # if "Assessment" in ENABLED:
    #     from src.agents.assessment import AssessmentAgent

    #     @function_tool
    #     async def go_assessment(self, context: RunContext[dict]):
    #         agent = context.session.current_agent
    #         return (
    #             AssessmentAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
    #         )

    # --- speaks immediately after the router becomes active ---
    async def on_enter(self):
        await self.session.say(get_cfg()["greeting"])

    

#______________________________________________________________________________________________________________#