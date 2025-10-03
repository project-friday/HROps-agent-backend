# src/agents/router.py
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent
from livekit.plugins import elevenlabs, openai, silero

from src.agents.assessment import AssessmentAgent
from src.agents.job_application import JobApplicationAgent

# At the top of router.py
from src.agents.onboarding import OnboardingAgent
from src.config.loader import get_cfg, render
from src.tools.handover import get_handover_tools
from src.utils.stt_config import make_deepgram_stt

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


def build_router_config() -> tuple[str, list]:
    """
    Build router instructions and tools dynamically
    based on active tenant config.
    Returns (instructions, tools).
    """
    cfg = get_cfg()
    enabled = set(cfg["enabled_agents"])

    tools_lines = []
    tools = get_handover_tools("router")

    if "Applications" in enabled:
        tools_lines.append(
            "- `go_applications` → application status/updates, application ID, job ID (JR-xxx), stages "
            "(submitted/in review/interview/selected/rejected) OR General HR FAQs via the knowledge base (RAG)."
        )

    if "Onboarding" in enabled:
        tools_lines.append(
            "- `go_onboarding` → offer letter, joining date/DOJ, pre-boarding, required documents, background check (BGV), "
            "reporting manager, location, workstation/laptop."
        )

    if "Assessments" in enabled:
        tools_lines.append(
            "- `go_assessment` → assessment related queries, issues and doubts"
        )

    tools_snippet = (
        f"You are **Eve** from the {cfg.get('company_name', cfg['tenant'].title())} Talent Acquisition Team as the **router**. "
        "Decide which domain should handle the user’s request and call exactly one transfer tool:\n"
        "Available transfer tools:\n"
        + ("\n".join(tools_lines) if tools_lines else "- (none)\n")
    )

    router_base = load_prompt("src/prompts/router.txt")
    router_instructions = tools_snippet + "\n\n" + render(router_base, cfg)

    return router_instructions, tools


class RouterAgent(Agent):
    def __init__(self, room: rtc.Room, chat_ctx=None):
        self.room = room

        router_instructions, tools = build_router_config()

        super().__init__(
            instructions=router_instructions,
            stt=make_deepgram_stt(language="en-US", endpointing_ms=200),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(),
            tts=elevenlabs.TTS(
                voice_id="xctasy8XvGp2cVO9HL9k",
                model="eleven_turbo_v2_5",
            ),
            chat_ctx=chat_ctx,
            tools=tools,
        )

    async def on_enter(self):
        cfg = get_cfg()
        await self.session.say(cfg["greeting"])
