# src/agents/router.py
from livekit.agents import Agent, function_tool, RunContext
from livekit import rtc
from src.agents.job_application import JobApplicationAgent
from src.agents.onboarding import OnboardingAgent
from livekit.plugins import assemblyai, elevenlabs,openai, silero
from src.utils.stt_config import make_deepgram_stt
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


ROUTER_INSTRUCTIONS = load_prompt("src/prompts/router.txt")

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
                        llm=openai.LLM(model="gpt-4.1"),
                        vad=silero.VAD.load(),
                         tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                voice_id="H8bdWZHK2OgZwTN7ponr",
                # model="eleven_multilingual_v2",
                model="eleven_turbo_v2_5",
            )

                         )

    @function_tool
    async def go_onboarding(self, context: RunContext[dict]):
        agent = context.session.current_agent
        # Generic, smooth transition
        return (
            OnboardingAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
        )

    @function_tool
    async def go_applications(self, context: RunContext[dict]):
        agent = context.session.current_agent
        return (
            JobApplicationAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
        )

    # --- speaks immediately after the router becomes active ---
    async def on_enter(self):
        await self.session.say("Hey there, I’m Eve, speaking from Walmart HR Department. How may I help you?")

    

#______________________________________________________________________________________________________________#