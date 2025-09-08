# src/main.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from livekit.plugins import openai 

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, room_io
from livekit.plugins import noise_cancellation
from livekit import rtc
from src.agents.job_application import JobApplicationAgent
from src.agents.onboarding import OnboardingAgent

# Configure detailed logging
# logging.basicConfig(
#     level=logging.WARN,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
from livekit.plugins import silero, assemblyai, elevenlabs

# from src.agents.job_application import JobApplicationAgent
# from src.agents.onboarding import OnboardingAgent
from src.agents.router import RouterAgent
from src.models.data import CandidateData




# Load .env from repo root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY missing. Expected in {ROOT / '.env'}")

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session=AgentSession[CandidateData](
        userdata=CandidateData(),
    )
    # attach disconnect handler
    # @ctx.room.on("participant_disconnected")
    # async def _on_disconnected(_):
    #     print("Room disconnected → clearing session history")      # gracefully stop
    #     session._chat_ctx.empty()    # drop past conversation memory

    await session.start(
        agent=RouterAgent(room=ctx.room),
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))