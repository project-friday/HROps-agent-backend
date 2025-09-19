# src/main.py
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, room_io
from livekit.plugins import assemblyai, noise_cancellation

from src.agents.router import RouterAgent
from src.models.data import CandidateData

# --- NEW: stash tenant arg early, before cli.run_app ---
TENANT = "walmart"
if "--tenant" in sys.argv:
    print("Found --tenant arg, overriding default tenant")
    idx = sys.argv.index("--tenant")
    if idx + 1 < len(sys.argv):
        TENANT = sys.argv[idx + 1]
        print(f"Using tenant: {TENANT}")
        # strip these args so LiveKit's CLI doesn't get confused
        sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]

# Make tenant globally available
os.environ["TENANT_CLI_OVERRIDE"] = TENANT
print(f"{os.environ['TENANT_CLI_OVERRIDE']}")


# Load .env from repo root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY missing. Expected in {ROOT / '.env'}")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()

    await session.start(
        agent=RouterAgent(room=ctx.room),
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
