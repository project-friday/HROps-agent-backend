# src/main.py
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, room_io
from livekit.plugins import noise_cancellation

from src.agents.router import RouterAgent
from src.helpers.arg_parser import parse_cli_args

# Make tenant & flow globally available
TENANT, FLOW, sys.argv = parse_cli_args(sys.argv)

# Make tenant & flow globally available
os.environ["TENANT_CLI_OVERRIDE"] = TENANT
os.environ["FLOW_CLI_OVERRIDE"] = FLOW
print(f"Tenant={TENANT}, Flow={FLOW}")


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
