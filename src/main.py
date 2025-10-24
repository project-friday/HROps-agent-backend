import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, room_io
from livekit.plugins import noise_cancellation

from src.agents.powercut import PowercutAgent
from src.agents.router import RouterAgent
from src.agents.translator import TranslatorAgent
from src.helpers.arg_parser import parse_cli_args
from src.models.data import UserData

# Load .env from repo root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY missing. Expected in {ROOT / '.env'}")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()

    tenant = os.getenv("TENANT_CLI_OVERRIDE", "walmart")
    flow = os.getenv("FLOW_CLI_OVERRIDE", "default")

    if tenant == "translator":
        # Start translator agent with the selected language flow
        from src.config.loader import get_cfg

        cfg = get_cfg()

        await session.start(
            agent=TranslatorAgent(cfg, ctx.room),
            room=ctx.room,
            room_input_options=room_io.RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            ),
            room_output_options=room_io.RoomOutputOptions(
                transcription_enabled=True, sync_transcription=True
            ),
        )
    elif tenant == "powercut":
        print("Starting in powercut mode")
        user_data = UserData(ctx=ctx)
        session.userdata = user_data
        await session.start(
            agent=PowercutAgent(room=ctx.room),
            room_input_options=room_io.RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            ),
            room=ctx.room,
        )
    else:
        print("Starting in router mode")
        await session.start(
            agent=RouterAgent(room=ctx.room),
            room_input_options=room_io.RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            ),
            room=ctx.room,
        )


if __name__ == "__main__":
    tenant, flow, sys.argv = parse_cli_args(sys.argv)
    os.environ["TENANT_CLI_OVERRIDE"] = tenant
    os.environ["FLOW_CLI_OVERRIDE"] = flow
    print(f"Tenant={tenant}, Flow={flow}")
    # Read telephony mode from .env (true/false)
    is_telephony = os.getenv("LIVEKIT_TELEPHONY_MODE", "false").lower() == "true"
    if is_telephony and tenant == "powercut":
        # Telephony mode → fixed agent name for SIP dispatch
        print("Starting in telephony mode for powercut agent")
        agents.cli.run_app(
            agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="powercut-agent")
        )
    else:
        # Default (normal) mode
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
