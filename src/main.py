import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, room_io
from livekit.plugins import elevenlabs, noise_cancellation, openai, silero, soniox, azure

from src.agents.router import RouterAgent
from src.agents.translator import TranslatorAgent
from src.helpers.arg_parser import parse_cli_args
from src.utils.stt_config import make_deepgram_stt

# --- Load environment variables ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY missing. Expected in {ROOT / '.env'}")


# --- Common helpers ---
async def create_translator_session(ctx: JobContext):
    """Start a session for the Translator Agent."""
    await ctx.connect()
    session = AgentSession()

    from src.config.loader import get_cfg

    cfg = get_cfg()  # includes tenant, flow, language, accent, voice_id

    await session.start(
        agent=TranslatorAgent(cfg, ctx.room),
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=room_io.RoomOutputOptions(
            transcription_enabled=True,
            sync_transcription=True,
        ),
    )


async def create_hr_session(ctx: JobContext):
    """Start a RouterAgent session with shared LLM, STT, VAD, and TTS."""
    from src.config.loader import get_cfg

    cfg = get_cfg()  # includes tenant, flow, language, accent, voice_id, llm, stt, tts

    # --- Load STT, LLM, VAD, and TTS from config ---
    stt_cfg = cfg["stt"]
    llm_cfg = cfg["llm"]
    tts_cfg = cfg["tts"]

    stt = make_deepgram_stt(
        language=stt_cfg.get("language_code"),
        endpointing_ms=stt_cfg.get("endpointing_ms"),
    )

    llm = openai.LLM(model=llm_cfg.get("model"), temperature=llm_cfg.get("temperature"))

    vad = silero.VAD.load()

    tts = elevenlabs.TTS(voice_id=cfg.get("voice_id"), model=tts_cfg.get("model"))

    await ctx.connect()
    session = AgentSession(
        stt=stt,
        llm=llm,
        vad=vad,
        tts=tts,
    )

    await session.start(
        agent=RouterAgent(room=ctx.room),
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )


# --- Emaar (Dubai Mall) support session ---
async def create_mallsupport_session(ctx: JobContext):
    """Start a session for the Emaar Dubai Mall Support Agent."""
    from src.agents.mallsupport import MallSupportAgent  # your custom agent
    from src.config.loader import get_cfg

    cfg = get_cfg()  # includes tenant config (no language section, has voices + llm)

    llm_cfg = cfg["llm"]
    tts_cfg = cfg["tts"]

    # Load LLM + VAD
    llm = openai.LLM(model=llm_cfg.get("model"), temperature=llm_cfg.get("temperature"))
    vad = silero.VAD.load()

    # For MallSupportAgent, you can dynamically switch voices (English/Arabic)
    # based on detection or user input inside the agent
    # We'll default to English voice at startup
    english_voice = cfg["voices"]["english"]
    tts = elevenlabs.TTS(voice_id=english_voice, model=tts_cfg.get("model"))

    await ctx.connect()
    session = AgentSession(
        llm=llm,
        vad=vad,
        tts=tts,
    )

    await session.start(
        agent=MallSupportAgent(cfg, ctx.room),
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )


# --- Main entrypoint ---
async def entrypoint(ctx: JobContext):
    tenant_env = os.getenv("TENANT_CLI_OVERRIDE", "walmart")

    if tenant_env == "translator":
        await create_translator_session(ctx)
    elif tenant_env == "emaar":
        await create_mallsupport_session(ctx)
    else:
        await create_hr_session(ctx)


if __name__ == "__main__":
    # --- Parse CLI arguments ---
    tenant, flow, language, accent, sys.argv = parse_cli_args(sys.argv)

    # --- Load overrides into environment ---
    os.environ["TENANT_CLI_OVERRIDE"] = tenant
    os.environ["FLOW_CLI_OVERRIDE"] = flow
    os.environ["LANGUAGE_OVERRIDE"] = language
    os.environ["ACCENT_OVERRIDE"] = accent

    print(f"Tenant={tenant}, Flow={flow}, Language={language}, Accent={accent}")

    # --- Launch worker ---
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
