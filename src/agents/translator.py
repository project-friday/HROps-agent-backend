import asyncio
from pathlib import Path

from livekit.agents import Agent
from livekit.plugins import elevenlabs, openai, silero

from src.config.loader import get_cfg, render
from src.utils.stt_config import make_deepgram_stt


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


class TranslatorAgent(Agent):
    def __init__(self, cfg: dict):
        """
        cfg: loaded from get_cfg() with keys: tts_voice, placeholder, etc.
        """
        self.cfg = cfg
        # Load system prompt and replace placeholder with target language
        prompt_text = load_prompt("src/prompts/translator.txt")
        prompt_text = render(prompt_text, ctx=cfg)
        print(f"Translator prompt:\n{prompt_text}\n")

        # Initialize Agent
        super().__init__(
            instructions=prompt_text,
            stt=make_deepgram_stt(
                language="multi", endpointing_ms=200, use_keyterms=False
            ),
            llm=openai.LLM(model="gpt-4.1", temperature=0.1),
            vad=silero.VAD.load(),
            tts=elevenlabs.TTS(voice_id=cfg["tts_voice"], model="eleven_turbo_v2_5"),
        )
