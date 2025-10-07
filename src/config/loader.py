from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml

_CFG = None
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


import os
from pathlib import Path

import yaml

_CFG = None  # Global cache


def get_cfg() -> dict:
    """Load tenants.yaml once and return the active tenant+flow config."""
    global _CFG
    if _CFG is not None:
        return _CFG

    path = Path(__file__).with_name("tenants.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    tenant = os.getenv("TENANT_CLI_OVERRIDE", "walmart")
    flow = os.getenv("FLOW_CLI_OVERRIDE", "default")
    lang = os.getenv("LANGUAGE_OVERRIDE", "english")
    accent = os.getenv("ACCENT_OVERRIDE", "indian")

    print(f"Using tenant config: {tenant}, flow: {flow}")

    tenants = data.get("tenants", {})
    if tenant not in tenants:
        raise ValueError(
            f"Unknown tenant '{tenant}'. Available tenants: {list(tenants.keys())}"
        )

    tenant_cfg = tenants[tenant]
    flow_cfg = tenant_cfg.get("flows", {}).get(flow, {})

    # --- Merge tenant-level and flow-level ---
    cfg = {**tenant_cfg, **flow_cfg}
    cfg["tenant"] = tenant
    cfg["flow"] = flow

    # --- Merge enabled_agents safely ---
    if "enabled_agents" not in cfg:
        if tenant == "translator":
            cfg["enabled_agents"] = ["Translator"]
        else:
            cfg["enabled_agents"] = ["Applications", "Onboarding", "Assessment"]

    # --- Tenant-level LLM config ---
    cfg["llm"] = tenant_cfg.get("llm", {"model": "gpt-4.1", "temperature": 0.1})

    # --- Language-level STT/TTS config ---
    language_cfg = tenant_cfg.get("language", {}).get(lang, {})
    cfg["stt"] = language_cfg.get(
        "stt", {"provider": "deepgram", "language_code": "en-US", "endpointing_ms": 200}
    )
    cfg["tts"] = language_cfg.get(
        "tts", {"provider": "elevenlabs", "model": "eleven_turbo_v2_5"}
    )

    # --- Voice ID based on language+accent ---
    cfg["voice_id"] = language_cfg.get("accent", {}).get(accent)
    if not cfg["voice_id"]:
        print(
            f"Warning: No voice ID found for language '{lang}' and accent '{accent}' "
            f"in tenant '{tenant}'."
        )

    cfg["language"] = lang
    cfg["accent"] = accent

    print(f"Using STT config:", cfg)

    _CFG = cfg
    return _CFG


def render(text: str, ctx: dict | None = None) -> str:
    """Replace {{placeholders}} in text using ctx (defaults to active cfg)."""
    ctx = ctx or get_cfg()
    return _PLACEHOLDER.sub(lambda m: str(ctx.get(m.group(1), m.group(0))), text)


def reset_cfg():
    """Clear cached config so it reloads next time get_cfg() is called."""
    global _CFG
    _CFG = None
