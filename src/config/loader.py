from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CFG = None
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def get_cfg() -> dict:
    """
    Load tenants.yaml once and return the active tenant + flow config.
    Supports:
      - Tenant-level (HSBC/Walmart style)
      - Agent-level (Emaar style) with per-agent STT/TTS/voices
    Each agent will include both english & arabic voice IDs if available.
    """
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

    # --- Merge tenant + flow config ---
    cfg = {**tenant_cfg, **flow_cfg}
    cfg["tenant"] = tenant
    cfg["flow"] = flow

    # --- Enabled agents ---
    enabled_agents = flow_cfg.get("enabled_agents") or tenant_cfg.get("enabled_agents")
    if not enabled_agents:
        enabled_agents = (
            ["Translator"]
            if tenant == "translator"
            else ["Applications", "Onboarding", "Assessment"]
        )
    cfg["enabled_agents"] = enabled_agents

    # --- Default tenant-level LLM ---
    cfg["llm"] = tenant_cfg.get("llm", {"model": "gpt-4.1", "temperature": 0.1})

    agents_cfg = {}

    # --- Agent-level configuration (Emaar-style) ---
    if "agents" in tenant_cfg:
        for agent_name in enabled_agents:
            agent_def = tenant_cfg["agents"].get(agent_name)
            if not agent_def:
                print(f"⚠️ Skipping undefined agent: {agent_name}")
                continue

            agent_cfg = {
                "name": agent_name,
                "llm": agent_def.get("llm", cfg["llm"]),
                "stt": agent_def.get(
                    "stt",
                    tenant_cfg.get(
                        "stt", {"provider": "deepgram", "language_code": "en-US"}
                    ),
                ),
                "tts": agent_def.get(
                    "tts",
                    tenant_cfg.get(
                        "tts", {"provider": "elevenlabs", "model": "eleven_turbo_v2_5"}
                    ),
                ),
                "greeting": agent_def.get("greeting", tenant_cfg.get("greeting", "")),
            }

            # --- Voice resolution ---
            tts_cfg = agent_cfg["tts"]
            voice_ids = {}

            if "voices" in tts_cfg:
                # Multi-voice setup (english + arabic)
                voice_ids = {
                    lang_key: voice_id
                    for lang_key, voice_id in tts_cfg["voices"].items()
                    if voice_id
                }
            elif "voice" in tts_cfg:
                # Single-voice setup
                voice_ids["default"] = tts_cfg["voice"]
            else:
                # Fallback to tenant-level voices
                tenant_voices = tenant_cfg.get("voices", {})
                if tenant_voices:
                    voice_ids = {
                        lang_key: voice_id
                        for lang_key, voice_id in tenant_voices.items()
                        if voice_id
                    }

            # Attach both voices
            agent_cfg["voice_ids"] = voice_ids
            # For backward compatibility, also keep default voice_id
            agent_cfg["voice_id"] = (
                voice_ids.get(accent)
                or voice_ids.get("english")
                or next(iter(voice_ids.values()), None)
            )

            agents_cfg[agent_name] = agent_cfg

        cfg["agents"] = agents_cfg

    else:
        # --- Legacy tenant-style (HSBC/Walmart) ---
        if "language" in tenant_cfg:
            language_cfg = tenant_cfg["language"].get(lang, {})
            cfg["stt"] = language_cfg.get(
                "stt",
                {
                    "provider": "deepgram",
                    "language_code": "en-US",
                    "endpointing_ms": 200,
                },
            )
            cfg["tts"] = language_cfg.get(
                "tts", {"provider": "elevenlabs", "model": "eleven_turbo_v2_5"}
            )
            cfg["voice_id"] = language_cfg.get("accent", {}).get(accent)
        else:
            cfg["stt"] = tenant_cfg.get(
                "stt",
                {
                    "provider": "deepgram",
                    "language_code": "en-US",
                    "endpointing_ms": 200,
                },
            )
            cfg["tts"] = tenant_cfg.get(
                "tts", {"provider": "elevenlabs", "model": "eleven_turbo_v2_5"}
            )
            voices = tenant_cfg.get("voices", {})
            cfg["voice_ids"] = voices
            cfg["voice_id"] = voices.get("english")

    if not cfg.get("voice_id") and not cfg.get("agents"):
        print(f"⚠️ Warning: No voice ID found for tenant '{tenant}'")

    cfg["language"] = lang
    cfg["accent"] = accent

    print(f"[Config Loaded] Tenant={tenant}, Flow={flow}, Agents={enabled_agents}")
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
