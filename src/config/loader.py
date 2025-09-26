from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml

_CFG = None
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def get_cfg() -> dict:
    """Load tenants.yaml once and return the active tenant+flow config."""
    global _CFG
    if _CFG is not None:
        return _CFG

    path = Path(__file__).with_name("tenants.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Get tenant and flow from environment (set in main.py)
    tenant = os.getenv("TENANT_CLI_OVERRIDE", "walmart")
    flow = os.getenv("FLOW_CLI_OVERRIDE", "default")
    print(f"Using tenant config: {tenant}, flow: {flow}")

    tenants = data.get("tenants", {})
    if tenant not in tenants:
        raise ValueError(
            f"Unknown tenant '{tenant}'. Available tenants: {list(tenants.keys())}"
        )

    tenant_cfg = tenants[tenant]

    # --- ADDITION FOR TRANSLATOR TENANT ---
    if tenant == "translator":
        # Each flow key is a language (hindi/spanish)
        flow_cfg = tenant_cfg.get("flows", {}).get(flow, {})
        cfg = {"tenant": tenant, "flow": flow, **flow_cfg}
        # Add enabled_agents if not present
        cfg.setdefault("enabled_agents", ["Translator"])
    else:
        # Merge tenant-level and flow-level enabled_agents
        flow_cfg = tenant_cfg.get("flows", {}).get(flow, {})
        enabled_agents = flow_cfg.get(
            "enabled_agents",
            tenant_cfg.get(
                "enabled_agents", ["Applications", "Onboarding", "Assessment"]
            ),
        )

        cfg = {"tenant": tenant, "flow": flow, **tenant_cfg}
        cfg["enabled_agents"] = enabled_agents

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
