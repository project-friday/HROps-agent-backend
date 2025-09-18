from __future__ import annotations
import os, re
from pathlib import Path
import yaml

_CFG = None
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

def get_cfg() -> dict:
    """Load tenants.yaml once and return the active tenant config."""
    global _CFG
    if _CFG is not None:
        return _CFG
    path = Path(__file__).with_name("tenants.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tenant = os.getenv("TENANT", "walmart")
    cfg = {"tenant": tenant, **data["tenants"][tenant]}
    cfg.setdefault("enabled_agents", ["Applications", "Onboarding", "Assessment"])
    _CFG = cfg
    return _CFG

def render(text: str, ctx: dict | None = None) -> str:
    """Replace {{placeholders}} in text using ctx (defaults to active cfg)."""
    ctx = ctx or get_cfg()
    return _PLACEHOLDER.sub(lambda m: str(ctx.get(m.group(1), m.group(0))), text)
