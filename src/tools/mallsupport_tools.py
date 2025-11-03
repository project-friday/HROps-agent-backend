import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# from src.agents.onboarding import OnboardingAgent
from livekit.agents import function_tool  # decorator used by the agent to call tools
from livekit.agents import RunContext

from src.utils.emails import send_email  # SES email sender

load_dotenv()


@function_tool(
    description="""
    Answer general HR questions from the internal FAQ knowledge base (RAG).
    Use this for policy, onboarding, benefits, leave, documents, etc.
    Returns a concise answer plus supporting snippets for transparency.
    """
)
async def query_knowledge_base(question: str, top_k: int = 4) -> dict:
    """
    Thin wrapper over your existing retriever. It supports two setups:

    1) Function style:
         from src.utils.retriever import query as rag_query
         texts = rag_query(question, top_k=top_k)

    2) Class style:
         from src.utils.retriever import Retriever
         retriever = Retriever(index_dir="storage/faq_index")
         texts = retriever.query(question, top_k=top_k)

    It normalizes outputs into:
      {
        "answer": "<concise stitched text>",
        "snippets": [{"text": "...", "source": "<file or url or page ref>"}]
      }
    """
    # Lazy import so the file loads even if retriever deps aren't ready
    retriever_mode = None
    rag_query = None
    retriever = None

    try:
        # prefer function-style if available
        from src.utils.retriever import query as _q  # type: ignore

        rag_query = _q
        retriever_mode = "func"
    except Exception:
        try:
            from src.utils.retriever import Retriever  # type: ignore

            retriever = Retriever(index_dir="storage/faq_index")
            retriever_mode = "class"
        except Exception as e:
            return {
                "answer": "Sorry, I can't access the knowledge base right now.",
                "snippets": [],
                "error": f"retriever import failed: {e.__class__.__name__}",
            }

    # Run the search
    try:
        if retriever_mode == "func":
            results = rag_query(question, top_k=top_k)  # could be str or list-like
        else:
            results = retriever.query(question, top_k=top_k)
    except Exception as e:
        return {
            "answer": "Sorry, I had trouble searching the knowledge base.",
            "snippets": [],
            "error": f"retriever query failed: {e.__class__.__name__}",
        }

    # Normalize to a list of snippet dicts
    snippets = []
    if isinstance(results, str):
        # Some implementations return a single stitched string
        snippets = [{"text": results, "source": None}]
    elif isinstance(results, list):
        # Expect list of strings or dicts
        for item in results[: top_k or 4]:
            if isinstance(item, str):
                snippets.append({"text": item, "source": None})
            elif isinstance(item, dict):
                text = (
                    item.get("text") or item.get("content") or item.get("chunk") or ""
                )
                src = (
                    item.get("source")
                    or item.get("file")
                    or item.get("doc")
                    or item.get("metadata", {}).get("source")
                    or item.get("metadata", {}).get("file_name")
                )
                snippets.append({"text": text, "source": src})
    else:
        # Unknown format; best effort
        snippets = [{"text": str(results), "source": None}]

    # Build a short, speakable answer by stitching first few snippets
    stitched = " ".join(s["text"].strip() for s in snippets if s.get("text"))[:800]
    stitched = (
        stitched.strip() or "I couldn't find a clear answer in the knowledge base."
    )

    return {"answer": stitched, "snippets": snippets}


@function_tool(description="Create ticket for any issue reported by customer.")
async def create_ticket(phone_number: int, issue_description: str) -> dict:
    ticket_id = time.strftime("ticket_%Y-%m-%d_%H-%M-%S")
    ticket = {
        "ticket_id": ticket_id,
        "phone": phone_number,
        "issue_description": issue_description,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "Emergency" if "EMERGENCY" in issue_description.upper() else "Open",
    }

    return ticket

@function_tool(description="Send feedback SMS link when customer declines to rate on the call.")
async def feedback_sms_tool(phone_number: str) -> dict:
    """UI-only tool – shows confirmation on screen when feedback SMS is sent."""
    link = "https://feedback.thedubaimall.ae/rate"
    sms = {
        "phone_number": phone_number,
        "message": (
            f"Dubai Mall Concierge: Thank you for calling! We value your feedback. "
            f"Please rate your experience here: {link}"
        ),
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return sms