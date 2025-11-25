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


@function_tool(description="Record customer feedback given during the call.")
async def feedback_call_tool(rating: int, phone_number: str) -> dict:
    """UI-only tool – shows feedback rating on screen when user rates on the call."""
    rating = max(1, min(5, int(rating)))
    feedback = {
        "rating": f"{rating}/5",
        "phone_number": phone_number,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return feedback


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
