from dataclasses import dataclass, field
from typing import Optional

from livekit.agents import JobContext


@dataclass
class CandidateData:
    name: Optional[str] = None
    email: Optional[str] = None
    query: Optional[str] = None


@dataclass
class UserData:
    """Store user data and state for the phone assistant."""

    ctx: Optional[JobContext] = None
