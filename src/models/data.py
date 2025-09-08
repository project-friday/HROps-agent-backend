from typing import Optional
from dataclasses import dataclass, field


@dataclass
class CandidateData:
    name: Optional[str] = None
    email: Optional[str] = None
    query: Optional[str] = None