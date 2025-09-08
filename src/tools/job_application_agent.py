# hr/tools/job_application_agent.py

import re
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo
from livekit.agents import RunContext
# from src.agents.onboarding import OnboardingAgent
from livekit.agents import function_tool  # decorator used by the agent to call tools

# ---------- Constants & Helpers ----------

APPS_DIR = Path("data/applications")


# Professional, friendly phone-call style status lines
_STATUS_LINES: Dict[str, str] = {
    "submitted": "We’ve received your application and will begin our review shortly. No action needed from you right now.",
    "in_review": "Your application is currently under review by our recruiting team.",
    "interview_scheduled": "Your interview has been scheduled. Let me know if you'd like any preparation guidance",
    "selected": "Congratulations—your application has been selected. Our team will reach out with next steps.",
    "rejected": "Unfortunately, we won’t be moving forward with this application, but we appreciate your time. Thank you for your interest!"
}


def _normalize_email_for_filename(email: str) -> str:
    """Match your filename convention: lowercase; non-alphanumerics -> '_'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", (email or "").strip().lower())

def _humanize_status(raw: Optional[str]) -> str:
    return _STATUS_LINES.get((raw or "").strip().lower(), "We’re processing your application.")

def _humanize_updated_at(ts: Optional[str]) -> str:
    """Return a simple 'dd-mm-YYYY' string for speech; fall back to raw if parse fails."""
    if not ts:
        return ""
    try:
        # Accept '...Z' and no 'Z'
        dt = datetime.fromisoformat(str(ts).replace("Z", ""))
        return dt.strftime("%d-%m-%Y")
    except Exception:
        return str(ts)

# ---------- Interview helpers (new) ----------

def _load_record_by_application_id(application_id: str) -> tuple[dict, Path] | tuple[None, None]:
    """
    Scan data/applications for the given application_id and return (record, path).
    """
    # Try filename pattern first (fast path)
    fp = next(iter(APPS_DIR.glob(f"{application_id}_*.json")), None)
    if fp is None:
        # Fallback: scan contents if filenames don't follow pattern
        for cand in APPS_DIR.glob("*.json"):
            try:
                rec = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(rec.get("application_id", "")).strip().lower() == str(application_id).strip().lower():
                return rec, cand
        return None, None

    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        return rec, fp
    except Exception:
        return None, None


def _iso_to_dt(s: str) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _dt_to_iso(dt: datetime) -> str:
    # Store up to minutes (cleaner for speech + logs)
    return dt.isoformat(timespec="minutes")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_business_slot(dt: datetime, tz_name: str) -> bool:
    """
    Accept weekday (Mon–Fri) and 09:00–18:00 in interview's local timezone.
    dt must be timezone-aware.
    """
    try:
        local = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local = dt  # fallback if tz is weird

    if local.weekday() >= 5:  # 5=Sat, 6=Sun
        return False

    start = local.replace(hour=9, minute=0, second=0, microsecond=0)
    end   = local.replace(hour=18, minute=0, second=0, microsecond=0)
    return start <= local <= end


def _suggest_alternatives(base_dt: datetime, tz_name: str, count: int = 2) -> list[str]:
    """
    Offer up to 2 nearby business-hour options (10:00, 15:00 local).
    """
    alt: list[str] = []
    try:
        local = base_dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local = base_dt

    candidates: list[datetime] = []
    for hour in (10, 15):
        cand = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        # If in the past or weekend, roll forward to next weekday at same hour
        while cand <= _now_utc().astimezone(local.tzinfo) or cand.weekday() >= 5:
            cand = (cand + timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
        candidates.append(cand)

    for c in candidates:
        if _is_business_slot(c, tz_name):
            # Return in the same timezone/offset as base_dt
            alt.append(_dt_to_iso(c.astimezone(base_dt.tzinfo)))
        if len(alt) >= count:
            break
    return alt

# ---------- Tool: list all apps for an email (login + fan-out) ----------

@function_tool(description="""
Return the upcoming interview for an application, if any (scheduled or rescheduled in the future).
Use after the user has selected which application to manage.
""")
async def get_upcoming_interview(application_id: str) -> dict:
    """
    Returns:
      {
        "has_interview": bool,
        "interview": { ... } | {},
        "application_id": str
      }
    """
    rec, fp = _load_record_by_application_id(application_id)
    if not rec:
        return {"has_interview": False, "interview": {}, "application_id": application_id}

    itv = rec.get("interview") or {}
    status = str(itv.get("status") or "").lower()
    scheduled_at = _iso_to_dt(itv.get("scheduled_at"))
    if scheduled_at and scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)

    if status in {"scheduled", "rescheduled"} and scheduled_at and scheduled_at > _now_utc():
        return {
            "has_interview": True,
            "interview": {
                "status": status,
                "scheduled_at": _dt_to_iso(scheduled_at),
                "timezone": itv.get("timezone") or "Asia/Kolkata",
                "mode": itv.get("mode") or "video",
                "location_or_link": itv.get("location_or_link") or "",
                "interviewer": itv.get("interviewer") or "",
            },
            "application_id": application_id,
        }

    return {"has_interview": False, "interview": {}, "application_id": application_id}

@function_tool(description="""
Check if the interview panel is available at the proposed time for this application.
- Expects ISO-8601 with timezone offset (e.g., 2025-09-12T10:00:00+05:30).
- Simple rule: time must be in the future, on a weekday, and between 09:00–18:00 in the interview's timezone.
- Does NOT modify any data. Use this before asking the user to confirm booking.
""")
async def check_interview_availability(application_id: str, proposed_time_iso: str) -> dict:
    """
    Returns:
      {
        "ok": bool,
        "reason": str | null,
        "suggested": [iso, ...] | null,
        "normalized_time": str | null,   # ISO string for the proposed time, normalized
        "timezone": str | null
      }
    """
    rec, fp = _load_record_by_application_id(application_id)
    if not rec:
        return {
            "ok": False,
            "reason": "not_found",
            "suggested": None,
            "normalized_time": None,
            "timezone": None,
        }

    itv = rec.get("interview")
    if not itv:
        return {
            "ok": False,
            "reason": "no_interview",
            "suggested": None,
            "normalized_time": None,
            "timezone": None,
        }

    new_dt = _iso_to_dt(proposed_time_iso)
    if not new_dt:
        return {
            "ok": False,
            "reason": "invalid_time",
            "suggested": None,
            "normalized_time": None,
            "timezone": itv.get("timezone") or "Asia/Kolkata",
        }

    tz_name = itv.get("timezone") or "Asia/Kolkata"

    # Must be in the future
    if new_dt <= _now_utc():
        suggestions = _suggest_alternatives(
            _now_utc().astimezone(new_dt.tzinfo) + timedelta(hours=2),
            tz_name
        )
        return {
            "ok": False,
            "reason": "past_time",
            "suggested": suggestions,
            "normalized_time": None,
            "timezone": tz_name,
        }

    # Must be a business slot
    if not _is_business_slot(new_dt, tz_name):
        suggestions = _suggest_alternatives(new_dt, tz_name)
        return {
            "ok": False,
            "reason": "outside_business_hours",
            "suggested": suggestions,
            "normalized_time": None,
            "timezone": tz_name,
        }

    # Looks good—return normalized ISO (minutes precision) and tz
    return {
        "ok": True,
        "reason": None,
        "suggested": None,
        "normalized_time": _dt_to_iso(new_dt),
        "timezone": tz_name,
    }

@function_tool(description="""
Reschedule the interview for an application to a new time.
- Expects ISO-8601 with timezone offset (e.g., 2025-09-12T10:00:00+05:30).
- Simple availability check: future time, weekday, and 09:00–18:00 in the interview's timezone.
- On success: updates JSON in place and returns the updated interview.
- On failure: returns ok=false with 'reason' and optional 'suggested' alternatives.
""")
async def reschedule_interview(application_id: str, new_time_iso: str, reason: str = "candidate_request") -> dict:
    """
    Returns:
      {
        "ok": bool,
        "reason": str | null,
        "suggested": [iso, ...] | null,
        "interview": { ... } | {}
      }
    """
    rec, fp = _load_record_by_application_id(application_id)
    if not rec:
        return {"ok": False, "reason": "not_found", "suggested": None, "interview": {}}

    itv = rec.get("interview")
    if not itv:
        return {"ok": False, "reason": "no_interview", "suggested": None, "interview": {}}

    new_dt = _iso_to_dt(new_time_iso)
    if not new_dt:
        return {"ok": False, "reason": "invalid_time", "suggested": None, "interview": {}}

    tz_name = itv.get("timezone") or "Asia/Kolkata"

    # Must be in the future
    if new_dt <= _now_utc():
        suggestions = _suggest_alternatives(_now_utc().astimezone(new_dt.tzinfo) + timedelta(hours=2), tz_name)
        return {"ok": False, "reason": "past_time", "suggested": suggestions, "interview": {}}

    # Must be within business hours on a weekday
    if not _is_business_slot(new_dt, tz_name):
        suggestions = _suggest_alternatives(new_dt, tz_name)
        return {"ok": False, "reason": "outside_business_hours", "suggested": suggestions, "interview": {}}

    # Passed checks → update the JSON
    old_time = itv.get("scheduled_at")
    itv["scheduled_at"] = _dt_to_iso(new_dt)
    itv["status"] = "rescheduled"

    # Optional small audit trail
    trail = rec.get("reschedules")
    if not isinstance(trail, list):
        trail = []
    trail.append({
        "old_time": old_time,
        "new_time": itv["scheduled_at"],
        "changed_at": _dt_to_iso(_now_utc()),
        "reason": reason,
    })
    rec["reschedules"] = trail

    rec["updated_at"] = _dt_to_iso(_now_utc())

    try:
        fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        return {"ok": False, "reason": f"write_failed: {e}", "suggested": None, "interview": {}}

    return {
        "ok": True,
        "reason": None,
        "suggested": None,
        "interview": {
            "status": itv["status"],
            "scheduled_at": itv["scheduled_at"],
            "timezone": tz_name,
            "mode": itv.get("mode") or "video",
            "location_or_link": itv.get("location_or_link") or "",
            "interviewer": itv.get("interviewer") or "",
        },
    }


@function_tool(
    description="""
    Return all applications for a given email (case-insensitive).
    Matches files named: <application_id>_<normalized_email>.json in data/applications.

    Output:
    {
      "email": "<input email>",
      "name": "<first found full name or null>",
      "count": <int>,
      "applications": [
        {
          "application_id": "<id>",
          "job_title": "<title>",
          "status": "<submitted|in_review|interview_scheduled|selected|rejected>",
          "human_status": "<friendly sentence>",
          "updated_at": "<ISO8601>",
          "updated_at_human": "dd-mm-YYYY"
        }
      ]
    }
    """
)
async def list_applications_by_email(email: str) -> dict:
    """
    This doubles as the 'login fetch':
    - normalizes email per filename
    - returns a user-facing summary for 1-or-many apps
    """
    apps: List[Dict[str, Any]] = []
    inferred_name: Optional[str] = None

    email_norm = _normalize_email_for_filename(email)

    if APPS_DIR.exists():
        # Sort newest first by file mtime for nicer UX.
        files = sorted(
            APPS_DIR.glob(f"*_{email_norm}.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for fp in files:
            try:
                rec = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            status = (rec.get("status") or "submitted").strip().lower()
            human = _humanize_status(status)

            updated_at = rec.get("updated_at")
            if not updated_at:
                try:
                    updated_at = datetime.fromtimestamp(fp.stat().st_mtime).isoformat(timespec="seconds") + "Z"
                except Exception:
                    updated_at = None
            updated_at_human = _humanize_updated_at(updated_at)

            job_title = rec.get("job_title") or rec.get("title") or "Unknown role"

            nm = rec.get("name") or rec.get("full_name") or rec.get("applicant_name")
            if nm and not inferred_name:
                inferred_name = nm

            apps.append({
                "application_id": rec.get("application_id"),
                "job_title": job_title,
                "status": status,
                "human_status": human,
                "updated_at": updated_at,
                "updated_at_human": updated_at_human,
            })

    return {
        "email": email,
        "name": inferred_name,
        "count": len(apps),
        "applications": apps,
    }

@function_tool(
    description="""
    Given an application_id, return the application's current status in a human-friendly form.
    It looks for file: <application_id>_*.json in data/applications.

    Returns:
    {
      "found": true|false,
      "application": {
        "application_id": "...",
        "job_title": "...",
        "status": "submitted|in_review|interview_scheduled|selected|rejected",
        "human_status": "...",
        "updated_at": "YYYY-MM-DDTHH:MM:SSZ",
        "updated_at_human": "dd-mm-YYYY",
        "name": "Full Name or null",
        "email": "raw email",
        "phone": "raw phone or null",
        "response_timeframe": "natural text or null",
        "description": "free text or null"
      }
    }
    """
)
async def check_application_status(application_id: str) -> dict:
    # Locate the file by filename pattern first
    fp = next(iter(APPS_DIR.glob(f"{application_id}_*.json")), None)

    # Fallback: scan contents if filename didn’t match (handles manual naming)
    if fp is None:
        for cand in APPS_DIR.glob("*.json"):
            try:
                rec = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(rec.get("application_id", "")).strip().lower() == application_id.strip().lower():
                fp = cand
                break

    if fp is None or not fp.exists():
        return {"found": False, "application": None}

    # Read and shape the record
    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {"found": False, "application": None}

    status = (rec.get("status") or "submitted").strip().lower()
    human = _humanize_status(status)

    updated_at = rec.get("updated_at")
    if not updated_at:
        try:
            updated_at = datetime.fromtimestamp(fp.stat().st_mtime).isoformat(timespec="seconds") + "Z"
        except Exception:
            updated_at = None
    updated_at_human = _humanize_updated_at(updated_at)

    job_title = rec.get("job_title") or rec.get("title") or "Unknown role"

    return {
        "found": True,
        "application": {
            "application_id": rec.get("application_id"),
            "job_title": job_title,
            "status": status,
            "human_status": human,
            "updated_at": updated_at,
            "updated_at_human": updated_at_human,
            "name": rec.get("name"),
            "email": rec.get("email"),
            "phone": rec.get("phone"),
            "response_timeframe": rec.get("response_timeframe"),
            "description": rec.get("description"),
        }
    }


@function_tool(
    description="""
    Interpret a user's selection among multiple applications and return the chosen application.
    The user can reply with a number (as listed) or a job title (full or partial).

    Inputs:
      - email (str): The email used to fetch the applications.
      - choice (str): The user's reply, e.g., "2" or "Backend Engineer".

    Behavior:
      - Loads apps for the given email from files named <application_id>_<normalized_email>.json
      - If 'choice' is a number within range, selects that index (1-based)
      - Otherwise, matches by job title (case-insensitive), supporting exact, prefix, and substring matches

    Returns:
    {
      "matched": true|false,
      "ambiguous": true|false,
      "selection": { "index": <1-based>, "application_id": "...", "job_title": "..." } | null,
      "options": [ { "index": <1-based>, "application_id": "...", "job_title": "..." }, ... ]  // present when ambiguous or not matched
    }
    """
)
async def select_application_by_choice(email: str, choice: str) -> dict:
    email_norm = _normalize_email_for_filename(email)
    files = sorted(
        APPS_DIR.glob(f"*_{email_norm}.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Build the in-memory list (titles only for speech, but keep IDs)
    apps = []
    for fp in files:
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        apps.append({
            "application_id": rec.get("application_id"),
            "job_title": rec.get("job_title") or rec.get("title") or "Unknown role",
        })

    if not apps:
        return {"matched": False, "ambiguous": False, "selection": None, "options": []}

    # 1) Numeric selection (1-based)
    choice_str = (choice or "").strip()
    if choice_str.isdigit():
        idx = int(choice_str)
        if 1 <= idx <= len(apps):
            sel = apps[idx - 1]
            return {
                "matched": True,
                "ambiguous": False,
                "selection": {"index": idx, "application_id": sel["application_id"], "job_title": sel["job_title"]},
                "options": []
            }
        # fall through to text matching if out of range

    # 2) Title-based selection
    def _norm_text(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()

    norm_choice = _norm_text(choice_str)
    if not norm_choice:
        return {"matched": False, "ambiguous": False, "selection": None, "options": [
            {"index": i+1, "application_id": a["application_id"], "job_title": a["job_title"]} for i, a in enumerate(apps)
        ]}

    exact, prefix, contains, token = [], [], [], []

    # Build normalized titles once
    norm_titles = [_norm_text(a["job_title"]) for a in apps]
    choice_tokens = set(norm_choice.split())

    for i, (a, t) in enumerate(zip(apps, norm_titles)):
        title_tokens = set(t.split())
        if t == norm_choice:
            exact.append(i)
        elif t.startswith(norm_choice):
            prefix.append(i)
        elif norm_choice in t:
            contains.append(i)
        elif choice_tokens and choice_tokens.issubset(title_tokens):
            token.append(i)

    def pick(indices):
        if not indices:
            return None
        if len(indices) == 1:
            i = indices[0]
            return {
                "matched": True,
                "ambiguous": False,
                "selection": {"index": i+1, "application_id": apps[i]["application_id"], "job_title": apps[i]["job_title"]},
                "options": []
            }
        # ambiguous: return compact options
        opts = [{"index": i+1, "application_id": apps[i]["application_id"], "job_title": apps[i]["job_title"]} for i in indices][:5]
        return {"matched": False, "ambiguous": True, "selection": None, "options": opts}

    # Preference order: exact > prefix > contains > token
    for bucket in (exact, prefix, contains, token):
        res = pick(bucket)
        if res:
            return res

    # No match: return all options so Eve can re-ask politely
    return {
        "matched": False,
        "ambiguous": False,
        "selection": None,
        "options": [{"index": i+1, "application_id": a["application_id"], "job_title": a["job_title"]}
                    for i, a in enumerate(apps)]
    }

# ---------- Tool: query the FAQ knowledge base (RAG) ----------

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
                "error": f"retriever import failed: {e.__class__.__name__}"
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
            "error": f"retriever query failed: {e.__class__.__name__}"
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
                text = item.get("text") or item.get("content") or item.get("chunk") or ""
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
    stitched = stitched.strip() or "I couldn't find a clear answer in the knowledge base."

    return {
        "answer": stitched,
        "snippets": snippets
    }
