import json
import os
import random

from livekit.agents import function_tool

DATA_PATH = "data/assesments"


def _normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("@", "_").replace(".", "_")


def _load_candidate_record(name: str, email: str) -> dict:
    fname = f"{_normalize(name)}_{_normalize(email)}.json"
    fpath = os.path.join(DATA_PATH, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r") as f:
        return json.load(f)


@function_tool(
    description="Create a case number for the candidate issue or resolution."
)
async def create_case_record(name: str, email: str, issue_type: str) -> dict:
    case_number = f"HRC{random.randint(10000000, 99999999)}"
    return {"success": True, "case_number": case_number, "issue_type": issue_type}


@function_tool(
    description="Get assessment details for the candidate (instructions, duration, etc.)."
)
async def get_assessment_details(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    return rec.get("assessment", {})


@function_tool(description="Check assessment status (done, pending, dispositioned).")
async def check_assessment_status(name: str, email: str) -> dict:
    def _format_status(status: str) -> str:
        """Normalize status: lowercase and replace spaces with underscores."""
        if not status:
            return status
        return status.lower().replace(" ", "_")

    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    a = rec.get("assessment")
    if not a:
        return {"error": "No assessment found"}

    return {
        "status": _format_status(a.get("status")),
        "title": a.get("title"),
    }


@function_tool(description="Reschedule candidate assessment. Returns new deadline.")
async def reschedule_assessment(name: str, email: str, new_time: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    a = rec.get("assessment")
    if not a:
        return {"error": "No assessment found"}
    a["deadline"] = new_time
    case = await create_case_record(name, email, "Assessment Rescheduled")
    return {
        "success": True,
        "new_deadline": new_time,
        "case_number": case["case_number"],
    }


@function_tool(description="Send a reminder about the assessment.")
async def send_assessment_reminder(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    a = rec.get("assessment")
    case = await create_case_record(name, email, "Assessment Reminder Sent")
    if not a:
        return {"error": "No assessment found"}
    return {
        "success": True,
        "message": f"Reminder sent for {a['title']}",
        "case_number": case["case_number"],
    }


@function_tool(description="Escalate issue to recruiter/assessment team.")
async def escalate_to_assessment_team(name: str, email: str, issue: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    case = await create_case_record(name, email, "Assessment Escalation")
    return {
        "success": True,
        "escalated_issue": issue,
        "case_number": case["case_number"],
    }


# from livekit.agents import function_tool


@function_tool(
    description="Get the candidate’s assessment result (traffic light model: Green, Yellow, Orange, Red)."
)
async def get_assessment_result(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    a = rec.get("assessment")
    if not a or "result" not in a:
        return {"error": "No assessment result found"}
    return {
        "result": a["result"],
        "score": a.get("score"),
        "submitted_at": a.get("submitted_at"),
    }
