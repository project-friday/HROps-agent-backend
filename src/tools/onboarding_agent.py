import os, re, json
from datetime import datetime
from typing import Dict, Any, Optional, List
from livekit.agents import function_tool
from pathlib import Path
from livekit.agents import RunContext
from src.utils.emails import send_email
import os 
from dotenv import load_dotenv

load_dotenv()
# from src.agents.job_application import JobApplicationAgent
# from src.utils.helpers import normalize_email


OFFERS_DIR = Path("data/offers")

def _normalize_for_filename(name: str, email: str) -> str:
    # email = normalize_email(email) or (email or "").strip().lower()
    safe_name = re.sub(r'[^a-z0-9]+', '_', name.lower())
    safe_email = re.sub(r'[^a-z0-9@]+', '_', email.lower())
    print(safe_email)
    return f"{safe_name}_{safe_email}.json"

def _load_candidate_record(name: str, email: str) -> Optional[Dict[str, Any]]:
    fn = _normalize_for_filename(name, email)
    print("Loading candidate record:", fn)
    fp = OFFERS_DIR / fn
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None



@function_tool(
    description="Check the offer status and ETA for sending the offer letter."
)
async def check_offer_status(name: str, email: str) -> dict:
    print("Checking offer status for:", name, email)
    rec = _load_candidate_record(name, email)
    print("Record loaded:", rec)
    if not rec:
        result = {"error": "No record found"}
    else:
        offer = rec.get("offer", {})
        result = {"status": offer.get("status"), "eta_hours": offer.get("eta_hours")}
    
    # Save result to shared file
    return result



@function_tool(
    description="Get the high-level summary of the candidate’s offer (title, level, base, variable, benefits, location, tentative joining)."
)
async def get_offer_summary(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    return rec.get("offer", {}).get("summary", {})



@function_tool(
    description="Confirm the candidate’s tentative joining date."
)
async def confirm_joining_date(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    
    return {"joining_date": rec.get("joining", {}).get("date")}


@function_tool(
    description="Get the reporting manager details (name, title, email, calendar link)."
)
async def get_reporting_manager(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        result = {"error": "No record found"}
    else:
        result = rec.get("reporting", {})
    
    # Save result to shared file
    return result



@function_tool(
    description="Return the list of documents the candidate must upload for pre-boarding."
)
async def get_documents_checklist(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        result = {"error": "No record found"}
    else:
        result = {"documents": rec.get("preboarding", {}).get("documents", [])}
    
    return result


@function_tool(
    description="Return the list of tasks the candidate must complete before joining."
)
async def get_preboarding_tasks(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}
    return {"tasks": rec.get("preboarding", {}).get("tasks", [])}
@function_tool(
    description="Return the candidate's background verification (BGV) status, expected completion days, and remarks."
)
async def get_background_verification_status(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    bgv = rec.get("bgv", {})
    return {
        "status": bgv.get("status", "unknown"),
        "expected_days": bgv.get("expected_days", ""),
        "remarks": bgv.get("remarks", "")
    }



 # your SES email sender


@function_tool(
    description="""
    Mark candidate’s joining deferral request with a new date.
    This will also notify the hiring team about the deferral request.
    """
)
async def mark_deferral(name: str, email: str, new_date: str, send: bool = True) -> dict:
    """
    new_date: must be in format YYYY-MM-DD
    ⚠️ Always call with `send=True` to actually notify the hiring team.
    """
    fn = _normalize_for_filename(name=name, email=email)
    fp = OFFERS_DIR / fn

    if not fp.exists():
        return {"error": "Offer record not found"}

    data = json.loads(fp.read_text(encoding="utf-8"))
    data.setdefault("escalations", {})
    data["escalations"]["joining_deferral"] = {
        "requested": True,
        "new_date": new_date,
    }

    fp.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Hiring team email
    hiring_subject = f"Deferral Request: {name}"
    hiring_body = (
        f"The candidate {name} ({email}) has requested to defer their joining date.\n\n"
        f"New requested date: {new_date}\n\n"
        "Please review and update the onboarding process accordingly."
    )
    hiring_team_email = os.getenv("ONBOARDING_TEAM_EMAIL")

    if send and hiring_team_email:
        try:
            send_email(hiring_team_email, hiring_subject, hiring_body)
        except Exception as e:
            return {
                "success": True,
                "message": "Deferral request submitted, but failed to notify hiring team.",
                "warning": str(e),
            }

    elif send and not hiring_team_email:
        return {
            "success": True,
            "message": "Deferral request submitted, but no hiring team email is configured.",
        }

    return {"success": True, "status": "Deferral request submitted"}


@function_tool(
    description="""
    Schedule or mark the 1:1 intro call with the reporting manager.
    """
)
async def schedule_intro_call(name: str, email: str, date: str) -> dict:
    """
    date: must be in format YYYY-MM-DDTHH:MM (ISO8601)
    """
    fn = _normalize_for_filename(name=name,email=email)

    fp = OFFERS_DIR / fn

    if not fp.exists():
        return {"error": "Offer record not found"}

    data = json.loads(fp.read_text(encoding="utf-8"))
    data.setdefault("reporting", {})
    data["reporting"]["intro_call_scheduled"] = True
    data["reporting"]["intro_call_date"] = date

    fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"success": True, "intro_call_date": date}

@function_tool(
    description="""
    Update or add candidate’s preferred laptop shipping address.
    """
)
async def update_shipping_address(name: str, email: str, address: str) -> dict:
    # name_norm = _normalize_for_filename(name)
    # email_norm = _normalize_for_filename(email)
    fn = _normalize_for_filename(name=name,email=email)
    fp = OFFERS_DIR / fn

    if not fp.exists():
        return {"error": "Offer record not found"}

    data = json.loads(fp.read_text(encoding="utf-8"))
    data.setdefault("it_assets", {})
    data["it_assets"]["preferred_shipping_address"] = address

    fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"success": True, "preferred_shipping_address": address}

# async def update_joining_date(name: str, email: str, new_date: str) -> dict:
#     """
#     new_date: must be in format YYYY-MM-DD
#     """
    
#     fn = _normalize_for_filename(name=name,email=email)
#     fp = OFFERS_DIR / fn
#     if not fp.exists():
#         return {"error": "Offer record not found"}

#     data = json.loads(fp.read_text(encoding="utf-8"))
#     data["offer"]["joining_date"] = new_date

#     fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
#     return {"success": True, "joining_date": new_date}

@function_tool(
    description="""
    Fetch candidate's offer details from stored JSON.
    Only call this for offer letter received candidates.
    Input: name + email (asked only once in session).
    Matches file: data/offers/<normalized_name>_<normalized_email>.json

    Output: full candidate record (offer, reporting, documents, preboarding, IT, etc.)
    """
)
async def get_offer_details(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        result = {"error": "No record found"}
    else:
        result = {"offer_letter":rec.get("offer")}
    
    return result



@function_tool(
    description="""
    Log a negotiation request from the candidate.
    This no longer stores the request in JSON, but directly notifies
    the compensation team via email.
    Returns a friendly confirmation message (not raw email details).
    """
)
async def log_negotiation(name: str, email: str, request: str, send: bool = True) -> dict:
    comp_team_email = os.getenv("ONBOARDING_TEAM_EMAIL")
    print("email:",comp_team_email)
    subject = f"Negotiation Request: {name}"
    body = (
        f"The candidate {name} ({email}) has submitted a negotiation request.\n\n"
        f"Request details:\n{request}\n\n"
        f"Timestamp: {datetime.utcnow().isoformat()}Z\n\n"
        "Please review this request and follow up as appropriate."
    )

    if send and comp_team_email:
        try:
            send_email(comp_team_email, subject, body)
            # print(response)
        except Exception as e:
            print("error sending email")
            return {
                "success": True,
                "message": "Negotiation request submitted, but failed to notify the compensation team.",
                "warning": str(e),
            }

    elif send and not comp_team_email:
        print("comp email not found")
        return {
            "success": True,
            "message": "Negotiation request submitted, but no compensation team email is configured.",
        }

    return {
        "success": True,
        "message": "Your request for negotiation has been shared with the onboarding team, thank you.",
        "status":"email sent to compensation team"
    }




@function_tool(
    description="""
    Email the candidate their required document checklist.
    This will send a secure email to the candidate’s registered email address with the list of documents.

    Always call this function with `send=True` to actually send the email.

    """
)
async def email_documents_checklist(name: str, email: str, send: bool = True) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    documents = rec.get("preboarding", {}).get("documents", [])
    if not documents:
        return {"error": "No documents checklist found for this candidate"}

    # Build plain text email body
    subject = "Your Preboarding Document Checklist"
    body = (
        f"Hello {name},\n\n"
        "Here is your preboarding document checklist:\n\n"
        + "\n".join(f"- {doc}" for doc in documents)
        + "\n\nPlease make sure to have these ready.\n\n"
        "Best regards,\n"
        "The HR Team"
    )

    if send:
        try:
            send_email(email, subject, body)  # use SES function
            return {
                "message": f"✅ I’ve just sent the document checklist to {email}. Please check your inbox (and spam folder just in case)."
            }
        except Exception as e:
            return {"error": f"Failed to send email: {str(e)}"}
    else:
        # Simulated mode
        return {
            "status": f"Sent checklist to {email}"
        }


@function_tool(
    description="""
    Send a summary email of the current onboarding conversation to the candidate.
    The summary text must be provided by the calling agent (not fetched from JSON).
    The email will always be sent to the candidate.
    """
)
async def send_onboarding_summary(name: str, email: str, conversation_summary: str, send: bool = True) -> dict:
    subject = f"Onboarding Plan – {name}"
    body = f"""
Hi {name},

Here's a quick recap of our conversation today:

{conversation_summary}

Excited to have you onboard! 🎉

Regards,  
HR Team
"""

    if send:
        try:
            send_email(email, subject, body)
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send onboarding summary to {email}.",
                "error": str(e),
            }

    return {
        "success": True,
        "message": f"✅ I've sent the conversation summary to {email}. Please check your inbox for '{subject}'",
        "status": f"Summary sent to {email}"
    }

@function_tool(
    description="""
    Retrieve the candidate's Day-1 agenda (orientation and activities planned on the first day).
    """
)
async def get_day1_agenda(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    day1_agenda = rec.get("it_assets", {}).get("day1_agenda", {})

    if not day1_agenda:
        return {"error": "No Day-1 agenda found for this candidate"}

    return {
        "day1_agenda": day1_agenda
    }

@function_tool(
    description="""
    Retrieve the candidate's IT asset provisioning details 
    (laptop shipping, email provisioning, VPN access, shipping address).
    """
)
async def get_it_assets(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    it_assets = rec.get("it_assets", {})

    if not it_assets:
        return {"error": "No IT assets details found for this candidate"}

    return {
        "laptop_shipping": it_assets.get("laptop_shipping"),
        "email_provisioning": it_assets.get("email_provisioning"),
        "vpn_access": it_assets.get("vpn_access"),
        "preferred_shipping_address": it_assets.get("preferred_shipping_address"),
    }

@function_tool(
    description="""
    Retrieve the candidate's assigned work location (e.g., office campus, building, or remote model).
    """
)
async def get_work_location(name: str, email: str) -> dict:
    rec = _load_candidate_record(name, email)
    if not rec:
        return {"error": "No record found"}

    location = rec.get("candidate", {}).get("location")
    work_model = rec.get("candidate", {}).get("work_model")

    if not location and not work_model:
        return {"error": "No work location details found for this candidate"}

    return {
        "work_location": location,
        "work_model": work_model
    }




@function_tool(
    description="""
    Escalate to the onboarding team by sending them an email.
    Use this when the candidate raises an onboarding-related request
    that the agent cannot handle directly, or when human support is required.
    """
)
async def escalate_to_onboarding_team(name: str, email: str, request: str, send: bool = True) -> dict:
    onboarding_team_email = os.getenv("ONBOARDING_TEAM_EMAIL")
    print("onboarding email:",onboarding_team_email)
    subject = f"Escalation Required: Onboarding Support for {name}"
    body = f"""
The candidate {name} ({email}) has raised a request that requires onboarding team intervention.

Request details:
{request}

Timestamp: {datetime.utcnow().isoformat()}Z

Please review this request and follow up with the candidate.
"""

    if send and onboarding_team_email:
        try:
            response= send_email(onboarding_team_email, subject, body)
            print(response)
        except Exception as e:
            return {
                "success": False,
                "message": "Escalation request captured, but failed to notify the onboarding team.",
                "error": str(e),
            }
    elif send and not onboarding_team_email:
        return {
            "success": False,
            "message": "Escalation request captured, but no onboarding team email is configured.",
        }

    return {
        "success": True,
        "message": "✅ I've shared your request with the onboarding team. They’ll follow up with you soon.",
        "status":"email sent to onboarding team"
    }
