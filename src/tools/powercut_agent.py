import json
import os
import time

from livekit.agents import function_tool

BASE_DIR = "data/powercut"


def _read_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@function_tool(description="Fetch customer details by USC (usc_number).")
async def get_customer_details(usc_number: str) -> dict:
    path = os.path.join(BASE_DIR, f"usc_{usc_number}.json")
    data = _read_json(path)
    if not data:
        return {"found": False, "message": f"No customer found for USC {usc_number}"}
    return data


@function_tool(description="Return outage info for area_code.")
async def get_outage_details(area_code: str) -> dict:
    path = os.path.join(BASE_DIR, f"outage_{area_code}.json")
    data = _read_json(path)
    if not data:
        return {"area_code": area_code, "status": "No Outage"}
    return data


@function_tool(description="Infer area_code from address/district/panchayat.")
async def get_area_mapping(district: str, panchayat: str, address: str) -> dict:
    text = " ".join([district or "", panchayat or "", address or ""]).lower()
    if "warangal" in text:
        return {"area_code": "WGL-110"}
    elif "hyderabad" in text:
        return {"area_code": "HYD-209"}
    return {"area_code": "UNKNOWN"}


@function_tool(description="Create service ticket for localized issue or hazard.")
async def schedule_service(address: str, issue_description: str) -> dict:
    ticket_id = time.strftime("ticket_%Y-%m-%d_%H-%M-%S")
    ticket = {
        "ticket_id": ticket_id,
        "address": address,
        "issue_description": issue_description,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "Emergency" if "EMERGENCY" in issue_description.upper() else "Open",
    }
    path = os.path.join(BASE_DIR, f"{ticket_id}.json")
    _write_json(path, ticket)
    return {"success": True, "ticket_id": ticket_id}


import json
import os
import time

from livekit.agents import function_tool

BASE_DIR = "data/powercut"


def _read_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@function_tool(description="Fetch customer details by USC (usc_number).")
async def get_customer_details(usc_number: str) -> dict:
    path = os.path.join(BASE_DIR, f"usc_{usc_number}.json")
    data = _read_json(path)
    if not data:
        return {"found": False, "message": f"No customer found for USC {usc_number}"}
    return data


@function_tool(description="Return outage info for area_code.")
async def get_outage_details(area_code: str) -> dict:
    path = os.path.join(BASE_DIR, f"outage_{area_code}.json")
    data = _read_json(path)
    if not data:
        return {"area_code": area_code, "status": "No Outage"}
    return data


@function_tool(description="Infer area_code from address/district/panchayat.")
async def get_area_mapping(district: str, panchayat: str, address: str) -> dict:
    text = " ".join([district or "", panchayat or "", address or ""]).lower()
    if "warangal" in text:
        return {"area_code": "WGL-110"}
    elif "hyderabad" in text:
        return {"area_code": "HYD-209"}
    return {"area_code": "UNKNOWN"}


@function_tool(description="Create service ticket for localized issue or hazard.")
async def schedule_service(address: str, issue_description: str) -> dict:
    ticket_id = time.strftime("ticket_%Y-%m-%d_%H-%M-%S")
    ticket = {
        "ticket_id": ticket_id,
        "address": address,
        "issue_description": issue_description,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "Emergency" if "EMERGENCY" in issue_description.upper() else "Open",
    }
    path = os.path.join(BASE_DIR, f"{ticket_id}.json")
    _write_json(path, ticket)
    return {"success": True, "ticket_id": ticket_id}


@function_tool(description="Send SMS notification with ticket id and content.")
async def send_sms(phone_number: str, ticket_id: str, message: str) -> dict:
    """
    Simulates sending an SMS. Stores log in BASE_DIR/sms_logs.json
    """
    sms_data = {
        "phone_number": phone_number,
        "ticket_id": ticket_id,
        "message": message,
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # In real case: integrate with Twilio, AWS SNS, etc.
    return sms_data
