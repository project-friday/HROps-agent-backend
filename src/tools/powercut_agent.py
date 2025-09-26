# src/tools/powercut_agent.py


async def get_customer_details(uac_number: str) -> dict:
    """Fetch customer details by UAC number."""
    # Mock response - replace with DB/API lookup
    return {
        "name": "Ramesh Kumar",
        "age": 42,
        "address": "123 Main Street, West Panchayat, Trichy",
        "area_code": "TNY-045",
    }


async def get_outage_details(area_code: str) -> dict:
    """Check outage info for a given area code."""
    outages = {
        "TNY-045": {"reason": "Transformer failure", "restoration_time": "2 hours"},
        "CHN-101": {"reason": "Scheduled maintenance", "restoration_time": "5 PM"},
    }
    return outages.get(area_code, {"reason": None, "restoration_time": None})


async def get_area_mapping(district: str, panchayat: str, address: str) -> dict:
    """Resolve district+panchayat+address → area code."""
    # Mock mapping
    if "trichy" in district.lower():
        return {"area_code": "TNY-045"}
    elif "chennai" in district.lower():
        return {"area_code": "CHN-101"}
    return {"area_code": "UNKNOWN"}


async def schedule_service(address: str, issue_description: str) -> dict:
    """Schedule a service visit if no outage is found."""
    return {
        "success": True,
        "ticket_id": "SRV-23910",
        "scheduled_time": "Tomorrow 10 AM",
        "address": address,
        "issue": issue_description,
    }
