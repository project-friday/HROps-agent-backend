from livekit.agents import function_tool


@function_tool(
    description="""
    Submit feedback from the user.
    """
)
async def record_feedback(feedback: str) -> dict:
    return {"message": f"feedback:{feedback} submitted successfully"}
