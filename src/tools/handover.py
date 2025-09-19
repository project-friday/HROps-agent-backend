from livekit.agents import RunContext, function_tool


@function_tool
async def handover_to_onboarding(
    context: RunContext[dict],
    name: str | None = None,
    email: str | None = None,
):
    from src.agents.onboarding import OnboardingAgent  # avoid circular import

    s = context.session
    s.state = {**getattr(s, "state", {}), "name": name, "email": email}
    agent = s.current_agent
    # return ONLY the next agent (silent handover)
    return OnboardingAgent(room=agent.room, chat_ctx=s._chat_ctx)


@function_tool
async def handover_to_applications(
    context: RunContext[dict],
    name: str | None = None,
    email: str | None = None,
):
    from src.agents.job_application import (
        JobApplicationAgent,  # local import avoids cycles
    )

    s = context.session
    s.state = {
        **getattr(s, "state", {}),
        "name": name,
        "email": email,
    }
    agent = s.current_agent
    # Return ONLY the next agent (silent handover; no mid-chat line)
    return JobApplicationAgent(room=agent.room, chat_ctx=s._chat_ctx)


@function_tool
async def go_onboarding(context: RunContext[dict]):
    """Transfer the conversation to the OnboardingAgent."""
    from src.agents.onboarding import OnboardingAgent  # avoid circular import

    agent = context.session.current_agent
    return OnboardingAgent(room=agent.room, chat_ctx=context.session._chat_ctx)


@function_tool
async def go_applications(context: RunContext[dict]):
    """Transfer the conversation to the JobApplicationAgent."""
    from src.agents.job_application import (
        JobApplicationAgent,  # local import avoids cycles
    )

    agent = context.session.current_agent
    return JobApplicationAgent(room=agent.room, chat_ctx=context.session._chat_ctx)


@function_tool
async def go_assessment(self, context: RunContext[dict]):
    from src.agents.assessment import AssessmentAgent  # local import avoids cycles

    agent = context.session.current_agent
    return (AssessmentAgent(room=agent.room, chat_ctx=context.session._chat_ctx),)
