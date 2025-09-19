from livekit.agents import RunContext, function_tool

from src.config.loader import get_cfg


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
async def go_assessment(context: RunContext[dict]):
    from src.agents.assessment import AssessmentAgent  # local import avoids cycles

    agent = context.session.current_agent
    return (AssessmentAgent(room=agent.room, chat_ctx=context.session._chat_ctx),)


def get_handover_tools(agent: str) -> list:
    """
    Return a list of tools enabled for the tenant.

    agent_type:
        - "onboarding": OnboardingAgent, adds handover to Applications
        - "applications": ApplicationsAgent, adds handover to Onboarding
        - "router": Router, include all enabled tools
    """
    cfg = get_cfg()
    enabled = set(cfg["enabled_agents"])
    tools = []
    print(f"Enabled agents: {enabled}")

    if agent == "router":
        if "Applications" in enabled:
            tools.append(go_applications)
        if "Onboarding" in enabled:
            tools.append(go_onboarding)
        if "Assessments" in enabled:
            tools.append(go_assessment)
    else:
        # Agents: cross-handover + assessments
        if "Assessments" in enabled:
            tools.append(go_assessment)

        if agent == "onboarding" and "Applications" in enabled:
            tools.append(handover_to_applications)
        elif agent == "applications" and "Onboarding" in enabled:
            tools.append(handover_to_onboarding)

    return tools
