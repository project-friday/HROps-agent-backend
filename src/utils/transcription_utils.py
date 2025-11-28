import regex as re
from livekit.agents import llm

# Allowed language scripts
ALLOWED_REGEX = re.compile(r"[A-Za-z0-9\p{Arabic}\p{P}\p{S}\s]")


def is_wrong_script(text: str) -> bool:
    # True if ANY character is NOT allowed
    for ch in text:
        if not ALLOWED_REGEX.match(ch):
            return True
    return False


async def repair_text(agent, wrong_text: str, target_lang: str) -> str:
    """
    Uses the agent's internal LLM and ChatContext to correctly convert
    wrong-script transcription into proper English or Arabic text.
    """

    chat_ctx = llm.ChatContext.empty()

    # System message
    chat_ctx.add_message(
        role="system",
        content=[
            (
                "You are a transcription repair model. The input text contains incorrect "
                "characters from the wrong language script. Convert the text into meaningful "
                f"{'Arabic' if target_lang.startswith('ar') else 'English'}."
                "only give the corrected text as output, no extra content or characters."
            )
        ],
    )

    # User message containing the corrupted STT text
    chat_ctx.add_message(
        role="user",
        content=[f"Fix this text: {wrong_text}"],
    )

    # Call the LLM
    activity = agent._get_activity_or_raise()
    llm_client = activity.llm

    async with llm_client.chat(chat_ctx=chat_ctx) as stream:
        chunks = []
        async for chunk in stream:
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)

    # Cleanup: remove messages from memory
    chat_ctx.items.clear()

    return "".join(chunks).strip()
