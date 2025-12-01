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
    Repairs wrong-script text if it phonetically resembles English or Arabic.
    Otherwise returns the original text.
    """

    chat_ctx = llm.ChatContext.empty()

    chat_ctx.add_message(
        role="system",
        content=[
            (
                "You are a transcription repair model. Your task is to detect whether the input text "
                "is a FAILED transcription of English or Arabic that has been incorrectly rendered in "
                "another non-Latin or non-Arabic script (e.g., Telugu, Hindi, Malayalam, Thai, etc.). "
                "If the text phonetically resembles English or Arabic words but is written in another "
                "script, convert it to proper English or Arabic based on the intended language. "
                "If the input text is a legitimate word or sentence in ANY other natural language "
                "(Chinese, Japanese, Korean, Hindi, Malayalam, etc.), or if the text does NOT correspond "
                "to English/Arabic speech, DO NOT modify it. "
                "In that case, output the exact string: __NO_REPAIR__. "
                "When repairing, output ONLY the corrected text with no explanations, extra characters, "
                "or additional sentences."
            )
        ],
    )

    chat_ctx.add_message(
        role="user",
        content=[f"Fix this text for {target_lang}: {wrong_text}"],
    )

    activity = agent._get_activity_or_raise()
    llm_client = activity.llm

    async with llm_client.chat(chat_ctx=chat_ctx) as stream:
        chunks = []
        async for chunk in stream:
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)

    chat_ctx.items.clear()

    result = "".join(chunks).strip()

    # If LLM says "no repair needed", return original
    if result == "__NO_REPAIR__":
        return wrong_text

    return result
