# Example using OpenAI
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def translate_to_english(text: str) -> str:
    """
    Translate given text into English.
    Replace with your preferred translation service (e.g. OpenAI, DeepL, Google).
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # or another model
        messages=[
            {"role": "system", "content": "You are a translation assistant."},
            {"role": "user", "content": f"Translate this into English: {text}"},
        ],
    )

    return response.choices[0].message.content.strip()
