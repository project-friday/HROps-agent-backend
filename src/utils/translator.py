# Example using OpenAI
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def translate_to_english(text: str) -> str:
    """
    Translate given text into English.
    Replace with your preferred translation service (e.g. OpenAI, DeepL, Google).
    """

    response = await client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",  # or another model
        messages=[
            {
                "role": "system",
                "content": "You are a translation assistant. Only transate the text, just output the translation of text given to you",
            },
            {"role": "user", "content": f"Translate this into English: {text}"},
        ],
    )

    return response.choices[0].message.content.strip()
