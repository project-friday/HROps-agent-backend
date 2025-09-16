from livekit.plugins import deepgram

EMAIL_KEYTERMS = [
    # Providers (+ spaced variants)
    "gmail","g mail","yahoo","y mail","outlook","out look",

    # Your / client domains (add more as needed)
    "renan.one","renan dot one","renan","hsbc.com","hsbc dot com","hsbc",  # keep if you expect it

    # Symbol words people say while dictating emails
    "at the rate","at","dot","underscore","under score","hyphen","dash",

    # Number words (local parts like `singh four`)
    "zero","oh","o","one","two","three","four","for","five","six","seven","eight","nine",

    # Common TLDs / ccTLDs (users often say these)
    "com","one","org","net","cloud","co","in","co.in",

    # Full phrases (stabilize)
    "gmail.com","g mail dot com",
    "renan.one","renan dot one",
    "yahoo.com","y mail dot com","yahoo.co.in",
    "outlook.com","out look dot com",
    "hotmail.com","hot mail dot com",
    "icloud.com","i cloud dot com",
]

def make_deepgram_stt(language: str = "en-US", endpointing_ms: int = 200):
    """
    language: use "en-IN" for Indian English or "en-US" for US speakers
    endpointing_ms: ~250–350 is a good starting band to avoid mid-email cutoffs
    """
    return deepgram.STT(
        model="nova-3",
        language=language,
        keyterms=EMAIL_KEYTERMS,
        endpointing_ms=endpointing_ms,
        filler_words=True,
        smart_format=True,
    )
