# src/utils/tts_sanitizer.py
import re

LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "eff", "g": "jee",
    "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "ell", "m": "em", "n": "en",
    "o": "oh", "p": "pee", "q": "queue", "r": "ar", "s": "ess", "t": "tee", "u": "you",
    "v": "vee", "w": "double you", "x": "ex", "y": "why", "z": "zed",
}

DIGIT_WORD = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    # Remove bold/inline-code markers so TTS doesn't read them oddly
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text

# Remove meta/instructional lines like “you can say it like …”, “for example …”
# We keep it conservative so normal content stays intact.
_META_LINE_RE = re.compile(r"(?im)^\s*(?:you can say it like|for example|e\.g\.).*$")

def _strip_meta_examples(text: str) -> str:
    return _META_LINE_RE.sub("", text)

def _digits_as_words(s: str) -> str:
    return " ".join(DIGIT_WORD.get(ch, ch) for ch in s)

def _speak_initials_if_any(token: str) -> str:
    """
    If token starts with 1–2 letters followed by lowercase letters, speak those as letter names:
    jmanish -> 'jay manish', mkumar -> 'em kay kumar'.
    """
    m = re.match(r"^([A-Za-z]{1,2})(?=[a-z])", token)
    if not m:
        return token
    head_letters = m.group(1)
    head_spoken = " ".join(LETTER_NAMES.get(c.lower(), c) for c in head_letters)
    tail = token[len(head_letters):]
    return f"{head_spoken} {tail}"

def _speak_local_part(local: str) -> str:
    # Split by separators and digit runs, then speak each piece
    pieces = re.split(r"([._+\-]|\d+)", local)
    out = []
    for p in pieces:
        if not p:
            continue
        if p in {".","_","-","+"}:
            out.append({".":"dot","_":"underscore","-":"dash","+":"plus"}[p])
        elif p.isdigit():
            out.append(_digits_as_words(p))
        elif p.isalpha():
            out.append(_speak_initials_if_any(p))
        else:
            out.append(p)
    spoken = " ".join(out).replace("  ", " ").strip()
    return spoken

def _pronounce_provider(label: str) -> str:
    # Force "Gmail" as "jee-mail"
    return "jee-mail" if label.lower() == "gmail" else label

def _email_to_spoken(email: str) -> str:
    local, domain = email.split("@", 1)
    local_spoken = _speak_local_part(local)
    domain_parts = domain.split(".")
    domain_spoken = " dot ".join(_pronounce_provider(p) for p in domain_parts)
    return f"{local_spoken} at {domain_spoken}"

# ─── Public ──────────────────────────────────────────────────────────────────────

def sanitize_for_tts(text: str) -> str:
    """
    1) Strip meta/example lines and markdown (keeps voice natural).
    2) Convert detected emails into a spoken form.
    3) Leave all other text untouched.
    """
    text = _strip_meta_examples(_strip_markdown(text))
    return EMAIL_RE.sub(lambda m: _email_to_spoken(m.group(0)), text)
