import html
import logging
import re
from fastapi import HTTPException
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from openai.types import ModerationCreateResponse
# Expanded, Docker-aware threat patterns
MALICIOUS_PATTERNS = [
    # Dangerous Python / OS execution
    r"(?i)\b(eval|exec|__import__|os\.system|subprocess|popen|pexpect|pickle\.loads)\b",

    # Shell metacharacters commonly used for injection ( | && ` $() <> {})
    r"[&|`$<>]",
    r"\$\((.*?)\)",            # $(cmd)
    r"`[^`]+`",                # `cmd`

    # Command injection one-liners
    r"(?i)\b(bash\s+-c|sh\s+-c|python\s+-c|perl\s+-e|php\s+-r|ruby\s+-e|node\s+-e)\b",

    # Reverse-shell patterns
    r"(?i)(/dev/tcp/|nc\s|ncat\s|netcat\s|bash\s+-i|mkfifo|/bin/sh\s+-i)",

    # Docker-specific escapes & mounts
    r"(?i)(/var/run/docker\.sock|--privileged|--pid=host|--network=host)",
    r"(?i)\b(docker\s+(run|exec|cp|kill|rm)|docker-compose|kubectl\s+(exec|run|apply))\b",

    # Path traversal / sensitive locations
    r"(\.\./|\.\.\\|/etc/passwd|/etc/shadow|/proc/\d+/mem|/sys/)",
    r"(?i)(C:\\Windows\\System32|/root/|/home/.+/.ssh/)",

    # Remote code download-and-execute
    r"(?i)(curl|wget).*(http|https|ftp)://.*\|\s*(sh|bash)",

    # XSS injections
    r"(?i)(<script\b|javascript:|on\w+\s*=)",

    # Suspicious long base64 blobs (obfuscation)
    r"[A-Za-z0-9+/]{120,}={0,2}",
]

COMPILED_PATTERNS = [re.compile(p) for p in MALICIOUS_PATTERNS]


def sanitize_passage(user_input: str, max_len: int = 5000) -> str:
    logging.info(f'scanning for malicious content: {user_input}')
    if not isinstance(user_input, str):
        raise HTTPException(400, "Invalid type")

    user_input = user_input.strip()
    if not user_input:
        raise HTTPException(400, "Empty passage")

    if len(user_input) > max_len:
        raise HTTPException(413, "Payload too large")

    # Reject non-UTF input
    try:
        user_input.encode("utf-8")
    except Exception:
        raise HTTPException(400, "Invalid encoding")

    # Neutralize HTML
    user_input = html.escape(user_input)

    # Remove invisible obfuscation characters (zero-width)
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", user_input)

    # Pattern scans
    for rx in COMPILED_PATTERNS:
        if rx.search(clean):
            raise HTTPException(400, "Malicious content detected by {}".format(rx.pattern))

    do_moderation(user_input)  # try with openai moderation

    return clean


def do_moderation(user_input: str):
    logging.info(f'moderation scanning for malicious content: {user_input}')
    client = OpenAI()

    try:
        resp: ModerationCreateResponse = client.moderations.create(
            model="omni-moderation-latest",
            input=user_input
        )
        logging.info(f'moderation result: {resp}')
        if any(result.flagged for result in resp.results):
            raise HTTPException(status_code=403, detail="Unsupported content detected")

    except RateLimitError:
        # Fail closed (block the request)
        raise HTTPException(status_code=429, detail="Moderation rate limit hit")

    except (APIError, APIConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Moderation service unavailable: {e}"
        )

