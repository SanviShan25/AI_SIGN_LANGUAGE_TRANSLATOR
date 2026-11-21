"""Gloss sequence -> natural English sentence.

Sign languages have their own grammar, so the raw word stream ("ME GO STORE
TOMORROW") needs rewriting ("I will go to the store tomorrow.").

Provider is chosen with environment variables (no extra pip packages needed):

    SIGN_LLM_PROVIDER = anthropic | openai | groq | ollama | none   (default: auto)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
    SIGN_LLM_MODEL    = optional model override

``auto`` uses the first provider whose key is set, otherwise a rule-based fallback,
so the project always runs offline too.

    python -m llm_layer.grammar_correction ME GO STORE TOMORROW
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from functools import lru_cache

SYSTEM_PROMPT = (
    "You convert American Sign Language gloss sequences into one short, natural English "
    "sentence. Keep the meaning, add only the small words English needs (articles, "
    "auxiliaries, tense). Do not invent new content. Reply with the sentence only."
)

PROVIDERS = {
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "claude-haiku-5-5"},
    "openai": {"key": "OPENAI_API_KEY", "model": "gpt-4o-mini",
               "url": "https://api.openai.com/v1/chat/completions"},
    "groq": {"key": "GROQ_API_KEY", "model": "llama-3.1-8b-instant",
             "url": "https://api.groq.com/openai/v1/chat/completions"},
    "ollama": {"key": None, "model": "llama3.1",
               "url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")},
}


def _post(url: str, headers: dict, body: dict, timeout: float = 15) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _pick_provider() -> str:
    p = os.getenv("SIGN_LLM_PROVIDER", "auto").lower()
    if p != "auto":
        return p
    for name, spec in PROVIDERS.items():
        if spec["key"] and os.getenv(spec["key"]):
            return name
    return "none"


def _call_llm(provider: str, gloss: str) -> str:
    spec = PROVIDERS[provider]
    model = os.getenv("SIGN_LLM_MODEL", spec["model"])
    user = f"Gloss: {gloss}"
    if provider == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": os.environ[spec["key"]], "anthropic-version": "2023-06-01"},
                     {"model": model, "max_tokens": 100, "system": SYSTEM_PROMPT,
                      "messages": [{"role": "user", "content": user}]})
        return data["content"][0]["text"].strip()
    headers = {"Authorization": f"Bearer {os.environ[spec['key']]}"} if spec["key"] else {}
    data = _post(spec["url"], headers, {
        "model": model, "temperature": 0.2, "max_tokens": 100,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}]})
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- fallback
_PRONOUNS = {"me": "I", "i": "I", "you": "you", "he": "he", "she": "she", "we": "we", "they": "they"}
_BE = {"I": "am", "you": "are", "we": "are", "they": "are", "he": "is", "she": "is"}
_ADJ = {"happy", "sad", "hungry", "tired", "fine", "good", "bad", "sick", "ready", "busy", "sorry"}
_TIME_FUTURE = {"tomorrow", "later", "will", "future"}
_TIME_PAST = {"yesterday", "before", "finish", "past"}


def rule_based(words: list[str]) -> str:
    """Tiny offline fallback - good enough for demos, not real grammar."""
    w = [x.lower().replace("_", " ") for x in words if x]
    if not w:
        return ""
    out = [_PRONOUNS.get(x, x) for x in w]
    if len(out) >= 2 and out[0] in _BE and out[1] in _ADJ:
        out.insert(1, _BE[out[0]])
    elif len(out) >= 2 and out[0] in _BE and set(w) & _TIME_FUTURE:
        out.insert(1, "will")
    if w[-1] in {"what", "where", "who", "why", "how", "when", "which"}:
        out = [out[-1]] + out[:-1]
        end = "?"
    else:
        end = "?" if w[0] in {"what", "where", "who", "why", "how", "when"} else "."
    s = " ".join(out)
    return s[0].upper() + s[1:] + end


@lru_cache(maxsize=512)
def _correct_cached(gloss: str, provider: str) -> tuple[str, str]:
    if provider != "none":
        try:
            return _call_llm(provider, gloss), provider
        except Exception as e:  # network down, bad key, ... -> never crash the app
            print(f"[llm] {provider} failed ({e}); using rule-based fallback", file=sys.stderr)
    return rule_based(gloss.split()), "rule-based"


def gloss_to_sentence(words: list[str]) -> tuple[str, str]:
    """Returns (sentence, backend_used)."""
    words = [w for w in words if w]
    if not words:
        return "", "none"
    return _correct_cached(" ".join(w.upper() for w in words), _pick_provider())


if __name__ == "__main__":
    args = sys.argv[1:] or ["ME", "HAPPY", "SEE", "YOU"]
    sentence, backend = gloss_to_sentence(args)
    print(f"{' '.join(args)}  ->  {sentence}   [{backend}]")
