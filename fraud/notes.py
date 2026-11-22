"""Analyst case notes written by a local LLM, and a check that they only use facts they were given."""
from __future__ import annotations

import json
import re
import urllib.request

PROMPT = """You write a short note for a fraud analyst reviewing one flagged transaction.
Use ONLY the facts below. Do not add numbers, names or causes that are not listed.
Write 2 or 3 plain sentences (under 70 words). Say what looks unusual and end by saying it should be reviewed.

Facts:
{facts}
"""

NUM = re.compile(r"\d[\d,]*\.?\d*")


def facts_for(alert: dict) -> dict:
    return {
        "transaction": f"{alert['type']} of {alert['amount']:,.2f} from {alert['nameOrig']} to {alert['nameDest']} at step {alert['step']}",
        "risk_score": f"{alert['score']:.3f}",
        "reasons": [r["text"] for r in alert["reasons"]],
    }


def _numbers(text: str) -> set[float]:
    out = set()
    for m in NUM.findall(text):
        s = m.replace(",", "").rstrip(".")
        if s:
            try:
                out.add(round(float(s), 2))
            except ValueError:
                pass
    return out


def _idents(text: str) -> set[str]:
    return set(re.findall(r"\b[CM]\d{4,}\b", text))


def check_grounded(note: str, facts: dict, max_words: int = 90) -> list[str]:
    """Return a list of problems; empty means the note is grounded."""
    blob = json.dumps(facts)
    problems = []
    allowed = _numbers(blob)
    allowed |= {round(float(s), 0) for s in allowed}
    for n in _numbers(note):
        if n not in allowed and round(n, 0) not in allowed and round(n * 100, 0) not in allowed:
            problems.append(f"number {n:g} is not in the facts")
    for ident in _idents(note) - _idents(blob):
        problems.append(f"identifier {ident} is not in the facts")
    if len(note.split()) > max_words:
        problems.append("too long")
    if "review" not in note.lower():
        problems.append("does not recommend review")
    return problems


def ask_ollama(prompt: str, model: str, host: str = "http://localhost:11434", timeout: int = 300) -> str:
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 2048},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"].strip()


def generate(alerts: list[dict], model: str, host: str, ask=ask_ollama, retries: int = 1) -> dict:
    """One note per alert; a note that fails the grounding check is retried, then dropped."""
    out, stats = {}, {"requested": 0, "accepted": 0, "first_try": 0, "rejected": 0}
    for a in alerts:
        facts = facts_for(a)
        prompt = PROMPT.format(facts=json.dumps(facts, indent=2))
        stats["requested"] += 1
        for attempt in range(retries + 1):
            note = ask(prompt, model, host)
            problems = check_grounded(note, facts)
            if not problems:
                out[str(a["id"])] = {"note": note, "attempts": attempt + 1}
                stats["accepted"] += 1
                stats["first_try"] += int(attempt == 0)
                break
            prompt += f"\n\nYour previous note was rejected: {'; '.join(problems)}. Rewrite it using only the facts."
        else:
            stats["rejected"] += 1
    return {"model": model, "stats": stats, "notes": out}
