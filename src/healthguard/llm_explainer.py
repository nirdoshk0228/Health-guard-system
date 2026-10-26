"""LLM explanation layer: turns a model score + SHAP drivers into an analyst-readable summary.

Backends (picked automatically, override with HEALTHGUARD_LLM=anthropic|openai|ollama|template):
  * anthropic : needs ANTHROPIC_API_KEY  (model via HEALTHGUARD_ANTHROPIC_MODEL, default claude-haiku-5-5)
  * openai    : needs OPENAI_API_KEY     (model via HEALTHGUARD_OPENAI_MODEL,    default gpt-4o-mini)
  * ollama    : local model at OLLAMA_HOST (default http://localhost:11434), model via HEALTHGUARD_OLLAMA_MODEL
  * template  : deterministic, offline narrative built from the same facts (always available)

The LLM only ever sees the structured facts below - score, tier, top SHAP drivers and peer
comparisons - never raw patient identifiers, and it is told not to add facts of its own.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are a healthcare payment-integrity analyst assistant.
You receive the output of machine-learning risk models (a probability, a risk tier, and the top
SHAP drivers with their values and peer benchmarks). Write a short, factual summary for a human
reviewer.
Rules:
- Use ONLY the facts provided. Do not invent numbers, diagnoses or history.
- A model score is a reason to review, never proof of fraud or of a clinical outcome. Do not accuse.
- Mention the 2-4 most important drivers in plain English, each with its value and how it compares
  to peers when a benchmark is given.
- End with one concrete, proportionate next step for the reviewer.
- Format: a one-line headline, then 3-5 bullet points, then 'Next step:' line. Under 170 words."""


@dataclass
class RiskCase:
    entity_type: str          # "provider" | "claim" | "patient" | "readmission"
    entity_id: str
    score: float              # model probability 0-1
    tier: str
    drivers: list[dict]       # from explain.top_factors(), optionally with "peer" benchmark
    extra: dict = field(default_factory=dict)

    def facts(self) -> dict:
        return {
            "entity_type": self.entity_type, "entity_id": self.entity_id,
            "model_score": round(self.score, 3), "risk_tier": self.tier,
            "top_drivers": [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "feature"}
                for d in self.drivers
            ],
            **self.extra,
        }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.2f}"


def template_summary(case: RiskCase) -> str:
    """Offline fallback - same facts, fixed wording."""
    what = {"provider": "Provider", "claim": "Claim", "patient": "Patient",
            "readmission": "Encounter"}.get(case.entity_type, case.entity_type.title())
    outcome = "30-day readmission risk" if case.entity_type == "readmission" else "fraud risk"
    lines = [f"**{what} {case.entity_id}: {case.tier} {outcome} (model score {case.score:.0%})**", ""]
    for d in [d for d in case.drivers if d["shap"] > 0][:4] or case.drivers[:3]:
        peer = f" vs peer median {_fmt(d['peer'])}" if d.get("peer") is not None else ""
        lines.append(f"- {d['label'].capitalize()} = {_fmt(d['value'])}{peer} - {d['direction']} the risk score.")
    offsets = [d for d in case.drivers if d["shap"] < 0][:1]
    for d in offsets:
        lines.append(f"- Offsetting factor: {d['label']} = {_fmt(d['value'])}.")
    if case.entity_type == "readmission":
        nxt = ("schedule a follow-up call within 7 days of discharge and reconcile medications."
               if case.tier == "High" else "standard discharge planning; no extra intervention indicated by the model.")
    else:
        nxt = {"Critical": "open a full desk audit and request medical records for the highest-value claims.",
               "High": "pull a sample of recent claims for documentation review.",
               "Medium": "add to the watch-list and re-score next cycle."}.get(case.tier, "no action; continue routine monitoring.")
    lines += ["", f"Next step: {nxt}", "", "_Generated offline from model outputs (template mode)._"]
    return "\n".join(lines)


def _anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("HEALTHGUARD_ANTHROPIC_MODEL", "claude-haiku-5-5"),
        max_tokens=400, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai(prompt: str) -> str:
    from openai import OpenAI

    r = OpenAI().chat.completions.create(
        model=os.getenv("HEALTHGUARD_OPENAI_MODEL", "gpt-4o-mini"), max_tokens=400, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def _ollama(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    body = json.dumps({"model": os.getenv("HEALTHGUARD_OLLAMA_MODEL", "llama3.1"), "stream": False,
                       "system": SYSTEM_PROMPT, "prompt": prompt, "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["response"]


BACKENDS = {"anthropic": _anthropic, "openai": _openai, "ollama": _ollama}


def active_backend() -> str:
    forced = os.getenv("HEALTHGUARD_LLM")
    if forced:
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OLLAMA_HOST"):
        return "ollama"
    return "template"


def explain_case(case: RiskCase) -> tuple[str, str]:
    """Returns (summary_markdown, backend_used). Falls back to the template on any LLM error."""
    backend = active_backend()
    if backend in BACKENDS:
        prompt = "Model output to summarise:\n" + json.dumps(case.facts(), indent=2, default=str)
        try:
            return BACKENDS[backend](prompt).strip(), backend
        except Exception as exc:  # network, auth, missing package...
            return template_summary(case) + f"\n\n_(LLM backend '{backend}' failed: {exc.__class__.__name__}; showing template.)_", "template"
    return template_summary(case), "template"
