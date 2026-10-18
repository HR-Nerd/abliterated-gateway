"""Mini Policy Gateway — allow/refuse/rewrite/redact/escalate.

Mirrors abliteration.ai's Policy Gateway but runs locally, free, no external services.
A policy = a yes/no question + sensitivity + mode (monitor|block) + scope.
"""
from __future__ import annotations
import json, re, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional


@dataclass
class Policy:
    id: str
    question: str                      # yes/no question evaluated against content
    mode: str = "monitor"              # "monitor" | "block"
    sensitivity: float = 0.5            # 0..1, lower = flag more
    applies_to: str = "all"            # "all" | "selected" | "explicit"
    scope: str = "both"                # "request" | "response" | "both"
    enabled: bool = True


@dataclass
class Decision:
    policy_id: str
    matched: bool
    outcome: str                       # allow|refuse|rewrite|redact|escalate
    reason: str
    latency_ms: float


# --- built-in judges (no LLM needed; swap for an LLM-as-judge if you want) ---
def judge_keyword(question: str, content: str) -> float:
    """Tiny rule-based judge: returns match score 0..1."""
    q = question.lower()
    c = content.lower()
    triggers = {
        "harass": (["harass", "bully", "dox", "stalk", "harassment"], ["harass", "harm", "dox", "twitter"]),
        "phi": (["patient", "chart", "diagnosis", "ssn", "dob", "phi"], ["patient", "chart", "differential", "ssn"]),
        "cyber": (["exploit", "cve", "payload", "recon", "backdoor", "offensive", "security", "red team"], ["exploit", "cve", "payload", "recon", "backdoor", "poc"]),
        "pii": (["ssn", "credit card", "password", "api key", "pii"], ["ssn", "credit card", "password", "api key"]),
        "tone": (["abusive", "insult", "idiot", "stupid"], ["idiot", "stupid", "worst", "idiots"]),
    }
    score = 0.0
    for topic, (qwords, cwords) in triggers.items():
        if any(w in q for w in qwords) or topic in q:
            if any(w in c for w in cwords):
                score = max(score, 0.9)
    if score == 0.0 and any(w in c for w in q.split() if len(w) > 4):
        score = 0.3
    return score


class PolicyGateway:
    def __init__(self, audit_path: str = "audit.jsonl"):
        self.policies: List[Policy] = []
        self.audit_path = audit_path
        self.judge: Callable[[str, str], float] = judge_keyword

    def add(self, p: Policy):
        self.policies.append(p)

    def evaluate(self, content: str, phase: str = "request",
                 policy_ids: Optional[List[str]] = None) -> List[Decision]:
        out = []
        for p in self.policies:
            if not p.enabled:
                continue
            if p.scope != "both" and p.scope != phase:
                continue
            if p.applies_to == "explicit" and (not policy_ids or p.id not in policy_ids):
                continue
            t0 = time.time()
            score = self.judge(p.question, content)
            matched = score >= (1.0 - p.sensitivity)  # lower sensitivity -> lower bar
            if not matched:
                outcome = "allow"
            elif p.mode == "monitor":
                outcome = "allow"   # recorded but not blocked
            else:
                # block mode: decide sub-outcome by question intent
                q = p.question.lower()
                if "redact" in q or "phi" in q or "pii" in q:
                    outcome = "redact"
                elif "rewrite" in q or "tone" in q:
                    outcome = "rewrite"
                elif "escalate" in q:
                    outcome = "escalate"
                else:
                    outcome = "refuse"
            d = Decision(p.id, matched, outcome, p.question, (time.time()-t0)*1000)
            out.append(d)
            self._audit(phase, content, d)
        return out

    def apply(self, content: str, decisions: List[Decision]) -> str:
        """Apply redact/rewrite transforms to content based on decisions."""
        result = content
        for d in decisions:
            if d.outcome == "redact":
                # simple PII-ish redaction
                result = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", result)
                result = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL REDACTED]", result)
            elif d.outcome == "rewrite":
                result = result.replace("idiot", "[removed]").replace("stupid", "[removed]")
            elif d.outcome == "refuse":
                return "I can't help with that request under the active policy."
        return result

    def _audit(self, phase, content, d: Decision):
        rec = {"ts": time.time(), "phase": phase, "content_sha": hash(content) & 0xffff,
               "decision": asdict(d)}
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass


# --- demo policies (mirrors abliteration.ai's Policy Gateway examples) ---
def default_gateway() -> PolicyGateway:
    gw = PolicyGateway()
    gw.add(Policy("cybersec-research", "Is this an authorized offensive-security request?",
                  mode="monitor", sensitivity=0.6, scope="both"))
    gw.add(Policy("patient-records", "Does this contain PHI that must be redacted?",
                  mode="block", sensitivity=0.7, scope="both"))
    gw.add(Policy("trust-and-safety", "Is this harassment or harm to others?",
                  mode="block", sensitivity=0.8, scope="both"))
    gw.add(Policy("support-tone", "Is the tone abusive?",
                  mode="block", sensitivity=0.5, scope="request"))
    return gw
