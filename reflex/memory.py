"""Optional Harness²-inspired Domain-Specific Self-Healing Memory (`site_memory.py`).

Stores structural recovery rules per domain when enabled (`enable_site_memory=True`).
Strictly sanitizes rules to prevent prompt injection from untrusted page content.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .config import PROJECT_ROOT

DEFAULT_MEMORY_PATH = PROJECT_ROOT / "artifacts" / "site_memory.json"

_FORBIDDEN_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"<untrusted_page_data>", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
]


class DomainRule(BaseModel):
    """A sanitized structural navigation rule learned for a specific domain."""

    domain: str
    rule: str
    trigger_reason: str
    created_at: float = Field(default_factory=time.time)


class SiteMemoryStore:
    """Persistent, sanitized domain hint bank."""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH):
        self.path = path
        self._rules: dict[str, list[DomainRule]] = {}
        self._load()

    @staticmethod
    def extract_domain(url: str) -> str:
        try:
            host = urlparse(url).hostname or ""
            return host.lower().removeprefix("www.")
        except Exception:
            return ""

    @staticmethod
    def is_safe_rule(rule_text: str) -> bool:
        text = (rule_text or "").strip()
        if not (5 <= len(text) <= 220):
            return False
        return not any(p.search(text) for p in _FORBIDDEN_INJECTION_PATTERNS)

    def get_domain_hint(self, url: str) -> str:
        domain = self.extract_domain(url)
        if not domain or domain not in self._rules:
            return ""
        rules = self._rules[domain][-2:]
        return " | ".join(r.rule for r in rules)

    def record_rule(self, url: str, rule_text: str, trigger_reason: str) -> bool:
        domain = self.extract_domain(url)
        if not domain or not self.is_safe_rule(rule_text):
            return False
        bucket = self._rules.setdefault(domain, [])
        cleaned = " ".join(rule_text.strip().split())
        if any(r.rule.lower() == cleaned.lower() for r in bucket):
            return False
        bucket.append(DomainRule(domain=domain, rule=cleaned, trigger_reason=trigger_reason[:120]))
        self._rules[domain] = bucket[-5:]
        self._save()
        return True

    def all_rules(self) -> dict[str, list[dict]]:
        return {dom: [r.model_dump() for r in rules] for dom, rules in self._rules.items()}

    def all_domains(self) -> dict[str, list[dict]]:
        return self.all_rules()

    def clear(self) -> None:
        self._rules = {}
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for dom, items in raw.items():
                self._rules[dom] = [DomainRule(**item) for item in items if self.is_safe_rule(item.get("rule", ""))]
        except Exception:
            self._rules = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {dom: [r.model_dump() for r in rules] for dom, rules in self._rules.items()}
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
