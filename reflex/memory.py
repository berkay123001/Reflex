"""Optional Harness²-inspired Domain-Specific Self-Healing Memory (`site_memory.py`).

Stores structural recovery rules per domain: rules the supervisor learns while `enable_site_memory=True`,
and rules the user writes for a website in Reflex Studio. Strictly sanitizes rules to prevent prompt
injection from untrusted page content.
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


MAX_LEARNED_RULES_PER_DOMAIN = 5
MAX_USER_RULES_PER_DOMAIN = 10
MAX_USER_RULE_CHARS = 300


class DomainRule(BaseModel):
    """A sanitized structural navigation rule for a specific domain, learned or written by the user."""

    domain: str
    rule: str
    trigger_reason: str
    source: str = "learned"  # "learned" | "user"
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
    def normalize_domain(target: str) -> str:
        """Domain for a URL or a bare host such as `compass.com`."""
        text = (target or "").strip()
        if not text:
            return ""
        if "://" not in text:
            text = "https://" + text
        return SiteMemoryStore.extract_domain(text)

    @staticmethod
    def is_safe_rule(rule_text: str, max_chars: int = 220) -> bool:
        text = (rule_text or "").strip()
        if not (5 <= len(text) <= max_chars):
            return False
        return not any(p.search(text) for p in _FORBIDDEN_INJECTION_PATTERNS)

    def _matching_domains(self, url: str) -> list[str]:
        host = self.extract_domain(url)
        if not host:
            return []
        return [dom for dom in self._rules if host == dom or host.endswith("." + dom)]

    def get_domain_hint(self, url: str) -> str:
        domain = self.extract_domain(url)
        if not domain or domain not in self._rules:
            return ""
        rules = self._rules[domain][-2:]
        return " | ".join(r.rule for r in rules)

    def user_rules_for(self, url: str) -> list[str]:
        """Rules the user wrote for this URL's website or a parent domain, oldest first."""
        return [r.rule for dom in self._matching_domains(url) for r in self._rules[dom] if r.source == "user"]

    def add_user_rule(self, target: str, rule_text: str) -> DomainRule:
        """Store a rule the user wrote for a website. Raises ValueError for a bad domain or an unsafe rule."""
        domain = self.normalize_domain(target)
        if not domain or "." not in domain:
            raise ValueError("Enter a website such as compass.com or https://www.compass.com/")
        cleaned = " ".join((rule_text or "").split())
        if not self.is_safe_rule(cleaned, max_chars=MAX_USER_RULE_CHARS):
            raise ValueError(
                f"A rule needs 5-{MAX_USER_RULE_CHARS} characters and cannot contain links or instructions about prompts."
            )
        bucket = self._rules.setdefault(domain, [])
        existing = next((r for r in bucket if r.source == "user" and r.rule.lower() == cleaned.lower()), None)
        if existing:
            return existing
        if sum(r.source == "user" for r in bucket) >= MAX_USER_RULES_PER_DOMAIN:
            raise ValueError(f"{domain} already has {MAX_USER_RULES_PER_DOMAIN} rules; delete one first.")
        rule = DomainRule(domain=domain, rule=cleaned, trigger_reason="added by the user", source="user")
        bucket.append(rule)
        self._save()
        return rule

    def delete_rule(self, domain: str, created_at: float) -> bool:
        bucket = self._rules.get(domain, [])
        kept = [r for r in bucket if abs(r.created_at - float(created_at)) > 1e-6]
        if len(kept) == len(bucket):
            return False
        if kept:
            self._rules[domain] = kept
        else:
            self._rules.pop(domain, None)
        self._save()
        return True

    def record_rule(self, url: str, rule_text: str, trigger_reason: str) -> bool:
        domain = self.extract_domain(url)
        if not domain or not self.is_safe_rule(rule_text):
            return False
        bucket = self._rules.setdefault(domain, [])
        cleaned = " ".join(rule_text.strip().split())
        if any(r.rule.lower() == cleaned.lower() for r in bucket):
            return False
        bucket.append(DomainRule(domain=domain, rule=cleaned, trigger_reason=trigger_reason[:120]))
        learned = [r for r in bucket if r.source != "user"][-MAX_LEARNED_RULES_PER_DOMAIN:]
        self._rules[domain] = [r for r in bucket if r.source == "user" or r in learned]
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
                self._rules[dom] = [
                    DomainRule(**item)
                    for item in items
                    if self.is_safe_rule(
                        item.get("rule", ""),
                        max_chars=MAX_USER_RULE_CHARS if item.get("source") == "user" else 220,
                    )
                ]
        except Exception:
            self._rules = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {dom: [r.model_dump() for r in rules] for dom, rules in self._rules.items()}
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
