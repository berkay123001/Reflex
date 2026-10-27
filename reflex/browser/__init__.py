"""CDP Browser observation and execution module."""

from .cdp import Browser, StalePage, TargetCoveredError, fingerprint

__all__ = ["Browser", "StalePage", "TargetCoveredError", "fingerprint"]
