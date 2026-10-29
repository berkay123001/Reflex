"""CDP Browser observation and execution module."""

from .cdp import Browser, StalePage, TargetCoveredError, fingerprint
from .isolated import _restore_harness_daemon_name, _set_harness_daemon_name, is_port_open, managed_isolated_browser

__all__ = [
    "Browser",
    "StalePage",
    "TargetCoveredError",
    "_restore_harness_daemon_name",
    "_set_harness_daemon_name",
    "fingerprint",
    "is_port_open",
    "managed_isolated_browser",
]
