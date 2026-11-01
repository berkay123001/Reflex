"""CDP Browser observation and execution module."""

from .cdp import Browser, NavigationTimeout, StalePage, TargetCoveredError, clean_surrogates, fingerprint
from .isolated import _restore_harness_daemon_name, _set_harness_daemon_name, is_port_open, managed_isolated_browser

__all__ = [
    "Browser",
    "NavigationTimeout",
    "StalePage",
    "TargetCoveredError",
    "_restore_harness_daemon_name",
    "_set_harness_daemon_name",
    "clean_surrogates",
    "fingerprint",
    "is_port_open",
    "managed_isolated_browser",
]
