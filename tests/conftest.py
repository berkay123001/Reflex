"""Test-wide settings: the final-answer check would call Baseten for real, so it is off unless a test enables it."""

import os

os.environ["REFLEX_ANSWER_CHECK_MODEL"] = "off"
