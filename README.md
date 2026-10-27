# Reflex — A Fast, Resilient Hybrid Browser Agent

Reflex separates high-level reasoning and failure recovery (**System 2 Supervisor**: `qwen/qwen3.8-omni-flash`, Meta `Muse Spark`, etc.) from low-level, sub-200ms structured DOM execution (**System 1 Reflex**: `Turnike-Web 4B` / `~typesafe/jev-latest`).

## Key Capabilities
- **Dual-Brain Execution Loop:** System 1 executes rapid structured DOM actions (`CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL`, `WAIT`) while a calibrated multi-metric gate monitors operation confidence, target confidence, top-1/top-2 ambiguity margin, and `BLOCKED` states.
- **3-Layer Action Verification:** Verifies technical CDP execution, element-level state diffs (`value`, `checked`, `selected`, `expanded`, options), and semantic loop / `A -> B -> A -> B` oscillation patterns.
- **Interactive Co-Pilot & `ASK_USER`:** Pauses cleanly when required form data is missing or choices are ambiguous, asking the human user in chat and resuming from the exact browser state.
- **Irreversible Action Safety Guard (`CONFIRM_USER`) & Untrusted DOM Isolation:** Intercepts high-consequence actions (payments, deletions) for human approval and isolates webpage text inside `<untrusted_page_data>` boundaries.
