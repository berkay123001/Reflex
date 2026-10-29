# Reflex — A Fast, Resilient Hybrid Browser Agent

Reflex separates high-level reasoning and failure recovery (**System 2 Supervisor**: `qwen/qwen3.8-omni-flash`, Meta `Muse Spark`, etc.) from low-level, sub-200ms structured DOM execution (**System 1 Reflex**: `Turnike-Web 4B` / `~typesafe/jev-latest`).

## Key Capabilities
- **Dual-Brain Execution Loop:** System 1 executes rapid structured DOM actions (`CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL`, `WAIT`) while a calibrated multi-metric gate monitors operation confidence, target confidence, top-1/top-2 ambiguity margin, and `BLOCKED` states.
- **3-Layer Action Verification:** Verifies technical CDP execution, element-level state diffs (`value`, `checked`, `selected`, `expanded`, options), and semantic loop / `A -> B -> A -> B` oscillation patterns.
- **Interactive Co-Pilot & `ASK_USER`:** Pauses cleanly when required form data is missing or choices are ambiguous, asking the human user in chat and resuming from the exact browser state.
- **Irreversible Action Safety Guard (`CONFIRM_USER`) & Untrusted DOM Isolation:** Intercepts high-consequence actions (payments, deletions) for human approval and isolates webpage text inside `<untrusted_page_data>` boundaries.

## Run Modes

Studio, the MCP server and `scripts/eval.py` default to **`s2_only`**: the Muse supervisor decides every step. On 19 real-site tasks run twice (2026-10-29) it completed 36/38 with either `muse-spark-1.3` or `muse-spark-1.3-contributor`; the strict hybrid (System 1 only for confident clicks/typing) completed 33/38 and the original hybrid 14/19. The hybrid modes stay selectable: the `MOD` switch in Studio, `mode` in `reflex_start`, `--mode` in `scripts/eval.py`.

## Design Influences

The task ledger (planning + memory) adapts ideas from these works; no third-party code is copied:
- **Plan-and-Act** (Erdogan et al., 2025), *Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks*, [arXiv:2503.09572](https://arxiv.org/abs/2503.09572): a separate planner whose plan is revised as execution progresses. Reflex's System 2 writes the plan once at the start and updates it at milestones.
- **StructuredAgent** (Lobo et al., 2026), *StructuredAgent: Planning with AND/OR Trees for Long-Horizon Web Tasks*, [arXiv:2603.05294](https://arxiv.org/abs/2603.05294): explicit, structured plan state instead of free-form notes. Reflex keeps a flat subgoal list with completion conditions and source-attributed findings.
- **BrowseMaster** (Pang et al., 2025), *Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair*, [arXiv:2508.09129](https://arxiv.org/abs/2508.09129): keeping the reasoning model's context focused while execution is delegated, and preserving where each fact came from.
- **Magentic-One** (Microsoft, 2024, MIT License): task ledger, progress ledger and stuck counter.

## MCP Server (Claude Code / Antigravity)

Install with the `mcp` extra and register the stdio server in Claude Code:

```bash
pip install -e ".[mcp]"
claude mcp add reflex -- /MUTLAK/YOL/.venv/bin/reflex-mcp
```

Note: Each `reflex-mcp` process supports a single active browser session at a time. Always call `reflex_stop` before starting a new session with `reflex_start`.
