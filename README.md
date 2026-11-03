# Reflex — a cheap, honest browser agent on Muse Spark

> **Meta Global AI Hackathon 2026 · Indie Developer track · Problem statement: TODO(Berkay)**
> - **Provenance:** TODO(Berkay): one line on what existed before Oct 26 and what was built during Oct 26 – Nov 4. The repository was created on 2026-10-27.
> - **Meta models at runtime:**
>   - `muse-spark-1.3-contributor` (Meta Model API) is the supervisor: it reads each page and decides every browser step.
>   - `meta-models/muse-spark-open-1026` (**Muse Spark Open**, open weights, served by Baseten) checks every final answer against the user's task before it is shown. If a requirement is unmet, the agent is sent back once.
> - **Bonus categories:** Best Build on Meta Open Weight Models · Best Agentic Use-Case
> - **Demo video:** TODO
> - **Results without any API key:** [40 live benchmark tasks, each with the agent's answer and final screenshot](reports/online_mind2web_pilot40/index.html)

You give Reflex a website and a goal in plain language, for example *"find the cheapest 2 bed and 3+ bath apartment for rent in New York"* on compass.com. Reflex drives a real Chromium over the DevTools protocol: it reads the page, picks one action, checks that the action changed something, and repeats until it can answer. It asks before irreversible actions (payments, submitting forms), never invents personal data, and says so plainly when a site blocks it instead of making up a result.

## Results

### Online-Mind2Web: 40 live tasks

[Online-Mind2Web](https://arxiv.org/abs/2504.01382) has 300 tasks on 136 live websites. We drew 40 of them with a fixed seed, stratified by difficulty (11 easy, 19 medium, 10 hard), after excluding the 8 tasks we had used to diagnose failures ([`scripts/benchmark/sample_online_mind2web.py`](scripts/benchmark/sample_online_mind2web.py)). The same 40 tasks were run three times while we fixed general bugs:

| Run (same 40 tasks, max 40 steps) | Success | Success on reachable sites | Cost per task |
|---|---|---|---|
| **Final code** (answer check with Muse Spark Open) | **20/40 (50%)** | **20/30 (67%)** | $0.008 |
| Answer check, first prompt | 21/40 (53%) | 21/30 (70%) | $0.008 |
| No answer check | 19/40 (48%) | 19/31 (61%) | $0.006 |

Final code by difficulty (reachable tasks in brackets): easy 8/11 (8/9), medium 9/19 (9/15), hard 3/10 (3/6).

The three runs are within noise of each other: 40 tasks give roughly ±15 points of uncertainty. On these 40 tasks the answer check did not measurably raise the success rate. In the final run it sent the agent back once (a false alarm on a correct Compass answer; the agent re-checked and finished correctly), and it missed a substituted place (AccuWeather, Cork) that it had caught in the previous run, where the agent then answered honestly that the place was not listed. It adds 1–4 s per finished task.

"Reachable" excludes sites that refused our network before the agent could do anything: geo-blocks (CVS, Healthgrades, Adopt-a-Pet), Cloudflare and bot checks (cars.com, StubHub, BBB, JCPenney) and dead hosts. In all of these the agent reported the block instead of inventing a result.

For context, the paper's human-judged success rates on all 300 tasks: OpenAI Operator 61.3%, Claude Computer Use 3.7 56.3%, SeeAct 30.7%, Browser Use 30.0%, Agent-E 28.0%. **This is not a like-for-like comparison:**
- Our runs were judged by Claude (Opus 5.5) from each final answer and, where needed, the last screenshot. They were not judged by humans or by WebJudge. We judged strictly: for example, "the first Best Paper of ICLR 2016" counts only if it matches the first paper listed on iclr.cc.
- Live sites change from run to run (BBB was reachable in one run and behind Cloudflare in the next).
- The tasks date from 2025, so some sites have changed since. Our runs came from Turkey, where several US sites block access.

Per-task answers, verdicts and final screenshots of the final run: [`reports/online_mind2web_pilot40/`](reports/online_mind2web_pilot40/).

Where it still fails: long multi-step forms and filter chains (a six-topic newsletter signup, setting a Trader Joe's home store) run out of the 40-step budget (4 of 40), it stopped to ask the user twice, and three answers gave something close to but not what was asked (a nearby place, national averages instead of a zip code, a data science course instead of computer science).

### Our own tasks

On 20 tasks recorded from real users (Turkish, mostly research, quizzes and shopping comparisons), Reflex finished all 20 with an answer in its default mode; these answers were not graded separately. In an earlier ablation on 19 of these tasks, each run twice, the median time per finished task was 67 s; the human demonstrations took a median of 82 s.

## How it works

```
 goal ──▶ Muse Spark 1.3 plans a short task ledger (subgoals, findings with source URLs)
            │
            ▼
 ┌─▶ observe page ── snapshot.js + accessibility tree + iframes + open shadow roots
 │          │        (numbered elements; page text wrapped as untrusted data)
 │          ▼
 │   Muse Spark 1.3 picks ONE action  (click / type / select / scroll / go to URL / finish)
 │          │
 │          ▼
 │   guards in code ── stay on the task website, loop and oscillation detection,
 │          │          blocked-domain detection, confirm irreversible actions
 │          ▼
 └── act + verify (did the page, URL or element state change?)
            │  finish
            ▼
   Muse Spark Open checks the answer against every requirement of the task
   ── pass: answer shown · fail: agent sent back once with the unmet requirements
```

Each guard came from a failure we observed and was measured before and after:
- **Stay on the task website.** Muse often jumped to a guessed URL on another site (StubHub → Eventbrite, AccuWeather → Windy). Following links shown on pages is still allowed. Runs that left the task site: 17 → 5 of 40, and the remaining 4 followed the site's own links.
- **No silent dead ends.** When an element is covered or off-screen, the supervisor is told why the action failed; before this fix it kept choosing the same element. Supervisor calls that never turn into a browser step are capped.
- **Answer check (Muse Spark Open).** Tested offline on 64 answers from our runs, labelled by us: it caught 3 of 4 answers that substituted a different place or general figures for the one asked for, with 1–2 false alarms among 60 good answers. The prompt was tuned on these same answers, so this is optimistic; on the live runs it caught one of two substitutes in one run and neither in the other (see above). It is bounded: one rejection per task, a 20 s timeout, and any error accepts the answer.

## Quickstart

```bash
git clone https://github.com/berkay123001/Reflex.git && cd Reflex
uv sync
cp .env.example .env      # META_API_KEY, BASETEN_API_KEY (Muse Spark Open)
uv run reflex             # Reflex Studio at http://127.0.0.1:8788
```

Benchmark (the task file comes from the gated Hugging Face dataset `osunlp/Online-Mind2Web`):

```bash
python -m scripts.benchmark.sample_online_mind2web Online_Mind2Web.json pilot40.json
python -m scripts.benchmark.run_online_mind2web pilot40.json runs/pilot40 --worker-id 0 --total-workers 1
```

Tests run offline: `uv run pytest -q`.

## Known limitations

- Long form flows and stacked filters often exceed the 40-step budget.
- When a site does not have exactly what was asked, the agent sometimes still answers with the closest match. The answer check is meant to catch these, but it does so only some of the time.
- Sites behind Cloudflare or geo-blocks cannot be used. Reflex reports the block and does not try to get around it.

## Experimental modes (not used by default)

Reflex started as a two-system design: a fast "System 1" for confident clicks and typing, with Muse Spark as "System 2" for everything else. System 1 runs either through Jev (OpenRouter) or through Turnike, a locally fine-tuned Qwen3.5-4B. On our tasks neither hybrid mode beat Muse alone, so the default mode is Muse-only (`s2_only`). The hybrid modes stay selectable (`MOD` in Studio, `--mode` in `scripts/eval.py`). Turnike is not a Meta model and has no part in the default pipeline.

## MCP Server (Claude Code / Antigravity)

Install with the `mcp` extra and register the stdio server in Claude Code:

```bash
pip install -e ".[mcp]"
claude mcp add reflex -- /ABSOLUTE/PATH/.venv/bin/reflex-mcp
```

Each `reflex-mcp` process runs one browser session at a time. Call `reflex_stop` before starting a new session with `reflex_start`.

## Design Influences

The task ledger (planning + memory) adapts ideas from these works; no third-party code is copied:
- **Plan-and-Act** (Erdogan et al., 2025), *Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks*, [arXiv:2503.09572](https://arxiv.org/abs/2503.09572): a separate planner whose plan is revised as execution progresses. Reflex's System 2 writes the plan once at the start and updates it at milestones.
- **StructuredAgent** (Lobo et al., 2026), *StructuredAgent: Planning with AND/OR Trees for Long-Horizon Web Tasks*, [arXiv:2603.05294](https://arxiv.org/abs/2603.05294): explicit, structured plan state instead of free-form notes. Reflex keeps a flat subgoal list with completion conditions and source-attributed findings.
- **BrowseMaster** (Pang et al., 2025), *Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair*, [arXiv:2508.09129](https://arxiv.org/abs/2508.09129): keeping the reasoning model's context focused while execution is delegated, and preserving where each fact came from.
- **Magentic-One** (Microsoft, 2024, MIT License): task ledger, progress ledger and stuck counter.

Built with Muse Spark Open.

## License

MIT, see [LICENSE](LICENSE).
