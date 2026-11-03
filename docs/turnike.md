# Turnike: an experimental System 1 model (research preview)

**Status:** research preview. Turnike is not used in Reflex's default mode (`s2_only`, Muse Spark decides every step), and its weights are not part of this hackathon submission. They are still experimental, and we are checking the licenses of some training sources.

## What it is

Turnike is a family of small decision models we fine-tuned from Qwen/Qwen3.5-4B (LoRA r=16, merged and stored in FP8 for local inference on an RTX 40/50-class GPU). For a browser step it reads the same text Reflex gives Muse Spark (goal, URL, the numbered list of page elements, recent actions) and returns probability distributions instead of free text:

- which operation comes next (`CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_DOWN`, `WAIT`, `DONE`, `BLOCKED`), and
- which element it acts on.

Reflex serves it with [`reflex/turnike_server.py`](../reflex/turnike_server.py) (`POST /v1/systemone`).

## Dates

Turnike predates the hackathon. We trained it from Oct 14 to Oct 25, 2026, in independent experiments before we knew about the hackathon. The version Reflex uses, `turnike-4b-web2`, was finished on Oct 25. During the hackathon we connected it to Reflex through the confidence gate below, measured it, and trained one more version on human demonstrations recorded on Oct 28 (`turnike-4b-ml-t1`, Oct 29).

## Training data (`turnike-4b-web2`)

342,466 labeled questions, sampled with group weights (browser questions about 10% of the draw, up to 8,192 tokens each):

- **Browser actions:** 12,096 questions built from the Mind2Web training split (CC BY 4.0), labeled with the human annotators' actions.
- **General decisions:** classification, question answering and rule-following questions in English and Turkish, from open datasets and our own templates. Part of these labels were written by Claude (Sonnet and Opus).
- **No Muse or other Meta model outputs** and no outputs of the Jev model were used.

## How Reflex routes steps to it

In the hybrid modes, a calibrated gate ([`reflex/gate.py`](../reflex/gate.py), thresholds in [`reflex/config.py`](../reflex/config.py)) lets Turnike act only when it is sure, and hands the step to Muse Spark otherwise:

- operation confidence below 0.42 for click, type and select, 0.15 for scroll and wait, 0.72 for "done";
- target confidence below 0.32, or the top two targets closer than 0.04 (an ambiguous target);
- joint confidence P(operation) × P(target) below 0.20, or below 0.60 for a form submit;
- required form fields still empty, a task constraint to check, or Turnike reporting that it is blocked;
- after acting: no visible progress, a loop or an oscillation, or 12 steps in a row without Muse on a planned task.

In strict mode, Muse Spark also decides scrolling. Irreversible actions (payments, submissions) always go to the user for confirmation, whichever model proposed them. The thresholds are fixed and hand-calibrated; they are not learned from past success.

## Measurements

- **19 real-site tasks, each run twice (Oct 29), with the cloud System 1 (Jev):** Muse alone 36/38, strict hybrid 33/38, the original flexible hybrid 14/19. The small model's confident mistakes cost extra recovery steps, and Muse Spark 1.3 answers in about 3 s per step, so there is little time left to save.
- **6 tasks, each run twice (Oct 31), strict hybrid with Turnike:** 12/12 finished, Turnike took about one step in five (mostly the first search and a few clicks), and the runs were about 10–15% faster than Muse alone on the same tasks (small sample; the Muse-only runs were measured a day earlier).

That is why Muse-only is the default. The hybrid modes stay selectable in Studio (`Mode`) and in `scripts/eval.py --mode`.

## Next steps

Routing that learns from history (how often a kind of action succeeded on similar pages) instead of fixed thresholds, more browser training data with human labels, and a held-out evaluation before any weight release.
