"""Draw the stratified 40-task Online-Mind2Web pilot used in the README.

Online_Mind2Web.json comes from the gated Hugging Face dataset `osunlp/Online-Mind2Web`; it is not redistributed here.
The 8 tasks used while diagnosing failures (before the pilot was drawn) are excluded so the pilot was never tuned on.

    python -m scripts.benchmark.sample_online_mind2web Online_Mind2Web.json pilot40.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 20261102
PER_LEVEL = {"easy": 11, "medium": 19, "hard": 10}
DIAGNOSIS_TASK_IDS = {
    "9ed3827266b3b804f485859c3d00401e_070826",
    "1bc154377120ec15b18dbabdba49c741_051526",
    "dcd26e662a616d373ddd339747c6ce5b",
    "3c1ffc3f494e423b3c434c79e35da8f3_110325",
    "85b284c18d7e78c9b5a9e074e7aa3b98_051526",
    "7562d9b4e4829a44245aafce2e1f62db",
    "515f2e5811cfdd5e0e669e40f17886d8",
    "f389398d2eeb29e5571e00439c57eb76",
}


def main(data_path: str, out_path: str) -> None:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    pool = [t for t in data if t["task_id"] not in DIAGNOSIS_TASK_IDS]
    random.seed(SEED)
    picked = [t for level, n in PER_LEVEL.items() for t in random.sample([p for p in pool if p["level"] == level], n)]
    tasks = [
        {
            "task_id": t["task_id"],
            "website": t["website"],
            "confirmed_task": t["confirmed_task"],
            "level": t["level"],
            "reference_length": t.get("reference_length"),
        }
        for t in picked
    ]
    Path(out_path).write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
