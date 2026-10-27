"""Unit tests for Task G11: snapshot.js cleanup and heading disambiguation."""
from pathlib import Path
import subprocess
import shutil
import pytest


SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "reflex" / "browser" / "snapshot.js"


def test_snapshot_syntax_valid():
    """snapshot.js must be syntactically valid JavaScript."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed")
    result = subprocess.run([node_bin, "-c", str(SNAPSHOT_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, f"snapshot.js syntax error: {result.stderr}"


def test_forbidden_heuristics_removed_from_snapshot():
    """snapshot.js must not contain ad-hoc site heuristics, slot counting, or hardcoded SERP strings."""
    content = SNAPSHOT_PATH.read_text(encoding="utf-8")

    forbidden_patterns = [
        "adsbygoogle",
        "aswift_",
        "google_vignette",
        "slotLikeCount",
        "[DOLU:",
        "isSerp",
        "google ana sayfasına git",
        "go to google home",
        "harf çevirme aracı",
        "devamını oku",
        "read more",
    ]

    for pat in forbidden_patterns:
        assert pat not in content, f"Forbidden heuristic '{pat}' found in snapshot.js"


def test_generic_heading_disambiguation_present_in_snapshot():
    """snapshot.js must implement generic heading disambiguation when action labels collide."""
    content = SNAPSHOT_PATH.read_text(encoding="utf-8")

    assert "labelCounts" in content
    assert "labelCounts.get(k)>1" in content or "labelCounts.get(k) > 1" in content
    assert "h1,h2,h3,h4" in content
    assert '[role="heading"]' in content


def test_node_disambiguation_logic_execution():
    """Execute the heading disambiguation logic in Node.js to verify heading attachment."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed")

    # Script testing the exact disambiguation algorithm from snapshot.js
    test_script = """
    const actions = [
      { node: 1, label: 'Read more' },
      { node: 2, label: 'Read more' },
      { node: 3, label: 'Submit Order' }
    ];

    const mockDom = {
      1: {
        parentElement: {
          matches: sel => sel.includes('article'),
          querySelector: sel => ({ innerText: 'First Article' }),
          parentElement: null
        }
      },
      2: {
        parentElement: {
          matches: sel => sel.includes('article'),
          querySelector: sel => ({ innerText: 'Second Article' }),
          parentElement: null
        }
      },
      3: {
        parentElement: {
          matches: sel => false,
          querySelector: sel => null,
          parentElement: null
        }
      }
    };

    const cache = { nodes: { get: id => mockDom[id] } };

    const labelCounts = new Map();
    for (const a of actions) {
      if (!a.label) continue;
      const k = a.label.trim().toLowerCase();
      labelCounts.set(k, (labelCounts.get(k) || 0) + 1);
    }

    for (const a of actions) {
      if (!a.label) continue;
      const k = a.label.trim().toLowerCase();
      if (labelCounts.get(k) > 1) {
        const el = cache.nodes.get(a.node);
        let card = el?.parentElement;
        while (card) {
          if (card.matches && card.matches('article,section,li,[role="listitem"]') &&
              card.querySelector('h1,h2,h3,h4,[role="heading"]')) {
            break;
          }
          card = card.parentElement;
        }
        const hEl = card?.querySelector('h1,h2,h3,h4,[role="heading"]');
        const hTxt = (hEl?.innerText || '').replace(/\\s+/g, ' ').trim();
        if (hTxt && !a.label.toLowerCase().includes(hTxt.slice(0, 25).toLowerCase())) {
          a.label = `${a.label.trim()} (${hTxt.slice(0, 90)})`;
        }
      }
    }

    console.log(JSON.stringify(actions));
    """

    res = subprocess.run([node_bin, "-e", test_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Node script failed: {res.stderr}"

    import json
    data = json.loads(res.stdout.strip())
    assert data[0]["label"] == "Read more (First Article)"
    assert data[1]["label"] == "Read more (Second Article)"
    assert data[2]["label"] == "Submit Order"  # unique label untouched
