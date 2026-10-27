"""Unit tests for Task G13: cdp.py wait and settling logic cleanup."""
from pathlib import Path
import json
import subprocess
import shutil
import pytest


CDP_PATH = Path(__file__).resolve().parent.parent / "reflex" / "browser" / "cdp.py"


def test_forbidden_heuristics_removed_from_cdp():
    """cdp.py must not contain scenario regexes or domain-specific settling hacks."""
    content = CDP_PATH.read_text(encoding="utf-8")

    forbidden = [
        "isGoogleSearch",
        "#rso",
        "rouletteSpinning",
        "google_vignette",
        "istem girin",
        "ask gemini",
        "message chatgpt",
        "durdur",
        "search|ara",
    ]
    for pat in forbidden:
        assert pat not in content, f"Forbidden heuristic '{pat}' found in cdp.py"


def test_general_quiescence_semantics_present():
    """cdp.py must use general page quiescence (DOM mutations, resource entries, aria-busy)."""
    content = CDP_PATH.read_text(encoding="utf-8")

    assert "MutationObserver" in content
    assert "getEntriesByType('resource')" in content
    assert '[aria-busy="true"]' in content
    assert "e.type === 'search' || e.getAttribute('role') === 'searchbox'" in content
    assert "e.getAttribute('role') === 'combobox'" in content


def test_search_detection_logic_execution():
    """Verify search input detection correctly identifies search vs combobox in Node.js."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed")

    test_script = """
    const isSearchFn = e => {
      if (!e) return false;
      if (e.getAttribute('role') === 'combobox' || e.getAttribute('aria-autocomplete') || e.getAttribute('aria-haspopup') === 'listbox') {
        return false;
      }
      return e.type === 'search' || e.getAttribute('role') === 'searchbox' || Boolean(e.closest && e.closest('form[role="search"]'));
    };

    // 1. type="search" -> true
    const searchInput = { type: 'search', getAttribute: attr => null, closest: sel => null };
    // 2. role="searchbox" -> true
    const searchboxRole = { type: 'text', getAttribute: attr => attr === 'role' ? 'searchbox' : null, closest: sel => null };
    // 3. input inside form[role="search"] -> true
    const formSearch = { type: 'text', getAttribute: attr => null, closest: sel => sel.includes('form') ? {} : null };
    // 4. combobox with search type -> false (avoid auto-Enter on autocomplete popups)
    const combobox = { type: 'search', getAttribute: attr => attr === 'role' ? 'combobox' : null, closest: sel => null };
    // 5. standard text input -> false
    const normalText = { type: 'text', getAttribute: attr => null, closest: sel => null };

    console.log(JSON.stringify({
      search: isSearchFn(searchInput),
      searchbox: isSearchFn(searchboxRole),
      formSearch: isSearchFn(formSearch),
      combobox: isSearchFn(combobox),
      normal: isSearchFn(normalText)
    }));
    """

    res = subprocess.run([node_bin, "-e", test_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Node script error: {res.stderr}"

    data = json.loads(res.stdout.strip())
    assert data["search"] is True
    assert data["searchbox"] is True
    assert data["formSearch"] is True
    assert data["combobox"] is False
    assert data["normal"] is False
