"""Adversarial Stress Test Suite for Milestone M1 (Reflex Task Ledger & Config).

Covers:
1. Malformed JSON, deeply nested structures, Unicode edge cases, extreme budgets.
2. Candidate deduplication with varied casing, whitespace, conflicting attributes.
3. 2500 character summarization boundary conditions (2499, 2500, 2501 chars, lifecycle).
4. Domain extraction with complex URLs, IP addresses, subdomains, missing schemes, pseudo-schemes.
5. Subgoal lifecycle and state machine edge cases.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from reflex.planner import (
    Budget,
    Subgoal,
    SubgoalResult,
    TaskLedger,
    _extract_domain,
    parse_task_ledger,
)

# ==============================================================================
# 1. MALFORMED JSON, DEEP NESTING, UNICODE, EXTREME BUDGETS & FUZZING
# ==============================================================================

class TestMalformedJsonAndFuzzing:
    """Stress tests for parse_task_ledger resilience and schema boundaries."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            "   ",
            "not json at all",
            "{",
            '{"needs_plan": true, "subgoals": [}',
            "null",
            "123",
            "true",
            "false",
            "[]",
            '["needs_plan", true]',
        ],
    )
    def test_malformed_json_strings_fallback_gracefully(self, invalid_input: str):
        """Malformed or non-dict JSON strings must fallback to needs_plan=False without crashing."""
        ledger = parse_task_ledger(invalid_input)
        assert isinstance(ledger, TaskLedger)
        assert ledger.needs_plan is False

    @pytest.mark.parametrize(
        "non_dict_object",
        [
            None,
            123,
            45.67,
            True,
            False,
            [],
            [{"needs_plan": True}],
            ("tuple", 1),
            object(),
        ],
    )
    def test_non_dict_python_objects_fallback_gracefully(self, non_dict_object: Any):
        """Non-dict and non-str Python objects must fallback to needs_plan=False without crashing."""
        ledger = parse_task_ledger(non_dict_object)
        assert isinstance(ledger, TaskLedger)
        assert ledger.needs_plan is False

    def test_deeply_nested_json_structures(self):
        """Deeply nested structures in attributes and subgoals do not cause RecursionError."""
        # Create a 50-level nested dictionary
        nested: dict[str, Any] = {"leaf": "deep_value"}
        for i in range(50):
            nested = {f"level_{i}": nested}

        payload = {
            "needs_plan": True,
            "finish_criteria": "Deep nesting test",
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Deep nest subgoal",
                    "results": [
                        {"fact": "Nested fact", "source_url": "https://example.com", "step": 1}
                    ],
                }
            ],
            "candidates": [
                {
                    "item": "Deep Candidate",
                    "source_url": "https://example.com/item",
                    "attributes": {"nested_meta": nested},
                }
            ],
        }

        ledger = parse_task_ledger(payload)
        assert ledger.needs_plan is True
        assert len(ledger.candidates) == 1
        assert "nested_meta" in ledger.candidates[0].attributes
        # format_for_s2 should not crash with nested dicts
        s2_text = ledger.format_for_s2()
        assert "Deep Candidate" in s2_text

    @pytest.mark.parametrize(
        "budget_input,expected_sources,expected_steps",
        [
            ({"sources": 0, "max_steps": 0}, 1, 30),  # 0 or 30 evaluates to 30
            ({"sources": -5, "max_steps": -100}, 1, 1),
            ({"sources": 10**18, "max_steps": 10**18}, 10**18, 10**18),
            ({"sources": "invalid", "max_steps": "nan"}, 1, 30),
            ({"sources": None, "max_steps": None}, 1, 30),
            ({"sources": "5", "max_steps": "15"}, 5, 15),
            ("not-a-dict", 1, 30),
            (12345, 1, 30),
            ([], 1, 30),
        ],
    )
    def test_extreme_and_malformed_budgets(
        self, budget_input: Any, expected_sources: int, expected_steps: int
    ):
        """Budget parsing enforces minimum boundaries and handles corrupted types gracefully."""
        payload = {
            "needs_plan": True,
            "budget": budget_input,
        }
        ledger = parse_task_ledger(payload)
        assert ledger.budget.sources == expected_sources
        assert ledger.budget.max_steps == expected_steps

    def test_unicode_edge_cases_across_all_fields(self):
        """TaskLedger gracefully handles Turkish, CJK, Arabic, RTL, and Emoji characters."""
        payload = {
            "needs_plan": True,
            "clarify": {
                "question": "Hangi mağazalar aransın? (İstanbul / Kadıköy) 🛍️",
                "options": ["Mağaza 1 (₺)", "Mağaza 2 (USD $)", "الكل 🌐"],
                "default": "Mağaza 1 (₺)",
            },
            "finish_criteria": "En ucuz Türkçe kitap bulundu. 📚",
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Kitap Yurdu & D&R araştırması: 'Saatleri Ayarlama Enstitüsü' 🕰️",
                    "done_when": "Fiyat doğrulandı",
                    "status": "active",
                    "results": [
                        {
                            "fact": "Fiyat: 145,50 ₺ (Stokta var) — עִבְרִית / 日本語",
                            "source_url": "https://www.kitapyurdu.com/kitap/saatleri-ayarlama-enstitusu/12345.html",
                            "step": 3,
                        }
                    ],
                }
            ],
            "candidates": [
                {
                    "item": "Saatleri Ayarlama Enstitüsü (Dergâh Yayınları) 📖",
                    "source_url": "https://www.kitapyurdu.com/kitap/12345",
                    "attributes": {"fiyat": "145,50 ₺", "puan": "4.9 ⭐", "yazar": "Ahmet Hamdi Tanpınar"},
                }
            ],
        }

        ledger = parse_task_ledger(payload)
        assert ledger.needs_plan is True
        assert ledger.clarify is not None
        assert "Kadıköy" in ledger.clarify.question
        assert ledger.subgoals[0].results[0].step == 3
        assert "Ahmet Hamdi Tanpınar" in ledger.candidates[0].attributes["yazar"]

        s2_formatted = ledger.format_for_s2()
        assert "145,50 ₺" in s2_formatted

        final_ans = ledger.synthesize_final_answer("Kitap fiyatı bul")
        assert "Ahmet Hamdi Tanpınar" in final_ans
        assert "145,50 ₺" in final_ans

    # --- Vulnerability Demonstrations in Malformed JSON Parsing ---

    def test_clarify_options_null_causes_type_error(self):
        """When LLM returns 'clarify': {'question': 'Q?', 'options': null}, parse_task_ledger should not crash."""
        payload = {
            "needs_plan": True,
            "clarify": {"question": "Kaç kaynak?", "options": None},
        }
        # In vulnerable code: TypeError: 'NoneType' object is not iterable at line 360
        ledger = parse_task_ledger(payload)
        assert ledger.needs_plan is True
        assert ledger.clarify is not None
        assert ledger.clarify.options == []

    def test_subgoal_result_step_non_int_causes_value_error(self):
        """When LLM returns non-integer step, parse_task_ledger should safely fallback to 0."""
        payload = {
            "needs_plan": True,
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1",
                    "results": [{"fact": "Found fact", "step": "step 1"}],
                }
            ],
        }
        # In vulnerable code: ValueError: invalid literal for int() with base 10: 'step 1' at line 377
        ledger = parse_task_ledger(payload)
        assert ledger.needs_plan is True
        assert len(ledger.subgoals[0].results) == 1
        assert ledger.subgoals[0].results[0].step == 0

    def test_candidate_attributes_non_dict_causes_value_error(self):
        """When LLM returns non-dict attributes string, parse_task_ledger should fallback to empty dict."""
        payload = {
            "needs_plan": True,
            "candidates": [
                {"item": "Phone", "attributes": "cheapest option"}
            ],
        }
        # In vulnerable code: ValueError: dictionary update sequence element #0 has length 1 at line 415
        ledger = parse_task_ledger(payload)
        assert ledger.needs_plan is True
        assert len(ledger.candidates) == 1
        assert ledger.candidates[0].attributes == {}

    def test_random_fuzzing_resilience(self):
        """Property check: Fuzzed corrupted dictionaries should either parse cleanly or return needs_plan=False, never crash."""
        random.seed(42)
        crashes: list[tuple[dict, str]] = []

        keys = ["needs_plan", "clarify", "subgoals", "candidates", "finish_criteria", "budget", "unknown_key"]
        corruptions = [
            None,
            123,
            "random_string",
            [],
            {},
            [None, 123, {}],
            {"sub": None},
        ]

        for _ in range(50):
            num_keys = random.randint(1, len(keys))
            selected_keys = random.sample(keys, num_keys)
            fuzzed_dict: dict[str, Any] = {}
            for k in selected_keys:
                fuzzed_dict[k] = random.choice(corruptions)

            # Ensure needs_plan might be True to reach deeper parsers
            if random.random() > 0.5:
                fuzzed_dict["needs_plan"] = True

            try:
                parse_task_ledger(fuzzed_dict)
            except Exception as exc:  # noqa: BLE001
                crashes.append((fuzzed_dict, f"{type(exc).__name__}: {exc}"))

        # Report fuzzing results
        if crashes:
            pytest.fail(f"Fuzzer exposed {len(crashes)} uncaught crashes in parse_task_ledger:\n" +
                        "\n".join(f"- Payload: {p} => {err}" for p, err in crashes[:5]))


# ==============================================================================
# 2. CANDIDATE DEDUPLICATION ADVERSARIAL TESTS
# ==============================================================================

class TestCandidateDeduplicationAdversarial:
    """Stress tests for candidate table deduplication, whitespace, and attribute conflicts."""

    def test_casing_deduplication(self):
        """Candidates with different casing are deduplicated properly."""
        ledger = TaskLedger(needs_plan=True)
        ledger.add_candidate("Apple iPhone 15 Pro", "https://apple.com", {"storage": "128GB"})
        ledger.add_candidate("apple iphone 15 pro", "https://amazon.com", {"color": "Natural Titanium"})
        ledger.add_candidate("APPLE IPHONE 15 PRO", "https://bestbuy.com", {"price": "$999"})

        assert len(ledger.candidates) == 1
        cand = ledger.candidates[0]
        assert cand.item == "Apple iPhone 15 Pro"
        assert cand.attributes == {"storage": "128GB", "color": "Natural Titanium", "price": "$999"}

    def test_leading_trailing_whitespace_deduplication(self):
        """Candidates with leading/trailing whitespace are deduplicated."""
        ledger = TaskLedger(needs_plan=True)
        ledger.add_candidate("  Sony WH-1000XM5  ", "https://sony.com", {"anc": "yes"})
        ledger.add_candidate("Sony WH-1000XM5", "https://amazon.com", {"price": "$399"})

        assert len(ledger.candidates) == 1
        assert ledger.candidates[0].item == "Sony WH-1000XM5"
        assert ledger.candidates[0].attributes == {"anc": "yes", "price": "$399"}

    def test_internal_irregular_whitespace_collapsing(self):
        """Candidates with varied internal whitespace should be recognized as identical."""
        ledger = TaskLedger(needs_plan=True)
        ledger.add_candidate("Sony WH-1000XM5", "https://sony.com", {"model": "XM5"})
        ledger.add_candidate("Sony   WH-1000XM5", "https://amazon.com", {"price": "$348"})

        # In current code: len(ledger.candidates) == 1 because internal whitespace is normalized
        assert len(ledger.candidates) == 1
        assert ledger.candidates[0].attributes["price"] == "$348"

    def test_conflicting_attributes_latest_overwrites(self):
        """When conflicting attribute values arrive, newer observations overwrite earlier ones."""
        ledger = TaskLedger(needs_plan=True)
        ledger.add_candidate("Dell XPS 15", "https://dell.com", {"price": "$1499", "cpu": "i7"})
        ledger.add_candidate("Dell XPS 15", "https://bestbuy.com", {"price": "$1399", "ram": "16GB"})

        assert len(ledger.candidates) == 1
        cand = ledger.candidates[0]
        assert cand.attributes["price"] == "$1399"
        assert cand.attributes["cpu"] == "i7"
        assert cand.attributes["ram"] == "16GB"

    def test_candidate_multi_source_url_tracking(self):
        """A candidate observed across multiple websites should track multi-source evidence."""
        ledger = TaskLedger(needs_plan=True, budget=Budget(sources=2))
        ledger.add_candidate("Logitech MX Master 3S", "https://amazon.com/dp/B09HM94VDS", {"price": "$99"})
        ledger.add_candidate("Logitech MX Master 3S", "https://bestbuy.com/site/logitech", {"price": "$99"})

        # Currently candidate source_url remains amazon.com, and verify_done only sees 1 source
        assert ledger.candidates[0].source_url == "https://amazon.com/dp/B09HM94VDS"
        # In a robust multi-source comparison architecture, sources from candidates should count towards budget
        is_done, reason, _ = ledger.verify_done()
        assert is_done is True, f"Multi-source candidate failed verification: {reason}"


# ==============================================================================
# 3. 2500 CHARACTER SUMMARIZATION BOUNDARY CONDITIONS
# ==============================================================================

class TestSummarizationBoundaries:
    """Stress tests for the 2500 character summarization boundary."""

    def test_boundary_exactly_2499_chars(self):
        """Under threshold (2499 chars): summarization is NOT triggered."""
        ledger = TaskLedger(needs_plan=True)
        sg = Subgoal(id="sg_1", text="Subgoal 1", status="completed")
        sg.results.append(SubgoalResult(fact="X" * 2499, source_url="https://site.com/1", step=1))
        ledger.subgoals.append(sg)

        assert ledger.total_results_chars() == 2499
        summarized = ledger.summarize_completed_results(threshold=2500)
        assert summarized is False
        assert ledger.total_results_chars() == 2499

    def test_boundary_exactly_2500_chars(self):
        """At threshold (exactly 2500 chars): summarization is NOT triggered (must strictly exceed)."""
        ledger = TaskLedger(needs_plan=True)
        sg = Subgoal(id="sg_1", text="Subgoal 1", status="completed")
        sg.results.append(SubgoalResult(fact="X" * 2500, source_url="https://site.com/1", step=1))
        ledger.subgoals.append(sg)

        assert ledger.total_results_chars() == 2500
        summarized = ledger.summarize_completed_results(threshold=2500)
        assert summarized is False
        assert ledger.total_results_chars() == 2500

    def test_boundary_exactly_2501_chars(self):
        """Exceeding threshold (2501 chars): summarization triggers and reduces character count."""
        ledger = TaskLedger(needs_plan=True)
        sg = Subgoal(id="sg_1", text="Subgoal 1", status="completed")
        sg.results.append(SubgoalResult(fact="X" * 2501, source_url="https://site.com/1", step=1))
        ledger.subgoals.append(sg)

        assert ledger.total_results_chars() == 2501
        summarized = ledger.summarize_completed_results(threshold=2500)
        assert summarized is True
        assert ledger.total_results_chars() <= 300
        assert ledger.subgoals[0].results[0].source_url == "https://site.com/1"
        assert ledger.subgoals[0].results[0].fact.endswith("...")

    def test_active_and_pending_subgoals_never_summarized(self):
        """Only 'completed' subgoals are summarized; 'active' or 'pending' subgoals remain intact."""
        ledger = TaskLedger(needs_plan=True)
        sg_active = Subgoal(id="sg_active", text="Active SG", status="active")
        sg_active.results.append(SubgoalResult(fact="A" * 3000, source_url="https://active.com", step=2))
        ledger.subgoals.append(sg_active)

        assert ledger.total_results_chars() == 3000
        summarized = ledger.summarize_completed_results(threshold=2500)
        # Cannot summarize because no completed subgoals exist
        assert summarized is False
        assert ledger.total_results_chars() == 3000

    def test_strict_preservation_of_all_source_urls(self):
        """Summarization strictly preserves all unique source URLs across merged facts."""
        ledger = TaskLedger(needs_plan=True)
        sg = Subgoal(id="sg_1", text="Multi-source subgoal", status="completed")

        urls = [
            "https://alpha.org/article1",
            "https://beta.com/report2",
            "https://gamma.edu/paper3",
            "https://delta.gov/data4",
        ]

        # Add 3 facts per URL (12 facts total), each ~300 chars (total ~3600 chars)
        for u in urls:
            for i in range(3):
                sg.results.append(
                    SubgoalResult(fact=f"Fact {i} from {u}: " + ("x" * 250), source_url=u, step=i + 1)
                )
        ledger.subgoals.append(sg)

        assert ledger.total_results_chars() > 2500
        summarized = ledger.summarize_completed_results(threshold=2500)
        assert summarized is True

        # Exactly 4 condensed results, one per unique source URL
        assert len(sg.results) == 4
        result_urls = {r.source_url for r in sg.results}
        assert result_urls == set(urls)

    def test_re_summarization_stability(self):
        """Calling summarize_completed_results multiple times is stable and idempotent."""
        ledger = TaskLedger(needs_plan=True)
        sg1 = Subgoal(id="sg_1", text="SG 1", status="completed")
        sg1.results.append(SubgoalResult(fact="Y" * 2600, source_url="https://test.com", step=1))
        ledger.subgoals.append(sg1)

        # First pass
        assert ledger.summarize_completed_results(threshold=2500) is True
        chars_pass1 = ledger.total_results_chars()
        assert chars_pass1 <= 300

        # Second pass (total chars now <= 2500)
        assert ledger.summarize_completed_results(threshold=2500) is False
        assert ledger.total_results_chars() == chars_pass1


# ==============================================================================
# 4. DOMAIN EXTRACTION & BUDGET VERIFICATION ADVERSARIAL TESTS
# ==============================================================================

class TestDomainExtractionAdversarial:
    """Stress tests for URL domain normalization, edge cases, and budget verification."""

    @pytest.mark.parametrize(
        "url,expected_domain",
        [
            ("https://example.com/path", "example.com"),
            ("http://www.example.com/path", "example.com"),
            ("https://sub.domain.co.uk:8080/query?a=1#hash", "sub.domain.co.uk"),
            ("http://192.168.1.100:3000/api/v1", "192.168.1.100"),
            ("//cdn.example.org/lib.js", "cdn.example.org"),
            ("ftp://files.example.com", "files.example.com"),
            ("", ""),
        ],
    )
    def test_standard_domain_extraction(self, url: str, expected_domain: str):
        """Standard and well-formed URLs extract normalized domains."""
        assert _extract_domain(url) == expected_domain

    # --- Vulnerability Demonstrations in Domain Extraction ---

    def test_missing_scheme_path_leakage_in_domain_extraction(self):
        """URLs without scheme (e.g. 'wikipedia.org/wiki/Ada_Lovelace') should extract only the hostname."""
        url1 = "wikipedia.org/wiki/Ada_Lovelace"
        url2 = "wikipedia.org/wiki/Charles_Babbage"

        d1 = _extract_domain(url1)
        d2 = _extract_domain(url2)

        # In vulnerable code: d1 == 'wikipedia.org/wiki/ada_lovelace', d2 == 'wikipedia.org/wiki/charles_babbage'
        assert d1 == "wikipedia.org", f"Expected 'wikipedia.org', got: {d1}"
        assert d2 == "wikipedia.org", f"Expected 'wikipedia.org', got: {d2}"

    def test_verify_done_falsely_satisfied_by_single_site_without_scheme(self):
        """Single-site pages without scheme must not satisfy a 2-source budget."""
        ledger = TaskLedger(
            needs_plan=True,
            budget=Budget(sources=2),
            subgoals=[
                Subgoal(
                    id="sg_1",
                    text="Ada",
                    status="completed",
                    results=[SubgoalResult(fact="Born 1815", source_url="wikipedia.org/wiki/Ada_Lovelace", step=1)],
                ),
                Subgoal(
                    id="sg_2",
                    text="Charles",
                    status="completed",
                    results=[SubgoalResult(fact="Born 1791", source_url="wikipedia.org/wiki/Charles_Babbage", step=2)],
                ),
            ],
        )

        is_done, reason, _ = ledger.verify_done()
        # Should be rejected because both results come from wikipedia.org (only 1 domain)
        assert is_done is False, f"Falsely verified done across single domain: {reason}"

    def test_ipv6_domain_extraction_truncation(self):
        """IPv6 address URLs must preserve the bracketed host or full IP."""
        url = "http://[2001:db8::1]:8080/status"
        domain = _extract_domain(url)
        # In vulnerable code: netloc.split(':')[0] yields '[2001'
        assert domain in ["[2001:db8::1]", "2001:db8::1"], f"IPv6 mangled to: {domain}"

    def test_userinfo_auth_domain_extraction(self):
        """URLs containing basic auth credentials must extract the hostname, not the username."""
        url = "http://admin:secret123@example.internal:8080/dashboard"
        domain = _extract_domain(url)
        # In vulnerable code: netloc.split(':')[0] yields 'admin'
        assert domain == "example.internal", f"Auth URL extracted username as domain: {domain}"

    def test_pseudo_schemes_rejected_as_valid_sources(self):
        """Non-web pseudo-schemes (about:blank, javascript:) must not count toward multi-source budget."""
        ledger = TaskLedger(
            needs_plan=True,
            budget=Budget(sources=2),
            subgoals=[
                Subgoal(
                    id="sg_1",
                    text="Nav",
                    status="completed",
                    results=[
                        SubgoalResult(fact="Blank tab", source_url="about:blank", step=1),
                        SubgoalResult(fact="JS hook", source_url="javascript:void(0)", step=2),
                    ],
                )
            ],
        )

        is_done, reason, _ = ledger.verify_done()
        # Should be rejected because 0 valid web domains were visited
        assert is_done is False, f"Pseudo-schemes falsely satisfied 2-source budget: {reason}"


# ==============================================================================
# 5. SUBGOAL ADVANCEMENT & STATE MACHINE ADVERSARIAL TESTS
# ==============================================================================

class TestSubgoalAdvancementAdversarial:
    """Stress tests for subgoal lifecycle state transitions."""

    def test_normal_subgoal_advancement_lifecycle(self):
        """Subgoals advance strictly from active -> completed while activating next pending."""
        ledger = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="First", status="active"),
                Subgoal(id="sg_2", text="Second", status="pending"),
                Subgoal(id="sg_3", text="Third", status="pending"),
            ],
        )

        assert ledger.get_active_subgoal().id == "sg_1"
        adv1 = ledger.advance_subgoal()
        assert adv1.id == "sg_2"
        assert ledger.subgoals[0].status == "completed"
        assert ledger.subgoals[1].status == "active"

        adv2 = ledger.advance_subgoal()
        assert adv2.id == "sg_3"
        assert ledger.subgoals[1].status == "completed"
        assert ledger.subgoals[2].status == "active"

        adv3 = ledger.advance_subgoal()
        assert adv3 is None
        assert ledger.subgoals[2].status == "completed"

    def test_advance_subgoal_when_active_is_failed(self):
        """When an active subgoal fails, advance_subgoal should activate the next pending without skipping it."""
        ledger = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="Failing Step", status="failed"),
                Subgoal(id="sg_2", text="Next Step To Attempt", status="pending"),
                Subgoal(id="sg_3", text="Final Step", status="pending"),
            ],
        )

        # Calling advance_subgoal when sg_1 is already failed
        next_sg = ledger.advance_subgoal()

        # Expected: sg_2 should become active and be returned to execute
        # In vulnerable code: sg_2 is marked 'completed' without running, and sg_3 is returned as active
        assert next_sg is not None
        assert next_sg.id == "sg_2", f"advance_subgoal skipped sg_2 and returned {next_sg.id}"
        assert ledger.subgoals[1].status == "active", f"sg_2 status became: {ledger.subgoals[1].status}"
