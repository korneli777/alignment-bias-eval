"""Framing-registry invariants.

These lock the properties the paper states in §2.3, so the registry cannot
drift away from the write-up without a test failing.
"""

from __future__ import annotations

from collections import Counter

from biaseval.benchmarks.framings import BBQ_FRAMINGS, CROWS_FRAMINGS, FRAMINGS


def test_registry_matches_the_paper():
    """§2.3: six conditions, five of them with a multiple-choice form."""
    assert len(FRAMINGS) == 6
    assert len(CROWS_FRAMINGS) == 6
    assert len(BBQ_FRAMINGS) == 5
    assert Counter(f.ptype for f in FRAMINGS.values()) == {
        "persona": 2, "instruction": 2, "non_persona": 2,
    }


def test_bbq_framings_are_those_with_an_mcq_mapping():
    assert set(BBQ_FRAMINGS) == {
        fid for fid, f in FRAMINGS.items() if f.bbq_instruction is not None
    }
    assert "jb_fluency" not in BBQ_FRAMINGS  # no coherent multiple-choice form


def test_fluency_control_refuses_a_bbq_prompt():
    """The control has no MCQ mapping, so asking for one must fail loudly."""
    try:
        FRAMINGS["jb_fluency"].bbq_user_message("body")
    except ValueError:
        return
    raise AssertionError("jb_fluency should refuse to build a BBQ prompt")


def test_bbq_prompt_places_the_instruction_before_the_item():
    """The framing must lead; the item follows it, separated by a blank line."""
    msg = FRAMINGS["jb_persona"].bbq_user_message("CONTEXT\nA) x\nB) y\nC) z")
    preamble, item = msg.split("\n\n", 1)
    assert preamble.startswith(FRAMINGS["jb_persona"].preamble)
    assert item == "CONTEXT\nA) x\nB) y\nC) z"


def test_every_framing_is_well_formed():
    for fid, f in FRAMINGS.items():
        assert fid.startswith("jb_"), fid
        assert f.preamble and not f.preamble.endswith(" "), fid
        assert f.ptype in {"persona", "instruction", "non_persona"}, fid
        assert f.grounding, fid
