"""Offline tests for the rules chunker. No network, no embeddings."""

from pathlib import Path

import pytest

from ingest.load_rules import chunk_by_section, chunk_fixed

# Mirrors the layout pypdf extracts from the official PDF: blank lines are
# " \n", the table of contents repeats headings, lines are hard-wrapped, and
# cross-references can wrap onto the start of a line.
SAMPLE = """Magic: The Gathering Comprehensive Rules 
 
Contents 
 
1. Game Concepts 
100. General 
7. Additional Rules 
702. Keyword Abilities 
 
Glossary 
 
Credits 
1. Game Concepts 
 
100. General 
 
100.1. These Magic rules apply to any Magic game with two or more players, including two-player 
games and multiplayer games. 
 
100.1a A two-player game is a game that begins with only two players. 
 
7. Additional Rules 
 
702. Keyword Abilities 
 
702.19. Trample 
 
702.19a Trample is a static ability that modifies the rules for assigning an attacking creature's combat damage. 
 
702.19b The controller of an attacking creature with trample first assigns damage. See rule 
202.2. 
 
Example: A 6/6 green creature with trample is blocked by a 2/2 creature. 
 
702.19c Multiple instances of trample on the same creature are redundant. 
 
Glossary 
 
Trample 
A keyword ability that modifies how a creature assigns combat damage. See rule 702.19, 
"Trample." 
 
Credits 
 
Magic: The Gathering Original Game Design: Richard Garfield 
"""


def _by_section(chunks):
    return {c["metadata"]["section"]: c["text"] for c in chunks if "section" in c["metadata"]}


def test_skips_table_of_contents_and_credits():
    chunks = chunk_by_section(SAMPLE)
    sections = [c["metadata"].get("section") for c in chunks if "section" in c["metadata"]]
    assert sections == ["100.1", "702.19"]
    assert not any("Richard Garfield" in c["text"] for c in chunks)


def test_rule_grouped_with_subrules_under_section_header():
    trample = _by_section(chunk_by_section(SAMPLE))["702.19"]
    assert trample.startswith("702. Keyword Abilities\n702.19. Trample")
    for sub in ("702.19a", "702.19b", "702.19c"):
        assert sub in trample


def test_wrapped_reference_and_example_stay_in_their_rule():
    trample = _by_section(chunk_by_section(SAMPLE))["702.19"]
    assert "See rule 202.2." in trample
    assert "Example: A 6/6 green creature" in trample


def test_chapter_headings_dropped():
    first = _by_section(chunk_by_section(SAMPLE))["100.1"]
    assert "7. Additional Rules" not in first


def test_glossary_terms_become_chunks():
    glossary = [c for c in chunk_by_section(SAMPLE) if "glossary_term" in c["metadata"]]
    assert len(glossary) == 1
    assert glossary[0]["metadata"]["glossary_term"] == "Trample"
    assert glossary[0]["text"].startswith("Glossary — Trample: A keyword ability")


def test_long_rule_split_between_subrules_with_parent_context():
    chunks = chunk_by_section(SAMPLE, max_words=30)
    by_section = _by_section(chunks)
    assert len([s for s in by_section if s.startswith("702.19")]) > 1
    for section, text in by_section.items():
        if section.startswith("702.19"):
            assert text.startswith("702. Keyword Abilities")
            assert "702.19. Trample" in text
    # Every subrule lands in exactly one chunk.
    joined = "\n".join(by_section.values())
    for sub in ("702.19a", "702.19b", "702.19c"):
        assert joined.count(f"{sub} ") == 1


def test_no_rule_numbers_returns_empty():
    assert chunk_by_section("just some prose\n \nwith no rules") == []


def test_chunk_fixed_overlap_and_guard():
    chunks = chunk_fixed(" ".join(str(i) for i in range(10)), chunk_size=4, overlap=1)
    assert [c["text"] for c in chunks][:2] == ["0 1 2 3", "3 4 5 6"]
    # overlap >= chunk_size must still terminate
    assert chunk_fixed("a b c", chunk_size=2, overlap=5)


PDF = Path(__file__).resolve().parent.parent / "mtg-rules.pdf"


@pytest.mark.skipif(not PDF.exists(), reason="mtg-rules.pdf not present")
def test_real_pdf_shape():
    from ingest.load_rules import extract_text_from_pdf

    chunks = chunk_by_section(extract_text_from_pdf(PDF))
    sections = [c["metadata"]["section"] for c in chunks if "section" in c["metadata"]]
    assert len(sections) == len(set(sections)), "duplicate rule chunks (TOC leaked?)"
    assert any(s.startswith("702.19") for s in sections)
    assert sum("glossary_term" in c["metadata"] for c in chunks) > 100
