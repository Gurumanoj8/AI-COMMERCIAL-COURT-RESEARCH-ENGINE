"""
Database Operation Tests.
Validates table creation, parameterized queries, atomic transactions,
uniqueness constraints, and judgment retrieval.
"""

from services.db_service import db
from services.judgment_service import JudgmentService


def test_database_connection_status():
    """Verifies that the database service is initialized and responding."""
    health = db.check_health()
    assert health["status"] == "healthy"
    assert health["active_engine"] in ["postgresql", "sqlite_fallback"]


def test_judgments_retrieval():
    """Verifies that judgments can be queried from the database."""
    rows = db.fetch_all("SELECT id, case_title FROM judgments LIMIT 5;")
    assert isinstance(rows, list)
    # If sample data is present, rows should not be empty
    if rows:
        assert "id" in rows[0]
        assert "case_title" in rows[0]


def test_judgment_insertion_and_deletion():
    """Tests creating a judgment, querying it, and cleaning it up."""
    metadata = {
        "case_title": "Test Commercial Entity v. Regulatory Board",
        "court_name": "High Court of Delhi (Commercial Division)",
        "decision_date": "2024-03-01",
        "citation": "2024 SCC OnLine Del 999",
        "case_number": "CS (COMM) 999/2024",
        "subject": "Commercial Regulatory Dispute",
        "material_facts": "Test facts for unit verification.",
        "legal_issues": "1. Is this a valid unit test?",
        "judicial_reasoning": "Test reasoning confirms functionality.",
        "ratio_decidendi": "Test ratio decidendi.",
        "final_decision": "Test suit allowed.",
        "relevant_statutes": "Commercial Courts Act 2015"
    }
    extracted_text = "Full sample judgment text for unit testing purposes."
    chunks = [
        {"chunk_index": 0, "chunk_text": "Chunk 0 test content.", "page_number": 1},
        {"chunk_index": 1, "chunk_text": "Chunk 1 test content.", "page_number": 2}
    ]

    new_id = JudgmentService.create_judgment(metadata, extracted_text, chunks)
    assert new_id > 0

    # Retrieve and verify
    retrieved = JudgmentService.get_judgment(new_id)
    assert retrieved is not None
    assert retrieved["case_title"] == metadata["case_title"]
    assert len(retrieved["chunks"]) == 2

    # Clean up test row
    assert JudgmentService.delete_judgment(new_id) is True
    assert JudgmentService.get_judgment(new_id) is None


def test_saved_cases_uniqueness():
    """Verifies that duplicate saves are prevented by unique constraint."""
    # Judgment 1 exists in sample data
    db.execute("DELETE FROM saved_cases WHERE judgment_id = 1;")

    # Save once
    ok1 = JudgmentService.save_case(1)
    assert ok1 is True

    # Save again (should not crash or create duplicate)
    ok2 = JudgmentService.save_case(1)
    assert ok2 is True

    # Count bookmarks for judgment 1
    count_row = db.fetch_one("SELECT COUNT(*) as count FROM saved_cases WHERE judgment_id = 1;")
    assert count_row is not None
    assert count_row["count"] == 1

    # Cleanup
    JudgmentService.unsave_case(1)


def test_advanced_judgment_filters():
    """Verifies advanced filtering by citation, legal issue, and court (SIH1701 Requirement F)."""
    # Test citation filter
    res_citation = JudgmentService.list_judgments(citation="SCC")
    assert res_citation["total"] > 0
    for j in res_citation["judgments"]:
        assert "SCC" in (j.get("citation") or "")

    # Test court filter
    res_court = JudgmentService.list_judgments(court="Delhi")
    assert res_court["total"] > 0
    for j in res_court["judgments"]:
        assert "Delhi" in j.get("court_name", "")

    # Test filter options returns citations
    opts = JudgmentService.get_filter_options()
    assert "citations" in opts
    assert len(opts["citations"]) > 0


def test_similar_cases_identification_and_reason():
    """Verifies similar case retrieval and display of retrieval reason (SIH1701 Requirement E)."""
    res = JudgmentService.find_similar_cases(1, top_k=3)
    assert res["success"] is True
    assert "reference_judgment" in res
    assert "similar_cases" in res
    if res["similar_cases"]:
        sim = res["similar_cases"][0]
        assert "case_title" in sim
        assert "court_name" in sim
        assert "relevance_percentage" in sim
        assert "retrieval_reason" in sim
        assert len(sim["retrieval_reason"]) > 5

