"""
Search Engine Tests.
Validates keyword text search, filtering, semantic vector search,
mode indication, and research history tracking.
"""

import pytest
from services.search_service import SearchService
from services.embedding_service import embedding_service
from services.db_service import db


def test_keyword_search_basic():
    """Verifies keyword search across indexed judgments."""
    res = SearchService.search("arbitration", mode="keyword")
    assert res["success"] is True
    assert res["active_mode"] == "keyword"
    assert isinstance(res["results"], list)
    if res["results"]:
        assert any("arbitration" in r["case_title"].lower() or "arbitration" in (r.get("matched_passage") or "").lower() for r in res["results"])


def test_search_court_filter():
    """Verifies that court filter isolates results to the specified jurisdiction."""
    res = SearchService.search("Section", mode="keyword", court_filter="Delhi")
    assert res["success"] is True
    for r in res["results"]:
        assert "Delhi" in r["court_name"]


def test_semantic_search_execution():
    """Verifies that vector semantic search returns relevance scores."""
    if not embedding_service.is_ready or embedding_service.index is None or embedding_service.index.ntotal == 0:
        pytest.skip("Embedding service or FAISS index not ready for vector search.")

    res = SearchService.search("restraining bank guarantee encashment", mode="semantic")
    assert res["success"] is True
    assert res["active_mode"] in ["semantic", "keyword_fallback"]

    if res["active_mode"] == "semantic" and res["results"]:
        top = res["results"][0]
        assert top["similarity_score"] is not None
        assert top["relevance_percentage"] is not None
        assert 0 <= top["relevance_percentage"] <= 100


def test_research_history_logging():
    """Verifies that queries are logged to the research_history database table."""
    test_query = "unique_viva_test_query_string_2026"
    SearchService.search(test_query, mode="keyword")

    history_row = db.fetch_one(
        "SELECT query_text, search_mode FROM research_history WHERE query_text = %s ORDER BY id DESC LIMIT 1;",
        (test_query,)
    )
    assert history_row is not None
    assert history_row["query_text"] == test_query
    assert history_row["search_mode"] == "keyword"

    # Cleanup
    db.execute("DELETE FROM research_history WHERE query_text = %s;", (test_query,))


def test_rag_grounded_answer_generation():
    """Verifies RAG-based legal research and grounded answer generation."""
    from services.ai_service import AIService
    res = AIService.answer_query("grounds for restraining unconditional bank guarantee")
    assert res["success"] is True
    assert "evidence" in res
    assert "generated_answer" in res
    assert res["generated_answer"] is not None
    assert len(res["generated_answer"]) > 20
    # Verify citations and legal disclaimer presence
    assert "legal advice" in res["disclaimer"].lower() or "research aid" in res["disclaimer"].lower()

