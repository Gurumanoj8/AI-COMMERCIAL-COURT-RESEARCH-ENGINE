"""
Tests for Flask Application Routes and REST Endpoints.
Uses Flask test client to verify homepage, health check, search, library,
details, comparison, bookmarks, and error handling.
"""

import io
import pytest
import pymupdf
from app import app
from services.db_service import db


@pytest.fixture
def client():
    """Configures Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_loads(client):
    """Verifies that the homepage loads successfully with status code 200."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"Find the Law. Understand the Judgment." in res.data
    assert b"Court Research AI" in res.data


def test_health_endpoint(client):
    """Verifies that the /api/health endpoint returns JSON health status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "database" in data
    assert "ai_engine" in data
    assert data["database"]["status"] == "healthy"


def test_search_page_loads(client):
    """Verifies that the /search route loads with search inputs."""
    res = client.get("/search")
    assert res.status_code == 200
    assert b"Indian Commercial Court Search" in res.data


def test_api_search_empty_query(client):
    """Verifies that empty query returns a prompt to enter a search term."""
    res = client.get("/api/search?q=")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_results"] == 0


def test_library_page_loads(client):
    """Verifies that the judgments library page renders."""
    res = client.get("/library")
    assert res.status_code == 200
    assert b"Indian Judgments Repository" in res.data


def test_upload_page_loads(client):
    """Verifies that the upload page renders."""
    res = client.get("/upload")
    assert res.status_code == 200
    assert b"Ingest Court Judgment Document" in res.data


def test_assistant_page_loads(client):
    """Verifies that the AI assistant page renders."""
    res = client.get("/assistant")
    assert res.status_code == 200
    assert b"AI Legal Research Assistant" in res.data


def test_compare_page_loads(client):
    """Verifies that the compare cases page renders."""
    res = client.get("/compare")
    assert res.status_code == 200
    assert b"Precedent Comparison Engine" in res.data


def test_judgment_detail_not_found(client):
    """Verifies that requesting a non-existent judgment returns 404."""
    res = client.get("/judgment/999999")
    assert res.status_code == 404


def test_saved_case_lifecycle(client):
    """Verifies adding, listing, and removing a saved case bookmark."""
    # Test adding bookmark for judgment 1
    post_res = client.post("/api/saved/1")
    assert post_res.status_code == 200
    assert post_res.get_json()["success"] is True

    # Test listing bookmarks
    list_res = client.get("/api/saved")
    assert list_res.status_code == 200
    saved_cases = list_res.get_json()["saved_cases"]
    assert any(c["id"] == 1 for c in saved_cases)

    # Test removing bookmark
    del_res = client.delete("/api/saved/1")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True


def test_research_history_endpoint(client):
    """Verifies that recent search history is accessible via API."""
    res = client.get("/api/history")
    assert res.status_code == 200
    data = res.get_json()
    assert "history" in data
    assert isinstance(data["history"], list)


def test_semantic_search_post_endpoint(client):
    """Verifies that POST /api/search/semantic returns valid results."""
    res = client.post("/api/search/semantic", json={"query": "arbitration jurisdiction", "limit": 3})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "results" in data


def test_compare_endpoint_post(client):
    """Verifies that POST /api/compare compares two distinct judgments."""
    res = client.post("/api/compare", json={"id1": 1, "id2": 2})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "case_a" in data
    assert "case_b" in data

    # Test comparing identical judgment
    res_same = client.post("/api/compare", json={"id1": 1, "id2": 1})
    assert res_same.status_code == 400


def test_judgment_status_endpoint(client):
    """Verifies GET /api/judgments/<id>/status returns actual status."""
    res = client.get("/api/judgments/1/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "completed"


def test_saved_cases_phase11_routes(client):
    """Verifies Phase 11 endpoints /api/saved-cases."""
    # List
    list_res = client.get("/api/saved-cases")
    assert list_res.status_code == 200
    assert "saved_cases" in list_res.get_json()

    # Save
    save_res = client.post("/api/saved-cases/2")
    assert save_res.status_code == 200
    assert save_res.get_json()["saved"] is True

    # Unsave
    del_res = client.delete("/api/saved-cases/2")
    assert del_res.status_code == 200
    assert del_res.get_json()["saved"] is False


def test_history_delete_endpoint(client):
    """Verifies deleting a research history record."""
    # Insert a test history entry
    db.execute("INSERT INTO research_history (query_text, search_mode) VALUES ('temp_history_test', 'keyword');")
    row = db.fetch_one("SELECT id FROM research_history WHERE query_text = 'temp_history_test' ORDER BY id DESC LIMIT 1;")
    assert row is not None

    del_res = client.delete(f"/api/history/{row['id']}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True


def test_cors_headers_present(client):
    """Verifies CORS headers are returned on API endpoints."""
    res = client.get("/api/health")
    assert res.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in res.headers.get("Access-Control-Allow-Methods")


def test_upload_pdf_endpoint(client):
    """Verifies PDF upload via /api/judgments/upload creates record and indexes chunks."""
    # Create valid synthetic PDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Commercial Court Appeal 2026: Zeta Corp v. Omega Infra Ltd. Arbitral injunction holding.")
    pdf_bytes = doc.tobytes()
    doc.close()

    data = {
        "file": (io.BytesIO(pdf_bytes), "test_zeta_omega.pdf"),
        "case_title": "Zeta Corp v. Omega Infra Ltd.",
        "court_name": "High Court of Bombay (Commercial Division)"
    }
    res = client.post("/api/judgments/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    resp_data = res.get_json()
    assert resp_data["success"] is True
    assert resp_data["status"] == "completed"
    assert "judgment_id" in resp_data

    # Clean up created judgment
    jid = resp_data["judgment_id"]
    del_res = client.delete(f"/api/judgments/{jid}")
    assert del_res.status_code == 200


def test_assistant_ask_endpoint(client):
    """Verifies that POST /api/assistant/ask returns legal evidence and answers."""
    res = client.post("/api/assistant/ask", json={"question": "What are grounds for injunction against bank guarantee?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "evidence" in data


def test_compare_endpoint_get(client):
    """Verifies that GET /api/compare with query params returns comparison data."""
    res = client.get("/api/compare?id1=1&id2=2")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "case_a" in data
    assert "case_b" in data


def test_judgments_api_endpoints(client):
    """Verifies /api/judgments list and detail endpoints."""
    list_res = client.get("/api/judgments?per_page=5")
    assert list_res.status_code == 200
    list_data = list_res.get_json()
    assert "judgments" in list_data

    detail_res = client.get("/api/judgments/1")
    assert detail_res.status_code == 200
    detail_data = detail_res.get_json()
    assert detail_data["success"] is True
    assert "judgment" in detail_data

