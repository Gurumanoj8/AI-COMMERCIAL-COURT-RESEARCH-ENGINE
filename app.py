"""
AI Commercial Court Research Engine - Flask Web Application
Entry point containing page routes and JSON REST API endpoints.
"""

import os
from pathlib import Path
from typing import Optional
from flask import (
    Flask, render_template, request, jsonify, 
    send_from_directory, abort
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from services.db_service import db
from services.pdf_service import PDFService
from services.embedding_service import embedding_service
from services.search_service import SearchService
from services.ai_service import AIService
from services.judgment_service import JudgmentService

app = Flask(
    __name__,
    static_folder=str(Config.BASE_DIR / "static"),
    static_url_path="/static",
    template_folder=str(Config.BASE_DIR / "templates")
)
app.config.from_object(Config)

# Ensure upload directory exists
Config.ensure_upload_dir()


def initialize_vector_index_if_needed():
    """
    On startup, if the vector index is empty, safely rebuild it from the database.
    """
    try:
        if embedding_service.index is None or embedding_service.index.ntotal == 0:
            count = embedding_service.rebuild_index_from_db()
            if count > 0:
                print(f"[OK] Startup vector indexing complete! Total vectors: {count}")
    except Exception as e:
        print(f"[!] Startup vector synchronization note: {e}")


# Initialize vectors on start
initialize_vector_index_if_needed()


@app.context_processor
def inject_system_status():
    """Injects accurate database and vector search health into all templates."""
    try:
        health = db.check_health()
        health["postgresql_connected"] = bool(health.get("is_postgresql_connected", False))
    except Exception:
        health = {"active_engine": "sqlite", "postgresql_connected": False, "status": "degraded"}

    try:
        is_vector_ready = embedding_service.is_ready and (embedding_service.index is not None and embedding_service.index.ntotal > 0)
        total_vectors = embedding_service.index.ntotal if (embedding_service.is_ready and embedding_service.index is not None) else 0
    except Exception:
        is_vector_ready = False
        total_vectors = 0

    return {
        "db_health": health,
        "vector_ready": is_vector_ready,
        "vector_count": total_vectors
    }


# ====================================================================
# PAGE ROUTES (HTML Templates)
# ====================================================================

@app.route("/")
def index():
    """Homepage: Hero section, quick search, feature highlights, and recent cases."""
    recent_cases = db.fetch_all("""
        SELECT id, case_title, court_name, decision_date, citation, subject
        FROM judgments
        ORDER BY decision_date DESC NULLS LAST, id DESC
        LIMIT 4;
    """)
    for c in recent_cases:
        c["decision_date"] = str(c.get("decision_date") or "")

    count_row = db.fetch_one("SELECT COUNT(*) as total FROM judgments;")
    total_judgments = count_row["total"] if count_row else len(recent_cases)

    db_health = db.check_health()
    return render_template(
        "index.html",
        recent_cases=recent_cases,
        total_judgments=total_judgments,
        db_health=db_health
    )


@app.route("/search")
def search_page():
    """Legal Search Page with mode toggle (Semantic / Keyword) and filters."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "semantic").strip()
    court = request.args.get("court", "").strip() or None
    subject = request.args.get("subject", "").strip() or None
    year = request.args.get("year", "").strip() or None

    search_data = None
    if query:
        search_data = SearchService.search(
            query=query,
            mode=mode,
            court_filter=court,
            subject_filter=subject,
            year_filter=year
        )

    filters = JudgmentService.get_filter_options()
    ai_status = embedding_service.get_info()
    db_health = db.check_health()

    return render_template(
        "search.html",
        query=query,
        mode=mode,
        selected_court=court,
        selected_subject=subject,
        selected_year=year,
        search_data=search_data,
        filters=filters,
        ai_status=ai_status,
        db_health=db_health
    )


@app.route("/assistant")
def assistant_page():
    """AI Legal Assistant / RAG interface with evidence verification."""
    db_health = db.check_health()
    ai_info = embedding_service.get_info()
    has_llm = Config.LLM_PROVIDER != "none" and bool(Config.LLM_API_KEY)
    return render_template(
        "assistant.html",
        db_health=db_health,
        ai_info=ai_info,
        has_llm=has_llm,
        llm_provider=Config.LLM_PROVIDER
    )


@app.route("/library")
def library_page():
    """Judgments Library: paginated directory of indexed cases."""
    page = int(request.args.get("page", 1))
    search = request.args.get("search", "").strip() or None
    court = request.args.get("court", "").strip() or None
    subject = request.args.get("subject", "").strip() or None
    year = request.args.get("year", "").strip() or None
    legal_issue = request.args.get("legal_issue", "").strip() or None
    citation = request.args.get("citation", "").strip() or None

    data = JudgmentService.list_judgments(
        search=search,
        court=court,
        subject=subject,
        year=year,
        legal_issue=legal_issue,
        citation=citation,
        page=page,
        per_page=9
    )
    db_health = db.check_health()

    return render_template(
        "library.html",
        data=data,
        search=search or "",
        selected_court=court or "",
        selected_subject=subject or "",
        selected_year=year or "",
        selected_legal_issue=legal_issue or "",
        selected_citation=citation or "",
        db_health=db_health
    )


@app.route("/upload")
def upload_page():
    """PDF Upload & Ingestion page."""
    db_health = db.check_health()
    return render_template("upload.html", db_health=db_health)


@app.route("/judgment/<int:judgment_id>")
def judgment_page(judgment_id: int):
    """Judgment Details: metadata, structured legal analysis, and full text."""
    judgment = JudgmentService.get_judgment(judgment_id)
    if not judgment:
        abort(404, description="Judgment not found in database.")
    db_health = db.check_health()
    return render_template("judgment.html", judgment=judgment, db_health=db_health)


@app.route("/similar/<int:judgment_id>")
def similar_page(judgment_id: int):
    """Similar Cases: vector similarity discovery based on judgment ratio."""
    data = JudgmentService.find_similar_cases(judgment_id, top_k=6)
    if not data.get("success"):
        abort(404, description=data.get("error", "Case not found"))
    db_health = db.check_health()
    return render_template("similar_cases.html", data=data, db_health=db_health)


@app.route("/compare")
def compare_page():
    """Side-by-side Case Comparison."""
    id1 = request.args.get("id1", type=int)
    id2 = request.args.get("id2", type=int)

    all_cases = db.fetch_all("SELECT id, case_title, court_name FROM judgments ORDER BY case_title;")
    comparison_data = None

    if id1 and id2:
        comparison_data = JudgmentService.compare_judgments(id1, id2)

    db_health = db.check_health()
    return render_template(
        "compare.html",
        all_cases=all_cases,
        id1=id1,
        id2=id2,
        comparison=comparison_data,
        db_health=db_health
    )


@app.route("/saved")
def saved_cases_page():
    """Saved Cases: dedicated directory of bookmarked cases."""
    saved_cases = JudgmentService.list_saved_cases()
    for s in saved_cases:
        s["is_saved"] = True
    db_health = db.check_health()
    filters = JudgmentService.get_filter_options()
    data = {
        "judgments": saved_cases,
        "total": len(saved_cases),
        "page": 1,
        "per_page": 50,
        "total_pages": 1,
        "filters": filters
    }
    return render_template(
        "library.html",
        data=data,
        is_saved_view=True,
        search="",
        selected_court="",
        selected_subject="",
        selected_year="",
        db_health=db_health
    )


# ====================================================================
# CORS & CROSS-ORIGIN SUPPORT (Phase 15)
# ====================================================================

@app.after_request
def add_cors_headers(response):
    """Ensure CORS is handled properly for cross-origin frontend requests."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ====================================================================
# REST API ENDPOINTS (JSON)
# ====================================================================

@app.route("/api/health")
def api_health():
    """Health check endpoint: reports database and AI vector engine status without leaking credentials."""
    db_status = db.check_health(recheck=True)
    embedding_status = embedding_service.get_info()

    is_pg = db_status.get("is_postgresql_connected", False)
    overall_status = "healthy" if (is_pg or db_status.get("status") == "healthy") else "degraded"

    return jsonify({
        "status": overall_status,
        "flask": "running",
        "postgresql_connected": is_pg,
        "database": {
            "status": db_status.get("status"),
            "engine": db_status.get("active_engine"),
            "postgresql_connected": is_pg,
            "notice": db_status.get("notice")
        },
        "ai_engine": {
            "available": embedding_status.get("available", False),
            "status": embedding_status.get("status", ""),
            "model_name": embedding_status.get("model_name", ""),
            "dimension": embedding_status.get("dimension", 384),
            "engine_backend": embedding_status.get("engine_backend", "NumPy"),
            "inference_mode": embedding_status.get("inference_mode", "corpus_semantic_projection"),
            "total_indexed_chunks": embedding_status.get("total_indexed_chunks", 0)
        },
        "llm": {
            "provider": Config.LLM_PROVIDER,
            "configured": Config.LLM_PROVIDER != "none" and bool(Config.LLM_API_KEY)
        }
    })


@app.route("/api/search")
def api_search():
    """Executes keyword or semantic search (Phase 6)."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "semantic").strip()
    court = request.args.get("court", "").strip() or None
    subject = request.args.get("subject", "").strip() or None
    year = request.args.get("year", "").strip() or None
    limit = int(request.args.get("limit", 10))

    if not query:
        return jsonify({
            "success": True,
            "message": "Query parameter 'q' is required to search.",
            "query": "",
            "search_mode": mode,
            "total_results": 0,
            "results": []
        })

    result = SearchService.search(
        query=query,
        mode=mode,
        court_filter=court,
        subject_filter=subject,
        year_filter=year,
        limit=limit
    )
    return jsonify(result)


@app.route("/api/search/semantic", methods=["POST"])
def api_search_semantic():
    """Executes dense vector semantic search using Sentence Transformers and FAISS (Phase 7)."""
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "Search query cannot be empty."}), 400

    court = data.get("court")
    subject = data.get("subject")
    year = data.get("year")
    limit = int(data.get("limit", 10))

    result = SearchService.search(
        query=query,
        mode="semantic",
        court_filter=court,
        subject_filter=subject,
        year_filter=year,
        limit=limit
    )
    return jsonify(result)


@app.route("/api/assistant/ask", methods=["POST"])
def api_assistant_ask():
    """Answers a user legal question using RAG and evidence verification (Phase 8)."""
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty."}), 400

    judgment_id = data.get("judgment_id")
    if judgment_id is not None:
        try:
            judgment_id = int(judgment_id)
        except (ValueError, TypeError):
            judgment_id = None

    result = AIService.answer_query(question, judgment_id=judgment_id)
    return jsonify(result)


@app.route("/api/judgments/upload", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def api_upload_pdf():
    """
    Ingests an uploaded legal PDF judgment (Phase 4):
    1. Validates file header and size
    2. Saves safely to disk
    3. Extracts text using PyMuPDF
    4. Chunks text with page preservation
    5. Stores metadata & chunks in database
    6. Adds vectors to FAISS index
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file attached in upload request."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = file.filename

    # 1. Validate PDF format and magic bytes
    is_valid, val_msg = PDFService.is_valid_pdf(file.stream, filename)
    if not is_valid:
        return jsonify({"success": False, "error": val_msg}), 400

    # 2. Save safely
    try:
        saved_path = PDFService.save_file_safely(file, filename)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save file: {str(e)}"}), 500

    # 3. Extract text
    extraction = PDFService.extract_text_and_pages(saved_path)
    if not extraction.get("success"):
        # Remove file on extraction failure to prevent orphan files
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": extraction.get("error", "Text extraction failed."),
            "is_scanned": extraction.get("is_scanned", False)
        }), 422

    # 4. Formulate authentic metadata and automatic legal summarization
    summary = extraction.get("structured_summary", {})
    hints = extraction.get("metadata_hints", {})

    metadata = {
        "case_title": request.form.get("case_title", "").strip() or hints.get("case_title") or Path(filename).stem,
        "court_name": request.form.get("court_name", "").strip() or hints.get("court_name") or "Indian Commercial Court",
        "decision_date": request.form.get("decision_date", "").strip() or hints.get("decision_date") or None,
        "citation": request.form.get("citation", "").strip() or hints.get("citation") or None,
        "case_number": request.form.get("case_number", "").strip() or hints.get("case_number") or None,
        "subject": request.form.get("subject", "").strip() or hints.get("subject") or "Commercial Law",
        "source_url": request.form.get("source_url", "").strip() or hints.get("source_url") or None,
        "material_facts": request.form.get("material_facts", "").strip() or summary.get("material_facts") or None,
        "legal_issues": request.form.get("legal_issues", "").strip() or summary.get("legal_issues") or None,
        "arguments": request.form.get("arguments", "").strip() or summary.get("arguments") or None,
        "judicial_reasoning": request.form.get("judicial_reasoning", "").strip() or summary.get("judicial_reasoning") or None,
        "ratio_decidendi": request.form.get("ratio_decidendi", "").strip() or summary.get("ratio_decidendi") or None,
        "final_decision": request.form.get("final_decision", "").strip() or summary.get("final_decision") or None,
        "relevant_statutes": request.form.get("relevant_statutes", "").strip() or summary.get("relevant_statutes") or None
    }

    # 5. Chunk text
    chunks = PDFService.chunk_text(
        extraction["pages"],
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    # 6. Save in database and index in FAISS
    try:
        judgment_id = JudgmentService.create_judgment(
            metadata=metadata,
            extracted_text=extraction["full_text"],
            chunks=chunks,
            document_path=saved_path.name
        )
    except Exception as e:
        # Clean up file on database failure
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"success": False, "error": f"Database insertion failed: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "judgment_id": judgment_id,
        "status": "completed",
        "message": "Judgment uploaded and processed successfully.",
        "case_title": metadata["case_title"],
        "pages_processed": extraction["total_pages"],
        "chunks_indexed": len(chunks)
    }), 201


@app.route("/api/judgments")
def api_judgments_list():
    """Lists judgments with pagination and search filters (Phase 5)."""
    try:
        page = int(request.args.get("page", 1))
    except (ValueError, TypeError):
        page = 1
        
    try:
        per_page = int(request.args.get("per_page", 10))
    except (ValueError, TypeError):
        per_page = 10

    search = request.args.get("search", "").strip() or None
    court = request.args.get("court", "").strip() or None
    subject = request.args.get("subject", "").strip() or None
    year = request.args.get("year", "").strip() or None
    legal_issue = request.args.get("legal_issue", "").strip() or None
    citation = request.args.get("citation", "").strip() or None
    date_from = request.args.get("date_from", "").strip() or None
    date_to = request.args.get("date_to", "").strip() or None

    data = JudgmentService.list_judgments(
        search=search,
        court=court,
        subject=subject,
        year=year,
        legal_issue=legal_issue,
        citation=citation,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=per_page
    )
    return jsonify(data)


@app.route("/api/judgments/<int:judgment_id>")
def api_judgment_detail(judgment_id: int):
    """Returns single judgment details and its chunks (Phase 5)."""
    j = JudgmentService.get_judgment(judgment_id)
    if not j:
        return jsonify({"success": False, "error": "Judgment not found"}), 404
    return jsonify({"success": True, "judgment": j})


@app.route("/api/judgments/<int:judgment_id>/status")
def api_judgment_status(judgment_id: int):
    """Returns actual processing status for a judgment (Phase 5)."""
    status_info = JudgmentService.get_judgment_status(judgment_id)
    if not status_info:
        return jsonify({"success": False, "error": "Judgment not found"}), 404
    return jsonify({"success": True, **status_info})


@app.route("/api/judgments/<int:judgment_id>", methods=["DELETE"])
def api_judgment_delete(judgment_id: int):
    """Safely deletes a judgment and associated records/files (Phase 5)."""
    deleted = JudgmentService.delete_judgment(judgment_id)
    if not deleted:
        return jsonify({"success": False, "error": "Judgment not found or deletion failed."}), 404
    return jsonify({"success": True, "message": f"Judgment {judgment_id} deleted successfully."})


@app.route("/api/judgments/<int:judgment_id>/similar")
def api_judgment_similar(judgment_id: int):
    """Retrieves similar cases for a given judgment (Phase 9)."""
    data = JudgmentService.find_similar_cases(judgment_id)
    if not data.get("success"):
        return jsonify(data), 404
    return jsonify(data)


@app.route("/api/compare", methods=["GET", "POST"])
def api_compare_cases():
    """Side-by-side Case Comparison REST endpoint (Phase 10)."""
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        id1 = data.get("id1") or data.get("case_a_id")
        id2 = data.get("id2") or data.get("case_b_id")
    else:
        id1 = request.args.get("id1") or request.args.get("case_a_id")
        id2 = request.args.get("id2") or request.args.get("case_b_id")

    if not id1 or not id2:
        return jsonify({"success": False, "error": "Both 'id1' and 'id2' parameters are required for comparison."}), 400

    try:
        id1, id2 = int(id1), int(id2)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Judgment IDs must be integers."}), 400

    result = JudgmentService.compare_judgments(id1, id2)
    status_code = 200 if result.get("success") else (400 if "itself" in result.get("error", "") else 404)
    return jsonify(result), status_code


@app.route("/api/saved-cases", methods=["GET"])
@app.route("/api/saved", methods=["GET"])
def api_list_saved():
    """Lists saved cases with multi-user local storage notice (Phase 11)."""
    saved = JudgmentService.list_saved_cases()
    return jsonify({
        "success": True,
        "saved_cases": saved,
        "count": len(saved),
        "notice": "Saved cases are stored in the local shared database installation."
    })


@app.route("/api/saved-cases", methods=["POST"])
@app.route("/api/saved", methods=["POST"])
@app.route("/api/saved-cases/<int:judgment_id>", methods=["POST"])
@app.route("/api/saved/<int:judgment_id>", methods=["POST"])
def api_save_case(judgment_id: Optional[int] = None):
    """Saves a judgment bookmark (Phase 11)."""
    if judgment_id is None:
        data = request.get_json(force=True, silent=True) or {}
        raw_id = data.get("judgment_id")
        if raw_id is not None:
            try:
                judgment_id = int(raw_id)
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "judgment_id must be an integer."}), 400

    if judgment_id is None:
        return jsonify({"success": False, "error": "Missing judgment_id parameter."}), 400

    # Verify judgment exists
    if not JudgmentService.get_judgment(judgment_id):
        return jsonify({"success": False, "error": "Judgment not found."}), 404

    ok = JudgmentService.save_case(judgment_id)
    return jsonify({"success": ok, "saved": ok, "judgment_id": judgment_id})


@app.route("/api/saved-cases", methods=["DELETE"])
@app.route("/api/saved", methods=["DELETE"])
@app.route("/api/saved-cases/<int:judgment_id>", methods=["DELETE"])
@app.route("/api/saved/<int:judgment_id>", methods=["DELETE"])
def api_unsave_case(judgment_id: Optional[int] = None):
    """Removes a saved judgment bookmark (Phase 11)."""
    if judgment_id is None:
        data = request.get_json(force=True, silent=True) or {}
        raw_id = data.get("judgment_id")
        if raw_id is not None:
            try:
                judgment_id = int(raw_id)
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "judgment_id must be an integer."}), 400

    if judgment_id is None:
        return jsonify({"success": False, "error": "Missing judgment_id parameter."}), 400

    ok = JudgmentService.unsave_case(judgment_id)
    return jsonify({"success": ok, "saved": False, "judgment_id": judgment_id})


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent search history (Phase 12)."""
    history = JudgmentService.get_recent_history()
    return jsonify({"success": True, "history": history})


@app.route("/api/history/<int:history_id>", methods=["DELETE"])
def api_delete_history(history_id: int):
    """Deletes a single research history item (Phase 12)."""
    ok = JudgmentService.delete_history_item(history_id)
    if not ok:
        return jsonify({"success": False, "error": "History entry not found."}), 404
    return jsonify({"success": True, "message": "History entry deleted."})


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    """Clears all research history entries (Phase 12)."""
    JudgmentService.clear_history()
    return jsonify({"success": True, "message": "Research history cleared."})


@app.route("/uploads/<path:filename>")
def download_uploaded_file(filename: str):
    """
    Secure file access for original uploaded PDF files.
    Restricts access strictly to .pdf files within the upload folder.
    """
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".pdf"):
        abort(403, description="Access restricted to PDF documents.")
    return send_from_directory(Config.UPLOAD_FOLDER, safe_name, as_attachment=False)


@app.route("/favicon.ico")
@app.route("/favicon.svg")
@app.route("/favicon.png")
@app.route("/favicon-32x32.png")
@app.route("/apple-touch-icon.png")
def favicon():
    """Serves favicon and device icons directly from the root URL."""
    path = request.path.lstrip("/")
    if path.endswith(".svg"):
        return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")
    elif path.endswith(".png"):
        filename = path if path in ("favicon-32x32.png", "apple-touch-icon.png") else "favicon.png"
        return send_from_directory(app.static_folder, filename, mimetype="image/png")
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/x-icon")


# ====================================================================
# ERROR HANDLERS
# ====================================================================

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint or resource not found."}), 404
    return render_template("index.html", error_message="Page not found (404)."), 404


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return jsonify({
        "success": False,
        "error": f"File is too large. Maximum allowed upload size is {Config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    }), 413


@app.errorhandler(500)

def internal_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500
    return render_template("index.html", error_message="Internal server error (500)."), 500


if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f" AI Commercial Court Research Engine Starting...")
    print(f" Database Mode: {db.engine_type.upper()}")
    print(f" Vector Search: {'ONLINE' if embedding_service.is_ready else 'OFFLINE'}")
    print(f" Access URL:    http://127.0.0.1:{Config.PORT}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)