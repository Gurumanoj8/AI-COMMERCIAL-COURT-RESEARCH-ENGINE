"""
Judgment Service for Court Research AI.
Coordinates database CRUD operations, structured legal analysis extraction,
case similarity discovery, side-by-side comparison, and bookmark management.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from config import Config
from services.db_service import db
from services.embedding_service import embedding_service


class JudgmentService:
    """Business logic for Indian commercial court judgments."""

    @classmethod
    def list_judgments(
        cls,
        search: Optional[str] = None,
        court: Optional[str] = None,
        subject: Optional[str] = None,
        year: Optional[str] = None,
        legal_issue: Optional[str] = None,
        citation: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        per_page: int = 10
    ) -> Dict[str, Any]:
        """Returns paginated judgment records with active filter counters."""
        is_pg = db.engine_type == "postgresql"
        like_op = "ILIKE" if is_pg else "LIKE"

        where_clauses = ["1=1"]
        params = []

        if search:
            where_clauses.append(f"""(
                case_title {like_op} %s OR 
                citation {like_op} %s OR 
                case_number {like_op} %s OR 
                subject {like_op} %s OR
                legal_issues {like_op} %s
            )""")
            term = f"%{search.strip()}%"
            params.extend([term, term, term, term, term])

        if court:
            where_clauses.append(f"court_name {like_op} %s")
            params.append(f"%{court.strip()}%")

        if subject:
            where_clauses.append(f"subject {like_op} %s")
            params.append(f"%{subject.strip()}%")

        if year:
            if is_pg:
                where_clauses.append("EXTRACT(YEAR FROM decision_date)::TEXT = %s")
            else:
                where_clauses.append("strftime('%Y', decision_date) = %s")
            params.append(str(year))

        if legal_issue:
            where_clauses.append(f"legal_issues {like_op} %s")
            params.append(f"%{legal_issue.strip()}%")

        if citation:
            where_clauses.append(f"citation {like_op} %s")
            params.append(f"%{citation.strip()}%")

        if date_from:
            where_clauses.append("decision_date >= %s")
            params.append(str(date_from).strip())

        if date_to:
            where_clauses.append("decision_date <= %s")
            params.append(str(date_to).strip())

        where_sql = " AND ".join(where_clauses)

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) as total FROM judgments WHERE {where_sql};"
        count_row = db.fetch_one(count_sql, tuple(params))
        total = count_row["total"] if count_row else 0

        # Calculate pagination
        offset = max(0, (page - 1) * per_page)
        data_sql = f"""
            SELECT id, case_title, court_name, decision_date, citation, 
                   case_number, subject, source_url, document_path, processing_status,
                   created_at
            FROM judgments
            WHERE {where_sql}
            ORDER BY decision_date DESC NULLS LAST, id DESC
            LIMIT {per_page} OFFSET {offset};
        """
        rows = db.fetch_all(data_sql, tuple(params))

        # Check saved cases status
        saved_ids = cls.get_saved_ids()
        for r in rows:
            r["is_saved"] = r["id"] in saved_ids
            r["decision_date"] = str(r["decision_date"] or "")

        total_pages = max(1, (total + per_page - 1) // per_page)

        return {
            "judgments": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "filters": cls.get_filter_options()
        }

    @classmethod
    def get_judgment(cls, judgment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves full details of a judgment including its segmented chunks."""
        sql = "SELECT * FROM judgments WHERE id = %s;"
        row = db.fetch_one(sql, (judgment_id,))
        if not row:
            return None

        # Fetch chunks
        chunks_sql = """
            SELECT id, chunk_index, chunk_text, page_number
            FROM judgment_chunks
            WHERE judgment_id = %s
            ORDER BY chunk_index ASC;
        """
        chunks = db.fetch_all(chunks_sql, (judgment_id,))

        saved_ids = cls.get_saved_ids()
        row["is_saved"] = judgment_id in saved_ids
        row["chunks"] = chunks
        row["decision_date"] = str(row.get("decision_date") or "")
        return row

    @classmethod
    def create_judgment(
        cls,
        metadata: Dict[str, Any],
        extracted_text: str,
        chunks: List[Dict[str, Any]],
        document_path: Optional[str] = None
    ) -> int:
        """
        Stores a judgment and its text chunks in the database,
        and synchronizes chunks into the FAISS vector index.
        """
        insert_sql = """
            INSERT INTO judgments (
                case_title, court_name, decision_date, citation, case_number,
                subject, source_url, document_path, extracted_text,
                material_facts, legal_issues, arguments, judicial_reasoning,
                ratio_decidendi, final_decision, relevant_statutes,
                processing_status
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s
            ) RETURNING id;
        """
        params = (
            metadata.get("case_title") or "Untitled Judgment",
            metadata.get("court_name") or None,
            metadata.get("decision_date") or None,
            metadata.get("citation") or None,
            metadata.get("case_number") or None,
            metadata.get("subject") or "Commercial Law",
            metadata.get("source_url") or None,
            document_path,
            extracted_text,
            metadata.get("material_facts") or None,
            metadata.get("legal_issues") or None,
            metadata.get("arguments") or None,
            metadata.get("judicial_reasoning") or None,
            metadata.get("ratio_decidendi") or None,
            metadata.get("final_decision") or None,
            metadata.get("relevant_statutes") or None,
            "completed"
        )

        judgment_id = db.insert_and_return_id(insert_sql, params)

        # Insert chunks into database with vector persistence
        if chunks:
            # Generate vector embeddings for the chunks if encoder is available
            computed_embeddings = None
            try:
                texts = [c["chunk_text"] for c in chunks]
                computed_embeddings = embedding_service.encode(texts)
            except Exception as e:
                print(f"[!] Warning generating embeddings for chunks: {e}")

            chunk_ops = []
            prepared_chunks = []
            for i, c in enumerate(chunks):
                emb_json = None
                if computed_embeddings is not None and i < len(computed_embeddings):
                    emb_json = json.dumps(computed_embeddings[i].tolist())

                p_num = c.get("page_number")
                if p_num is not None:
                    try:
                        p_num = int(p_num)
                    except (ValueError, TypeError):
                        p_num = None

                chunk_ops.append((
                    "INSERT INTO judgment_chunks (judgment_id, chunk_index, chunk_text, page_number, embedding) VALUES (%s, %s, %s, %s, %s);",
                    (judgment_id, c["chunk_index"], c["chunk_text"], p_num, emb_json)
                ))
                prepared_chunks.append({
                    "judgment_id": judgment_id,
                    "chunk_index": c["chunk_index"],
                    "chunk_text": c["chunk_text"],
                    "page_number": p_num
                })

            db.execute_transaction(chunk_ops)

            # Synchronize into in-memory vector index
            try:
                embedding_service.add_chunks(prepared_chunks, precomputed_embeddings=computed_embeddings)
            except Exception as e:
                print(f"[!] Warning updating vector index: {e}")

        return judgment_id

    @classmethod
    def find_similar_cases(cls, judgment_id: int, top_k: int = 4) -> Dict[str, Any]:
        """
        Retrieves semantically similar cases using vector embeddings of the reference judgment.
        Excludes the reference judgment itself from results.
        """
        reference = cls.get_judgment(judgment_id)
        if not reference:
            return {"success": False, "error": "Reference judgment not found."}

        # Build query text from the reference case (ratio decidendi or first chunk or title)
        query_text = reference.get("ratio_decidendi") or ""
        if len(query_text) < 50:
            query_text = reference.get("material_facts") or reference.get("extracted_text", "")[:400]

        similar_chunks = embedding_service.search_similar(query_text, top_k=15)

        # Filter out chunks that belong to the reference judgment itself
        other_chunks = [c for c in similar_chunks if c.get("judgment_id") != judgment_id]

        similar_cases = []
        seen_jids = set()

        for c in other_chunks:
            jid = c.get("judgment_id")
            if not jid or jid in seen_jids:
                continue

            j_data = cls.get_judgment(jid)
            if j_data:
                seen_jids.add(jid)
                relevance_pct = c.get("relevance_percentage") or 0
                subject_match = j_data.get("subject") == reference.get("subject")
                court_match = j_data.get("court_name") == reference.get("court_name")

                reasons = []
                if relevance_pct >= 65:
                    reasons.append(f"High semantic alignment ({relevance_pct}% vector match) on judicial principles and ratio decidendi")
                elif relevance_pct >= 45:
                    reasons.append(f"Substantive vector similarity ({relevance_pct}% match) on framed legal propositions")
                else:
                    reasons.append(f"Contextual similarity ({relevance_pct}% match) on commercial legal issues")

                if subject_match and j_data.get("subject"):
                    reasons.append(f"shared subject category ('{j_data.get('subject')}')")
                if court_match:
                    reasons.append(f"same jurisdiction ({j_data.get('court_name')})")

                why_retrieved = f"Retrieved due to {reasons[0]}" + (f", with {', '.join(reasons[1:])}." if len(reasons) > 1 else ".")

                similar_cases.append({
                    "id": jid,
                    "case_title": j_data["case_title"],
                    "court_name": j_data["court_name"],
                    "decision_date": j_data["decision_date"],
                    "citation": j_data["citation"],
                    "subject": j_data["subject"],
                    "similarity_score": c.get("similarity_score"),
                    "relevance_percentage": relevance_pct,
                    "relevant_passage": c.get("chunk_text"),
                    "page_number": c.get("page_number"),
                    "retrieval_reason": why_retrieved
                })
            if len(similar_cases) >= top_k:
                break

        return {
            "success": True,
            "reference_judgment": {
                "id": reference["id"],
                "case_title": reference["case_title"],
                "court_name": reference["court_name"],
                "decision_date": reference["decision_date"],
                "citation": reference["citation"],
                "subject": reference["subject"],
                "ratio_decidendi": reference.get("ratio_decidendi")
            },
            "similar_cases": similar_cases,
            "legal_note": (
                "Semantic similarity indicates linguistic and factual relevance. "
                "It does not automatically establish binding legal precedent under Article 141 or judicial hierarchy."
            )
        }

    @classmethod
    def compare_judgments(cls, id1: int, id2: int) -> Dict[str, Any]:
        """Provides side-by-side comparative analysis of two selected judgments."""
        if id1 == id2:
            return {"success": False, "error": "Cannot compare a judgment with itself. Please select two distinct cases."}

        j1 = cls.get_judgment(id1)
        j2 = cls.get_judgment(id2)

        if not j1 or not j2:
            return {"success": False, "error": "One or both judgments could not be found."}

        return {
            "success": True,
            "case_a": j1,
            "case_b": j2
        }

    @classmethod
    def save_case(cls, judgment_id: int) -> bool:
        """Saves a judgment to the user's bookmarks list."""
        is_pg = db.engine_type == "postgresql"
        sql = (
            "INSERT INTO saved_cases (judgment_id) VALUES (%s) ON CONFLICT (judgment_id) DO NOTHING;"
            if is_pg else
            "INSERT OR IGNORE INTO saved_cases (judgment_id) VALUES (%s);"
        )
        try:
            db.execute(sql, (judgment_id,))
            return True
        except Exception as e:
            print(f"[!] Error saving case: {e}")
            return False

    @classmethod
    def unsave_case(cls, judgment_id: int) -> bool:
        """Removes a judgment from the user's bookmarks list."""
        try:
            db.execute("DELETE FROM saved_cases WHERE judgment_id = %s;", (judgment_id,))
            return True
        except Exception:
            return False

    @classmethod
    def list_saved_cases(cls) -> List[Dict[str, Any]]:
        """Lists all judgments currently bookmarked by the user."""
        sql = """
            SELECT j.id, j.case_title, j.court_name, j.decision_date, 
                   j.citation, j.subject, s.created_at AS saved_at
            FROM saved_cases s
            JOIN judgments j ON s.judgment_id = j.id
            ORDER BY s.created_at DESC;
        """
        rows = db.fetch_all(sql)
        for r in rows:
            r["decision_date"] = str(r["decision_date"] or "")
            r["saved_at"] = str(r["saved_at"] or "")
        return rows

    @classmethod
    def get_saved_ids(cls) -> set:
        """Returns set of all saved judgment IDs."""
        try:
            rows = db.fetch_all("SELECT judgment_id FROM saved_cases;")
            return {r["judgment_id"] for r in rows}
        except Exception:
            return set()

    @classmethod
    def get_filter_options(cls) -> Dict[str, List[str]]:
        """Returns distinct filter values currently present in the database."""
        courts = [r["court_name"] for r in db.fetch_all("SELECT DISTINCT court_name FROM judgments WHERE court_name IS NOT NULL ORDER BY court_name;")]
        subjects = [r["subject"] for r in db.fetch_all("SELECT DISTINCT subject FROM judgments WHERE subject IS NOT NULL ORDER BY subject;")]

        if db.engine_type == "postgresql":
            year_sql = "SELECT DISTINCT EXTRACT(YEAR FROM decision_date)::TEXT AS yr FROM judgments WHERE decision_date IS NOT NULL ORDER BY yr DESC;"
        else:
            year_sql = "SELECT DISTINCT strftime('%Y', decision_date) AS yr FROM judgments WHERE decision_date IS NOT NULL ORDER BY yr DESC;"

        years = [r["yr"] for r in db.fetch_all(year_sql) if r.get("yr")]
        citations = [r["citation"] for r in db.fetch_all("SELECT DISTINCT citation FROM judgments WHERE citation IS NOT NULL AND citation != '' ORDER BY citation;")]

        return {
            "courts": courts,
            "subjects": subjects,
            "years": years,
            "citations": citations
        }

    @classmethod
    def get_recent_history(cls, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns recent research query history."""
        sql = f"SELECT id, query_text, search_mode, created_at FROM research_history ORDER BY id DESC LIMIT {limit};"
        rows = db.fetch_all(sql)
        for r in rows:
            r["created_at"] = str(r.get("created_at") or "")
        return rows

    @classmethod
    def get_judgment_status(cls, judgment_id: int) -> Optional[Dict[str, Any]]:
        """Returns the processing status of a judgment."""
        row = db.fetch_one("SELECT id, processing_status, created_at, updated_at FROM judgments WHERE id = %s;", (judgment_id,))
        if not row:
            return None
        return {
            "judgment_id": row["id"],
            "status": row.get("processing_status", "completed"),
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or "")
        }

    @classmethod
    def delete_judgment(cls, judgment_id: int) -> bool:
        """Safely deletes a judgment, its associated file on disk, and cascade-removes chunks and bookmarks."""
        judgment = cls.get_judgment(judgment_id)
        if not judgment:
            return False

        doc_path = judgment.get("document_path")
        if doc_path:
            try:
                safe_name = os.path.basename(doc_path)
                target_file = Config.UPLOAD_FOLDER / safe_name
                if target_file.exists():
                    target_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"[!] Warning cleaning up judgment file: {e}")

        # Delete from database (foreign keys and child records cascade-cleaned)
        db.execute("DELETE FROM judgment_chunks WHERE judgment_id = %s;", (judgment_id,))
        db.execute("DELETE FROM saved_cases WHERE judgment_id = %s;", (judgment_id,))
        db.execute("DELETE FROM judgments WHERE id = %s;", (judgment_id,))

        # Synchronize vector index with updated database
        try:
            embedding_service.rebuild_index_from_db()
        except Exception:
            pass

        return True

    @classmethod
    def delete_history_item(cls, history_id: int) -> bool:
        """Deletes a research history entry by ID."""
        rowcount = db.execute("DELETE FROM research_history WHERE id = %s;", (history_id,))
        return rowcount > 0

    @classmethod
    def clear_history(cls) -> bool:
        """Clears all research history records."""
        db.execute("DELETE FROM research_history;")
        return True
