"""
Search Service for Court Research AI.
Implements dual search modalities:
1. Keyword Search (SQL case-insensitive search across title, court, subject, and text)
2. Semantic Search (FAISS vector similarity retrieval mapped to database judgments)
"""

import re
from typing import List, Dict, Any, Optional
from services.db_service import db
from services.embedding_service import embedding_service


class SearchService:
    """Coordinates keyword text search and semantic vector search."""

    @classmethod
    def search(
        cls,
        query: str,
        mode: str = "semantic",
        court_filter: Optional[str] = None,
        subject_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Executes search according to selected mode with optional filters.
        Logs the query in research_history and returns structured results.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "success": True,
                "query": "",
                "search_mode": mode,
                "active_mode": mode,
                "total_results": 0,
                "results": [],
                "message": "Please enter a search query."
            }

        # Check if requested semantic search is actually available
        active_mode = mode
        mode_note = None

        if mode == "semantic":
            if not embedding_service.index or embedding_service.index.ntotal == 0:
                # Fallback to keyword search with clear notification
                active_mode = "keyword_fallback"
                mode_note = (
                    "Semantic vector index contains 0 records. "
                    "Displaying keyword text search results instead."
                )
                results = cls._keyword_search(clean_query, court_filter, subject_filter, year_filter, limit)
            else:
                try:
                    results = cls._semantic_search(clean_query, court_filter, subject_filter, year_filter, limit)
                    active_mode = "semantic"
                    mode_note = None
                except Exception as e:
                    active_mode = "keyword_fallback"
                    mode_note = f"Semantic search encountered an issue ({str(e)}). Switched to keyword fallback."
                    results = cls._keyword_search(clean_query, court_filter, subject_filter, year_filter, limit)
        else:
            active_mode = "keyword"
            results = cls._keyword_search(clean_query, court_filter, subject_filter, year_filter, limit)

        # Log query in research_history table
        try:
            db.execute(
                "INSERT INTO research_history (query_text, search_mode) VALUES (%s, %s);",
                (clean_query, active_mode)
            )
        except Exception:
            pass  # Non-critical if history logging encounters an error

        return {
            "success": True,
            "query": clean_query,
            "requested_mode": mode,
            "active_mode": active_mode,
            "note": mode_note,
            "total_results": len(results),
            "results": results
        }

    @classmethod
    def _semantic_search(
        cls,
        query: str,
        court_filter: Optional[str],
        subject_filter: Optional[str],
        year_filter: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Runs FAISS vector search, retrieves matching chunks, and joins them
        with judgment metadata from the database.
        """
        # Retrieve top 20 candidate chunks to allow for deduplication & filtering
        similar_chunks = embedding_service.search_similar(query, top_k=25)
        if not similar_chunks:
            return []

        # Group matched chunks by judgment_id, keeping the highest scoring chunk
        judgment_scores = {}
        for c in similar_chunks:
            jid = c.get("judgment_id")
            if not jid:
                continue
            if jid not in judgment_scores:
                judgment_scores[jid] = {
                    "score": c.get("similarity_score", 0.0),
                    "relevance_pct": c.get("relevance_percentage", 0.0),
                    "best_passage": c.get("chunk_text", ""),
                    "page_number": c.get("page_number")
                }

        if not judgment_scores:
            return []

        # Retrieve judgment records for all matched IDs
        jids = list(judgment_scores.keys())
        # Safe parameterized IN clause
        placeholders = ", ".join(["%s"] * len(jids))
        sql = f"""
            SELECT id, case_title, court_name, decision_date, citation, 
                   case_number, subject, source_url
            FROM judgments
            WHERE id IN ({placeholders})
        """
        params = list(jids)

        if court_filter:
            sql += " AND court_name ILIKE %s" if db.engine_type == "postgresql" else " AND court_name LIKE %s"
            params.append(f"%{court_filter}%")
        if subject_filter:
            sql += " AND subject ILIKE %s" if db.engine_type == "postgresql" else " AND subject LIKE %s"
            params.append(f"%{subject_filter}%")
        if year_filter:
            sql += " AND strftime('%Y', decision_date) = %s" if db.engine_type != "postgresql" else " AND EXTRACT(YEAR FROM decision_date)::TEXT = %s"
            params.append(str(year_filter))

        rows = db.fetch_all(sql, tuple(params))

        # Check which cases are saved
        saved_ids = cls._get_saved_case_ids()

        # Combine judgment metadata with similarity scores
        results = []
        for r in rows:
            jid = r["id"]
            score_data = judgment_scores.get(jid, {})
            results.append({
                "id": jid,
                "judgment_id": jid,
                "case_title": r.get("case_title"),
                "court_name": r.get("court_name"),
                "decision_date": str(r.get("decision_date") or ""),
                "citation": r.get("citation"),
                "case_number": r.get("case_number"),
                "subject": r.get("subject"),
                "source_url": r.get("source_url"),
                "similarity_score": score_data.get("score"),
                "relevance_percentage": score_data.get("relevance_pct"),
                "matched_passage": score_data.get("best_passage"),
                "page_number": score_data.get("page_number"),
                "search_mode": "semantic",
                "is_saved": jid in saved_ids
            })

        # Sort descending by relevance score
        results.sort(key=lambda x: x["similarity_score"] or 0, reverse=True)
        return results[:limit]

    @classmethod
    def _keyword_search(
        cls,
        query: str,
        court_filter: Optional[str],
        subject_filter: Optional[str],
        year_filter: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Performs parameterised SQL ILIKE search across judgment fields and extracted text.
        """
        is_pg = db.engine_type == "postgresql"
        like_op = "ILIKE" if is_pg else "LIKE"

        sql = f"""
            SELECT id, case_title, court_name, decision_date, citation, 
                   case_number, subject, source_url, extracted_text
            FROM judgments
            WHERE (
                case_title {like_op} %s OR
                court_name {like_op} %s OR
                subject {like_op} %s OR
                citation {like_op} %s OR
                extracted_text {like_op} %s
            )
        """
        param_term = f"%{query}%"
        params = [param_term, param_term, param_term, param_term, param_term]

        if court_filter:
            sql += f" AND court_name {like_op} %s"
            params.append(f"%{court_filter}%")
        if subject_filter:
            sql += f" AND subject {like_op} %s"
            params.append(f"%{subject_filter}%")
        if year_filter:
            if is_pg:
                sql += " AND EXTRACT(YEAR FROM decision_date)::TEXT = %s"
            else:
                sql += " AND strftime('%Y', decision_date) = %s"
            params.append(str(year_filter))

        sql += f" ORDER BY decision_date DESC LIMIT {limit};"

        rows = db.fetch_all(sql, tuple(params))

        # If strict phrase search yielded 0 results, fall back to multi-term matching
        if not rows and " " in query:
            clean_words = [w for w in re.findall(r"\w+", query) if len(w) >= 3 and w.lower() not in {"what", "which", "where", "when", "does", "have", "with", "from", "that", "this", "under"}]
            if clean_words:
                term_clauses = []
                term_params = []
                for word in clean_words[:4]:
                    w_term = f"%{word}%"
                    term_clauses.append(f"(case_title {like_op} %s OR subject {like_op} %s OR extracted_text {like_op} %s)")
                    term_params.extend([w_term, w_term, w_term])

                term_sql = f"""
                    SELECT id, case_title, court_name, decision_date, citation, 
                           case_number, subject, source_url, extracted_text
                    FROM judgments
                    WHERE ({" OR ".join(term_clauses)})
                """
                if court_filter:
                    term_sql += f" AND court_name {like_op} %s"
                    term_params.append(f"%{court_filter}%")
                if subject_filter:
                    term_sql += f" AND subject {like_op} %s"
                    term_params.append(f"%{subject_filter}%")
                if year_filter:
                    if is_pg:
                        term_sql += " AND EXTRACT(YEAR FROM decision_date)::TEXT = %s"
                    else:
                        term_sql += " AND strftime('%Y', decision_date) = %s"
                    term_params.append(str(year_filter))

                term_sql += f" ORDER BY decision_date DESC LIMIT {limit};"
                rows = db.fetch_all(term_sql, tuple(term_params))

        saved_ids = cls._get_saved_case_ids()

        results = []
        for r in rows:
            text = r.get("extracted_text") or ""
            # Generate excerpt around matching query word
            excerpt = cls._extract_snippet(text, query)

            results.append({
                "id": r["id"],
                "judgment_id": r["id"],
                "case_title": r.get("case_title"),
                "court_name": r.get("court_name"),
                "decision_date": str(r.get("decision_date") or ""),
                "citation": r.get("citation"),
                "case_number": r.get("case_number"),
                "subject": r.get("subject"),
                "source_url": r.get("source_url"),
                "similarity_score": None,  # Not applicable for keyword search
                "relevance_percentage": None,
                "matched_passage": excerpt,
                "page_number": None,  # Preserves None, never defaults to 1
                "search_mode": "keyword",
                "is_saved": r["id"] in saved_ids
            })

        return results

    @classmethod
    def _extract_snippet(cls, text: str, query: str, context_chars: int = 180) -> str:
        """Extracts a readable snippet of text around the matched search terms."""
        if not text:
            return "No text available."
        match = re.search(re.escape(query), text, re.IGNORECASE)
        if not match and " " in query:
            words = [w for w in re.findall(r"\w+", query) if len(w) >= 3]
            for word in words:
                w_match = re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE)
                if w_match:
                    match = w_match
                    break
        if not match:
            return text[:200] + ("..." if len(text) > 200 else "")
        start = max(0, match.start() - context_chars // 2)
        end = min(len(text), match.end() + context_chars // 2)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    @staticmethod
    def _get_saved_case_ids() -> set:
        """Helper to get list of judgment IDs currently saved by the user."""
        try:
            rows = db.fetch_all("SELECT judgment_id FROM saved_cases;")
            return {r["judgment_id"] for r in rows}
        except Exception:
            return set()
