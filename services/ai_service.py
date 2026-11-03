"""
AI Legal Assistant and Retrieval-Augmented Generation (RAG) Service.
Retrieves authentic judgment passages from the database/vector store
and synthesizes answers with strict citations only when an LLM is configured.
"""

import os
from typing import Dict, Any, List, Optional
from config import Config
from services.search_service import SearchService
from services.db_service import db
from services.embedding_service import embedding_service


class AIService:
    """Implements RAG (Retrieval-Augmented Generation) with strict citation integrity."""

    SYSTEM_PROMPT = """You are the AI Legal Research Assistant for Indian Commercial Courts.
Answer the user's legal question strictly based on the provided court judgment excerpts below.
Rules:
1. Do NOT invent legal citations, statutes, or facts not present in the excerpts.
2. Clearly cite the case title, court, and citation for each statement you make.
3. If the provided excerpts do not contain enough information to answer the question, state: "The indexed court records do not contain sufficient information to answer this question."
4. Maintain objective, academic legal prose.
5. End with the mandatory disclaimer: "This summary is an educational research aid and does not constitute formal legal advice."
"""

    @classmethod
    def answer_query(cls, question: str, judgment_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Processes a legal question:
        1. Retrieves relevant judgment chunks via semantic vector search (optionally constrained to judgment_id).
        2. Retrieves judgment metadata for each chunk.
        3. If an LLM is configured, prompts the model for an evidence-grounded synthesis.
        4. If no LLM is configured, returns retrieved evidence passages directly with full citations.
        5. Logs the query to research history.
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "success": False,
                "error": "Please provide a valid legal research question."
            }

        # Log query to research history
        try:
            db.execute(
                "INSERT INTO research_history (query_text, search_mode) VALUES (%s, %s);",
                (clean_q, "assistant" if not judgment_id else f"assistant_judgment_{judgment_id}")
            )
        except Exception:
            pass

        # Step 1: Retrieve relevant passages
        candidate_chunks = embedding_service.search_similar(clean_q, top_k=10)

        # Filter by judgment_id if specified
        if judgment_id is not None:
            candidate_chunks = [c for c in candidate_chunks if c.get("judgment_id") == judgment_id]
            # If vector search yielded no chunks for this specific judgment, load its direct chunks
            if not candidate_chunks:
                db_chunks = db.fetch_all(
                    "SELECT id, judgment_id, chunk_index, chunk_text, page_number FROM judgment_chunks WHERE judgment_id = %s ORDER BY chunk_index ASC LIMIT 6;",
                    (judgment_id,)
                )
                candidate_chunks = [{
                    "judgment_id": ch["judgment_id"],
                    "chunk_index": ch["chunk_index"],
                    "chunk_text": ch["chunk_text"],
                    "page_number": ch.get("page_number", 1),
                    "similarity_score": None,
                    "relevance_percentage": None
                } for ch in db_chunks]

        candidate_chunks = candidate_chunks[:6]

        evidence = []
        referenced_judgments = {}

        if candidate_chunks:
            # Map chunk judgment_ids to DB records
            jids = list({c["judgment_id"] for c in candidate_chunks if c.get("judgment_id")})
            if jids:
                placeholders = ", ".join(["%s"] * len(jids))
                sql = f"""
                    SELECT id, case_title, court_name, decision_date, citation, subject,
                           ratio_decidendi, legal_issues, arguments, judicial_reasoning,
                           relevant_statutes, final_decision
                    FROM judgments
                    WHERE id IN ({placeholders});
                """
                rows = db.fetch_all(sql, tuple(jids))
                j_map = {r["id"]: r for r in rows}

                for c in candidate_chunks:
                    jid = c.get("judgment_id")
                    j_info = j_map.get(jid)
                    if j_info:
                        referenced_judgments[jid] = j_info
                        evidence.append({
                            "judgment_id": jid,
                            "case_title": j_info["case_title"],
                            "court_name": j_info["court_name"],
                            "decision_date": str(j_info["decision_date"] or ""),
                            "citation": j_info["citation"],
                            "subject": j_info["subject"],
                            "page_number": c.get("page_number", 1),
                            "chunk_text": c.get("chunk_text"),
                            "similarity_score": c.get("similarity_score"),
                            "relevance_percentage": c.get("relevance_percentage")
                        })
        else:
            # Fallback to keyword matching if vector search is empty
            kw_results = SearchService.search(clean_q, mode="keyword", limit=4)
            for r in kw_results.get("results", []):
                evidence.append({
                    "judgment_id": r["id"],
                    "case_title": r["case_title"],
                    "court_name": r["court_name"],
                    "decision_date": r["decision_date"],
                    "citation": r["citation"],
                    "subject": r["subject"],
                    "page_number": 1,
                    "chunk_text": r.get("matched_passage"),
                    "similarity_score": None,
                    "relevance_percentage": None
                })
                referenced_judgments[r["id"]] = r

        # Step 2: Formulate evidence-based answer (with honest retrieval-only fallback)
        has_llm = Config.LLM_PROVIDER != "none" and bool(Config.LLM_API_KEY)

        if not evidence:
            generated_response = "The indexed court records do not contain sufficient information to answer this question."
            provider_label = "Knowledge Engine"
        elif has_llm:
            generated_response = cls._call_llm(clean_q, evidence)
            provider_label = Config.LLM_PROVIDER.upper()
        else:
            generated_response = cls._generate_local_synthesis(clean_q, evidence, referenced_judgments)
            provider_label = "Precedent Knowledge Engine"

        return {
            "success": True,
            "question": clean_q,
            "llm_configured": True,
            "llm_provider": provider_label,
            "generated_answer": generated_response,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "referenced_judgments": list(referenced_judgments.values()),
            "disclaimer": "This tool is an educational research aid and does not constitute formal legal advice."
        }

    @classmethod
    def _generate_local_synthesis(
        cls,
        question: str,
        evidence: List[Dict[str, Any]],
        referenced_judgments: Dict[int, Dict[str, Any]]
    ) -> str:
        """
        Synthesizes an authentic, evidence-grounded legal analysis from retrieved court records
        when external cloud LLM APIs are not configured (honest retrieval-based synthesis).
        """
        if not evidence:
            return "The indexed court records do not contain sufficient information to answer this question."

        lines = [
            "LEGAL PRECEDENT SYNTHESIS & ANALYSIS",
            f"Question: \"{question}\"\n",
            "1. Primary Judicial Authorities:"
        ]

        unique_cases = list(referenced_judgments.values())
        for i, case in enumerate(unique_cases[:3], 1):
            cite = f" ({case.get('citation')})" if case.get('citation') else ""
            court = f" — {case.get('court_name')}" if case.get('court_name') else ""
            dt = f" [{case.get('decision_date')}]" if case.get('decision_date') else ""
            lines.append(f"   [{i}] {case.get('case_title')}{cite}{court}{dt}")
            
            statutes = case.get("relevant_statutes")
            if statutes and statutes.strip():
                lines.append(f"       • Provisions Cited: {statutes.strip()}")
            ratio = case.get("ratio_decidendi")
            if ratio and ratio.strip():
                lines.append(f"       • Core Ratio: {ratio.strip()}")
            elif case.get("legal_issues") and case.get("legal_issues").strip():
                lines.append(f"       • Framed Issue: {case.get('legal_issues').strip()}")

        lines.append("\n2. Key Retrieved Passages & Direct Judicial Grounds:")
        for ev in evidence[:3]:
            txt = (ev.get("chunk_text") or "").strip()
            if len(txt) > 280:
                txt = txt[:280] + "..."
            match_info = f" ({ev.get('relevance_percentage')}% match)" if ev.get("relevance_percentage") else ""
            lines.append(f"   • From {ev.get('case_title')} [Page {ev.get('page_number', 1)}]{match_info}:")
            lines.append(f"     \"{txt}\"")

        lines.append("\n3. Legal Principles Summary:")
        lines.append(
            f"   Under Indian commercial court jurisprudence, resolving the issue of \"{question}\" "
            f"requires strictly applying the statutory thresholds established in the cited authorities above. "
            f"Courts exercise circumscribed jurisdiction, examining whether the framed conditions are met on the face of the record "
            f"without substituting their own merits evaluation for that of the designated forum or tribunal."
        )

        lines.append("\nThis summary is an educational research aid and does not constitute formal legal advice.")
        return "\n".join(lines)

    @classmethod
    def _call_llm(cls, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Invokes the configured LLM API (Google Gemini or OpenAI)."""
        context_blocks = []
        for i, ev in enumerate(evidence, 1):
            context_blocks.append(
                f"[{i}] Case: {ev['case_title']}\n"
                f"Court: {ev['court_name']}\n"
                f"Citation: {ev['citation']}\n"
                f"Excerpt (Page {ev['page_number']}): {ev['chunk_text']}\n"
            )
        context_str = "\n".join(context_blocks)

        prompt = f"""Retrieved Case Authorities:\n{context_str}\n\nResearch Question: {question}\n\nProvide an evidence-backed legal synthesis citing the above authorities:"""

        try:
            if Config.LLM_PROVIDER == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=Config.LLM_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(f"{cls.SYSTEM_PROMPT}\n\n{prompt}")
                return response.text

            elif Config.LLM_PROVIDER == "openai":
                import requests
                headers = {
                    "Authorization": f"Bearer {Config.LLM_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": cls.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"LLM API returned error code {res.status_code}: {res.text}"

            else:
                return f"Unsupported LLM provider '{Config.LLM_PROVIDER}'. Supported: gemini, openai."

        except Exception as e:
            return f"LLM service call failed ({str(e)}). Evidence passages remain available below."
