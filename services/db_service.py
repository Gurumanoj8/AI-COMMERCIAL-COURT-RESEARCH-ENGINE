"""
Database Service for Court Research AI.
Manages connections and transactions using psycopg for PostgreSQL.
Includes a transparent development fallback to SQLite if PostgreSQL is not reachable,
ensuring beginner resilience while strictly preserving PostgreSQL as the primary production engine.
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import psycopg
from psycopg.rows import dict_row

from config import Config

BASE_DIR = Path(__file__).resolve().parent.parent


class DatabaseService:
    """Encapsulates all database interactions for the application."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine_type = "postgresql"
        self.is_connected = False
        self.last_error = None

        # On Vercel, copy SQLite file to /tmp because root filesystem is read-only
        if Config.IS_VERCEL:
            tmp_db = Path("/tmp/court_research.db")
            if not tmp_db.exists():
                src_db = BASE_DIR / "court_research.db"
                if src_db.exists():
                    try:
                        import shutil
                        shutil.copy2(src_db, tmp_db)
                    except Exception as e:
                        print(f"[!] Note copying SQLite to /tmp on Vercel: {e}")
            self.sqlite_path = tmp_db
        else:
            self.sqlite_path = BASE_DIR / "court_research.db"

        # Test initial connection
        self._detect_and_initialize()

    def _detect_and_initialize(self):
        """Attempts connection to PostgreSQL; if unreachable within 1 second, prepares SQLite fallback."""
        try:
            with psycopg.connect(self.database_url, connect_timeout=1, autocommit=True, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    self.is_connected = True
                    self.engine_type = "postgresql"
                    self.last_error = None
        except Exception as err:
            self.is_connected = False
            self.last_error = str(err)
            self.engine_type = "sqlite_fallback"
            self._init_sqlite_fallback()

    def _init_sqlite_fallback(self):
        """Initializes tables in local SQLite database for development when PostgreSQL is offline."""
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_title TEXT NOT NULL,
                court_name TEXT NOT NULL,
                decision_date TEXT,
                citation TEXT,
                case_number TEXT,
                subject TEXT,
                source_url TEXT,
                document_path TEXT,
                extracted_text TEXT,
                material_facts TEXT,
                legal_issues TEXT,
                arguments TEXT,
                judicial_reasoning TEXT,
                ratio_decidendi TEXT,
                final_decision TEXT,
                relevant_statutes TEXT,
                processing_status TEXT DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS judgment_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                search_mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER NOT NULL UNIQUE REFERENCES judgments(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # If empty, populate initial sample benchmark cases so app is immediately testable
        cur.execute("SELECT COUNT(*) FROM judgments")
        count = cur.fetchone()[0]
        if count == 0:
            self._load_fallback_sample_data(conn)

        conn.commit()
        conn.close()

    def _load_fallback_sample_data(self, conn: sqlite3.Connection):
        """Loads sample benchmark cases into the SQLite fallback if empty."""
        sample_path = BASE_DIR / "database" / "sample_data.sql"
        if not sample_path.exists():
            return
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            # SQLite does not support setval; strip setval statements
            script = "\n".join([
                line for line in content.splitlines() 
                if not line.strip().startswith("SELECT setval")
            ])
            conn.executescript(script)
        except Exception as e:
            print(f"[!] Warning populating fallback sample data: {e}")

    def check_health(self, recheck: bool = False) -> Dict[str, Any]:
        """Returns the current database operational health status."""
        if recheck and self.engine_type != "postgresql":
            try:
                with psycopg.connect(self.database_url, connect_timeout=1, autocommit=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                        self.is_connected = True
                        self.engine_type = "postgresql"
                        self.last_error = None
            except Exception as e:
                self.is_connected = False
                self.last_error = str(e)

        return {
            "status": "healthy" if self.is_connected or self.engine_type == "sqlite_fallback" else "degraded",
            "active_engine": self.engine_type,
            "is_postgresql_connected": self.is_connected,
            "notice": (
                "Connected to primary PostgreSQL database via psycopg."
                if self.engine_type == "postgresql"
                else "Running in SQLite development fallback because PostgreSQL was unreachable at configured port."
            )
        }

    def fetch_all(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns all matching rows as dictionaries."""
        params = params or ()
        if self.engine_type == "postgresql":
            with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # Convert %s placeholders to ? placeholders for SQLite
            sql_conv = query.replace("%s", "?")
            cur.execute(sql_conv, params)
            rows = [dict(row) for row in cur.fetchall()]
            conn.close()
            return rows

    def fetch_one(self, query: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Executes a SELECT query and returns a single row as a dictionary."""
        params = params or ()
        if self.engine_type == "postgresql":
            with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            sql_conv = query.replace("%s", "?")
            cur.execute(sql_conv, params)
            row = cur.fetchone()
            conn.close()
            return dict(row) if row else None

    def execute(self, query: str, params: Optional[Tuple] = None) -> int:
        """Executes an INSERT, UPDATE, or DELETE query and returns the affected row count or last id."""
        params = params or ()
        if self.engine_type == "postgresql":
            with psycopg.connect(self.database_url, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.rowcount
        else:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            sql_conv = query.replace("%s", "?")
            cur.execute(sql_conv, params)
            conn.commit()
            rowcount = cur.rowcount
            conn.close()
            return rowcount

    def insert_and_return_id(self, query: str, params: Tuple) -> int:
        """Inserts a record and returns its generated primary key ID."""
        if self.engine_type == "postgresql":
            # PostgreSQL supports RETURNING id
            if "RETURNING" not in query.upper():
                query = query.rstrip("; ") + " RETURNING id;"
            with psycopg.connect(self.database_url, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                    return row[0]
        else:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            # Strip RETURNING id for SQLite if present
            sqlite_query = query
            if "RETURNING id" in sqlite_query:
                sqlite_query = sqlite_query.replace("RETURNING id", "").strip()
            sql_conv = sqlite_query.replace("%s", "?")
            cur.execute(sql_conv, params)
            conn.commit()
            last_id = cur.lastrowid
            conn.close()
            return last_id

    def execute_transaction(self, operations: List[Tuple[str, Tuple]]):
        """Executes multiple queries inside a single atomic transaction."""
        if self.engine_type == "postgresql":
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    for query, params in operations:
                        cur.execute(query, params)
                conn.commit()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            try:
                for query, params in operations:
                    sql_conv = query.replace("%s", "?")
                    cur.execute(sql_conv, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()


# Singleton instance
db = DatabaseService()
