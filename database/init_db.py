"""
Database Initialization Script for PostgreSQL.
Run this script to create tables, functions, triggers, and optionally load sample data.

Usage:
    python database/init_db.py [--sample-data]
"""

import sys
import argparse
from pathlib import Path
import psycopg
from psycopg import sql

# Add project root to sys.path so config can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


def parse_db_url(url: str):
    """Simple parser for postgresql connection parameters."""
    return url


def run_init(include_sample_data: bool = False):
    """Executes schema.sql, functions.sql, and optionally sample_data.sql."""
    db_url = Config.DATABASE_URL
    print(f"[*] Connecting to PostgreSQL at: {db_url}")

    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    functions_file = PROJECT_ROOT / "database" / "functions.sql"
    sample_file = PROJECT_ROOT / "database" / "sample_data.sql"

    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                print(f"[+] Applying schema from {schema_file.name}...")
                with open(schema_file, "r", encoding="utf-8") as f:
                    cur.execute(f.read())
                print("[OK] Tables and indexes created successfully.")

                print(f"[+] Applying stored functions & triggers from {functions_file.name}...")
                with open(functions_file, "r", encoding="utf-8") as f:
                    cur.execute(f.read())
                print("[OK] Functions and triggers installed successfully.")

                if include_sample_data:
                    print(f"[+] Loading sample benchmark records from {sample_file.name}...")
                    with open(sample_file, "r", encoding="utf-8") as f:
                        cur.execute(f.read())
                    print("[OK] Sample benchmark records loaded successfully.")

        print("\n=======================================================")
        print(" [SUCCESS] PostgreSQL Database initialized successfully!")
        print("=======================================================")
        return True

    except psycopg.OperationalError as e:
        print(f"\n[!] Connection Error: Could not connect to PostgreSQL database.")
        print(f"    Details: {e}")
        print("\n[HINT] Ensure PostgreSQL is running on the configured host & port.")
        print("       Check your credentials in the .env file (DATABASE_URL).")
        return False
    except Exception as e:
        print(f"\n[!] Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL Database for Court Research AI")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Populate database with synthetic benchmark records for testing"
    )
    args = parser.parse_args()
    success = run_init(include_sample_data=args.sample_data)
    sys.exit(0 if success else 1)
