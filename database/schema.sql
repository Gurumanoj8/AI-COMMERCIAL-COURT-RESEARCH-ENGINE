-- ====================================================================
-- AI Commercial Court Research Engine - Database Schema (PostgreSQL)
-- ====================================================================

-- 1. Judgments Table: Stores case metadata and extracted legal text
CREATE TABLE IF NOT EXISTS judgments (
    id SERIAL PRIMARY KEY,
    case_title VARCHAR(255) NOT NULL,
    court_name VARCHAR(150) NOT NULL,
    decision_date DATE,
    citation VARCHAR(100),
    case_number VARCHAR(100),
    subject VARCHAR(100),
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
    processing_status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Judgment Chunks Table: Stores segmented text passages for vector retrieval
CREATE TABLE IF NOT EXISTS judgment_chunks (
    id SERIAL PRIMARY KEY,
    judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    embedding TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Research History Table: Tracks queries executed by users
CREATE TABLE IF NOT EXISTS research_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    search_mode VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Saved Cases Table: Bookmarked judgments for quick review
CREATE TABLE IF NOT EXISTS saved_cases (
    id SERIAL PRIMARY KEY,
    judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_saved_judgment UNIQUE (judgment_id)
);

-- Indexes for high-performance query filtering
CREATE INDEX IF NOT EXISTS idx_judgments_court ON judgments(court_name);
CREATE INDEX IF NOT EXISTS idx_judgments_subject ON judgments(subject);
CREATE INDEX IF NOT EXISTS idx_judgments_date ON judgments(decision_date);
CREATE INDEX IF NOT EXISTS idx_chunks_judgment ON judgment_chunks(judgment_id);
CREATE INDEX IF NOT EXISTS idx_history_created ON research_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_saved_cases_judgment ON saved_cases(judgment_id);
