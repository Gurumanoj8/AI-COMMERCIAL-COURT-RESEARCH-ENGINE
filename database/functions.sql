-- ====================================================================
-- AI Commercial Court Research Engine - Stored Functions & Triggers
-- Language: PL/pgSQL
-- ====================================================================

-- 1. Trigger Function: Automatically update updated_at timestamp on row change
CREATE OR REPLACE FUNCTION update_judgment_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to judgments table
DROP TRIGGER IF EXISTS trg_update_judgment_timestamp ON judgments;
CREATE TRIGGER trg_update_judgment_timestamp
BEFORE UPDATE ON judgments
FOR EACH ROW
EXECUTE FUNCTION update_judgment_timestamp();

-- 2. Stored Function: Retrieve judgment metadata along with total indexed chunks count
CREATE OR REPLACE FUNCTION get_judgment_summary(p_judgment_id INTEGER)
RETURNS TABLE (
    judgment_id INTEGER,
    title VARCHAR(255),
    court VARCHAR(150),
    dec_date DATE,
    cite VARCHAR(100),
    sub VARCHAR(100),
    chunk_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        j.id,
        j.case_title,
        j.court_name,
        j.decision_date,
        j.citation,
        j.subject,
        COUNT(c.id) AS chunk_count
    FROM judgments j
    LEFT JOIN judgment_chunks c ON j.id = c.judgment_id
    WHERE j.id = p_judgment_id
    GROUP BY j.id, j.case_title, j.court_name, j.decision_date, j.citation, j.subject;
END;
$$ LANGUAGE plpgsql;
