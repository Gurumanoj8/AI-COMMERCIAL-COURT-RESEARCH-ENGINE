"""
PDF Processing Tests.
Validates file header verification, PyMuPDF text extraction,
page number preservation, chunk segmentation, and corrupt file detection.
"""

import io
import pymupdf
from services.pdf_service import PDFService


def test_pdf_magic_bytes_validation():
    """Verifies that only files with authentic %PDF- headers are accepted."""
    # Valid PDF header
    valid_stream = io.BytesIO(b"%PDF-1.4\n%...\n")
    is_valid, msg = PDFService.is_valid_pdf(valid_stream, "judgment.pdf")
    assert is_valid is True

    # Fake/Spoofed text file renamed to .pdf
    fake_stream = io.BytesIO(b"This is a text file renamed to .pdf")
    is_valid, msg = PDFService.is_valid_pdf(fake_stream, "fake.pdf")
    assert is_valid is False
    assert "Missing '%PDF-' header" in msg

    # Invalid extension
    is_valid, msg = PDFService.is_valid_pdf(valid_stream, "judgment.docx")
    assert is_valid is False
    assert "Invalid file extension" in msg


def test_real_pdf_extraction_with_pymupdf(tmp_path):
    """Creates a real multi-page PDF using PyMuPDF and tests extraction & chunking."""
    pdf_path = tmp_path / "test_judgment.pdf"

    # Create synthetic 2-page PDF
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "IN THE HIGH COURT OF DELHI AT NEW DELHI (COMMERCIAL DIVISION)\n"
        "Alpha Corporation v. Beta Infrastructure Ltd.\n"
        "Judgment concerning arbitration agreement under Section 7 of the 1996 Act."
    )
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Held, an arbitration clause in an unstamped contract is enforceable for interim measures.\n"
        "Application under Section 9 allowed with costs."
    )
    doc.save(str(pdf_path))
    doc.close()

    # Extract text using PDFService
    extraction = PDFService.extract_text_and_pages(pdf_path)
    assert extraction["success"] is True
    assert extraction["total_pages"] == 2
    assert "Alpha Corporation" in extraction["full_text"]
    assert "Section 9 allowed" in extraction["full_text"]
    assert len(extraction["pages"]) == 2

    # Test chunking
    chunks = PDFService.chunk_text(extraction["pages"], chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    assert chunks[0]["page_number"] == 1
    assert "chunk_text" in chunks[0]


def test_empty_or_corrupt_pdf(tmp_path):
    """Verifies graceful error handling for corrupted files."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nCORRUPT_INVALID_DATA_GARBAGE")

    extraction = PDFService.extract_text_and_pages(corrupt_file)
    # PyMuPDF should fail to parse or report no text
    assert extraction["success"] is False
    assert "error" in extraction


def test_indian_court_metadata_extraction():
    """Verifies extraction of authentic Indian court headers from judgment text."""
    sample_indian_header = """http://JUDIS.NIC.IN 
SUPREME COURT OF INDIA
Page 1 of 16 
PETITIONER:
RADHAKRISHNA SIVADUTTA RAI AND ORS.
        Vs.
RESPONDENT:
TAYEBALLI DAWOODBEAI
DATE OF JUDGMENT:
13/10/1961
BENCH:
GAJENDRAGADKAR, P.B.
CITATION:
 1962 AIR  538            1962 SCR  Supl. (1)  81
ACT:
     Breach of contract - Suit for damages - Indian Contract Act, 1872
"""
    hints = PDFService._heuristic_extract_metadata(sample_indian_header)
    assert hints["court_name"] == "Supreme Court Of India"
    assert "Radhakrishna Sivadutta Rai" in hints["case_title"]
    assert "Tayeballi" in hints["case_title"]
    assert hints["decision_date"] == "1961-10-13"
    assert "1962 AIR 538" in hints["citation"]
    assert hints["source_url"] == "http://JUDIS.NIC.IN"


def test_automatic_judgment_summarization():
    """Verifies automated summarization into the 6 required categories."""
    sample_judgment_text = """
    HEADNOTE:
    The appellant entered into an arbitration agreement and later sued for breach of contract.
    The respondent pleaded that the suit was not maintainable under Section 230 of the Contract Act.
    Held, that it is well established in commercial usage that bought and sold notes without disparity constitute a binding contract.
    The appellant as an agent for a disclosed principal had no independent right to maintain the suit.
    
    FACTS OF THE CASE:
    The appellant firm carried on business in Bombay and entered into transactions for delivery of goods.
    Disputes arose regarding non-delivery and market fluctuations.
    
    ISSUES:
    1. Whether an agent can sue on behalf of a disclosed principal under Section 230?
    2. Whether the bought and sold notes formed the conclusive terms of the contract?
    
    SUBMISSIONS:
    Senior counsel for the appellant contended that the confirmation letters superseded broker notes.
    Counsel for the respondent argued that broker notes remained final.
    
    REASONING:
    We have considered the rival contentions and commercial usage. In our view, broker notes define the scope of liability.
    
    ORDER:
    Accordingly, the appeal is dismissed with costs.
    """
    summary = PDFService.generate_judgment_summary(sample_judgment_text)
    assert "material_facts" in summary and len(summary["material_facts"]) > 10
    assert "legal_issues" in summary and "Whether" in summary["legal_issues"]
    assert "arguments" in summary and len(summary["arguments"]) > 10
    assert "judicial_reasoning" in summary and len(summary["judicial_reasoning"]) > 10
    assert "final_decision" in summary and ("dismissed" in summary["final_decision"].lower() or "appeal" in summary["final_decision"].lower())
    assert "ratio_decidendi" in summary and len(summary["ratio_decidendi"]) > 10
    assert "relevant_statutes" in summary

