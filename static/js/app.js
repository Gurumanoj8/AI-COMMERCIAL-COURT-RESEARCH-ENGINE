/**
 * AI Commercial Court Research Engine - Frontend Interactions
 * Vanilla JavaScript utilizing Fetch API for asynchronous backend communication.
 * Preserves all UI styling, themes, animations, and layouts.
 */

// Configurable API Base URL per Phase 3
const API_BASE_URL = (function () {
  if (typeof window.API_BASE_URL !== "undefined" && window.API_BASE_URL) {
    return window.API_BASE_URL;
  }
  if (window.location && window.location.protocol && window.location.protocol.startsWith("http")) {
    if (window.location.port && window.location.port !== "5000" && 
       (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")) {
      return `http://${window.location.hostname}:5000`;
    }
    return window.location.origin;
  }
  return "http://127.0.0.1:5000";
})();

/**
 * Reusable unified API request function with robust error handling and JSON parsing
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;
  try {
    const res = await fetch(url, options);
    let data = null;
    try {
      data = await res.json();
    } catch (parseErr) {
      data = null;
    }

    if (!res.ok) {
      const errMsg = (data && data.error) ? data.error : `Server returned status ${res.status}: ${res.statusText}`;
      return { success: false, error: errMsg, status: res.status, data };
    }
    return data || { success: true };
  } catch (err) {
    console.error(`Network error requesting ${endpoint}:`, err);
    return {
      success: false,
      error: `No API connected: The backend server is unreachable at ${API_BASE_URL}. Please ensure the Flask backend is running. (${err.message})`,
      isNetworkError: true
    };
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMobileNav();
  initBookmarkButtons();
  initUploadHandler();
  initAssistantHandler();
  initCompareHandler();
  initHistoryHandler();
});

/**
 * Responsive Mobile Navigation Toggle
 */
function initMobileNav() {
  const toggleBtn = document.getElementById("mobileNavToggle");
  const navLinks = document.getElementById("navLinks");
  if (toggleBtn && navLinks) {
    toggleBtn.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }
}

/**
 * Case Bookmark / Save Buttons (Phase 11)
 */
function initBookmarkButtons() {
  document.querySelectorAll(".btn-save-case").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      const judgmentId = btn.dataset.id;
      const isCurrentlySaved = btn.dataset.saved === "true";

      const originalHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = `<span style="opacity:0.7">Updating...</span>`;

      try {
        const method = isCurrentlySaved ? "DELETE" : "POST";
        const result = await apiRequest(`/api/saved-cases/${judgmentId}`, { method });

        if (result.success) {
          if (isCurrentlySaved) {
            btn.dataset.saved = "false";
            btn.innerHTML = `☆ Save Case`;
            btn.classList.remove("btn-gold");
            btn.classList.add("btn-secondary");
          } else {
            btn.dataset.saved = "true";
            btn.innerHTML = `★ Saved`;
            btn.classList.remove("btn-secondary");
            btn.classList.add("btn-gold");
          }
        } else {
          alert(`Could not update bookmark: ${result.error || "Please try again."}`);
          btn.innerHTML = originalHtml;
        }
      } catch (err) {
        console.error("Save case error:", err);
        btn.innerHTML = originalHtml;
      } finally {
        btn.disabled = false;
      }
    });
  });
}

/**
 * PDF Upload & Ingestion Workflow (Phase 4)
 */
function initUploadHandler() {
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("pdfFileInput");
  const uploadForm = document.getElementById("judgmentUploadForm");
  const progressBox = document.getElementById("uploadProgressBox");
  const errorBox = document.getElementById("uploadErrorBox");
  const successBox = document.getElementById("uploadSuccessBox");

  if (!uploadForm || !fileInput) return;

  // Click dropzone to open file picker
  if (dropzone) {
    dropzone.addEventListener("click", (e) => {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    // Drag & drop visual feedback
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  function handleFileSelection(file) {
    const fileNameDisplay = document.getElementById("selectedFileName");
    if (fileNameDisplay) {
      fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
  }

  // Handle Form Submission
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files || !fileInput.files[0]) {
      alert("Please select a PDF file first.");
      return;
    }

    const file = fileInput.files[0];
    if (file.size > 16 * 1024 * 1024) {
      alert("File exceeds maximum allowed size of 16 MB.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF documents (.pdf) are permitted.");
      return;
    }

    // Reset feedback alerts
    if (errorBox) {
      errorBox.style.display = "none";
      errorBox.innerHTML = "";
    }
    if (successBox) {
      successBox.style.display = "none";
      successBox.innerHTML = "";
    }
    if (progressBox) progressBox.style.display = "block";

    const submitBtn = document.getElementById("submitUploadBtn");
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Processing Judgment Pipeline...";
    }

    const formData = new FormData(uploadForm);

    try {
      // Direct POST using browser FormData (Do NOT set multipart Content-Type header manually!)
      const response = await fetch(`${API_BASE_URL}/api/judgments/upload`, {
        method: "POST",
        body: formData,
      });

      let result;
      try {
        result = await response.json();
      } catch (err) {
        result = { success: false, error: "Server returned non-JSON response." };
      }

      if (progressBox) progressBox.style.display = "none";

      if (response.ok && result.success) {
        if (successBox) {
          successBox.innerHTML = `
            <strong>Judgment Successfully Processed &amp; Indexed!</strong><br>
            Case Title: <em>${escapeHtml(result.case_title || file.name)}</em><br>
            Pages Analyzed: ${result.pages_processed || 1} | Passages Indexed: ${result.chunks_indexed || 1}<br>
            Status: <span class="badge" style="background:#eafaf1; color:#2e7d32;">${result.status || 'completed'}</span>
            <div style="margin-top: 1rem;">
              <a href="/judgment/${result.judgment_id}" class="btn btn-primary btn-sm">View Judgment Details</a>
              <a href="/similar/${result.judgment_id}" class="btn btn-secondary btn-sm" style="margin-left: 0.5rem;">Find Similar Precedents</a>
            </div>
          `;
          successBox.style.display = "block";
        }
        uploadForm.reset();
        const fileNameDisplay = document.getElementById("selectedFileName");
        if (fileNameDisplay) fileNameDisplay.textContent = "";
      } else {
        if (errorBox) {
          errorBox.innerHTML = `<strong>Upload Failed:</strong> ${escapeHtml(result.error || "An error occurred during processing.")}`;
          if (result.is_scanned) {
            errorBox.innerHTML += `<br><small>Note: This document contains image-only scans without computer-readable text. OCR processing is required for scanned judgments.</small>`;
          }
          errorBox.style.display = "block";
        }
      }
    } catch (err) {
      console.error("Upload network error:", err);
      if (progressBox) progressBox.style.display = "none";
      if (errorBox) {
        errorBox.innerHTML = `<strong>No API connected:</strong> The backend server is unreachable at <code>${API_BASE_URL}</code>. Please make sure the Flask application is started. (${escapeHtml(err.message)})`;
        errorBox.style.display = "block";
      }
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Index Judgment Document";
      }
    }
  });
}

/**
 * AI Legal Assistant Question Answering (Phase 8)
 */
function initAssistantHandler() {
  const form = document.getElementById("assistantForm");
  const questionInput = document.getElementById("assistantQuestion");
  const loadingIndicator = document.getElementById("assistantLoading");
  const resultsContainer = document.getElementById("assistantResults");

  if (!form || !questionInput) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    if (loadingIndicator) loadingIndicator.style.display = "block";
    if (resultsContainer) resultsContainer.style.display = "none";

    const submitBtn = document.getElementById("btn-submit-assistant");
    if (submitBtn) submitBtn.disabled = true;

    try {
      const data = await apiRequest("/api/assistant/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (loadingIndicator) loadingIndicator.style.display = "none";

      if (data.success && resultsContainer) {
        renderAssistantResponse(data, resultsContainer);
        resultsContainer.style.display = "block";
      } else {
        alert(data.error || "Could not retrieve response from AI assistant.");
      }
    } catch (err) {
      console.error("Assistant query error:", err);
      if (loadingIndicator) loadingIndicator.style.display = "none";
      alert(`No API connected: The assistant could not communicate with ${API_BASE_URL}. Ensure Flask is running.`);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

/**
 * Case Comparison Client Validation (Phase 10)
 */
function initCompareHandler() {
  const selectA = document.getElementById("caseSelectA");
  const selectB = document.getElementById("caseSelectB");
  const compareForm = selectA ? selectA.closest("form") : null;

  if (compareForm && selectA && selectB) {
    compareForm.addEventListener("submit", (e) => {
      if (selectA.value && selectB.value && selectA.value === selectB.value) {
        e.preventDefault();
        alert("Please select two different cases to compare. A judgment cannot be compared with itself.");
      }
    });
  }
}

/**
 * Research History Deletion Handling (Phase 12)
 */
function initHistoryHandler() {
  document.querySelectorAll(".btn-delete-history").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      const historyId = btn.dataset.id;
      if (!historyId) return;

      const itemRow = btn.closest(".history-item-row") || btn.parentElement;
      btn.disabled = true;

      const res = await apiRequest(`/api/history/${historyId}`, { method: "DELETE" });
      if (res.success) {
        if (itemRow) itemRow.remove();
      } else {
        alert(res.error || "Could not delete history entry.");
        btn.disabled = false;
      }
    });
  });
}

/**
 * Renders AI Assistant response with clear separation of retrieved evidence and AI synthesis
 */
function renderAssistantResponse(data, container) {
  let html = "";

  // 1. Legal Precedent Synthesis Section
  if (data.generated_answer) {
    html += `
      <div class="assistant-response-card fade-in">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
          <h3 style="color: var(--color-navy);">Legal Precedent Synthesis &amp; Analysis</h3>
          <span class="badge" style="background: var(--color-blue-subtle); color: var(--color-blue);">
            ${escapeHtml(data.llm_provider || "Precedent Knowledge Engine")}
          </span>
        </div>
        <div style="white-space: pre-wrap; line-height: 1.7; color: var(--color-body-dark); font-size: 1.05rem;">
          ${escapeHtml(data.generated_answer)}
        </div>
      </div>
    `;
  } else if (!data.evidence || data.evidence.length === 0) {
    html += `
      <div class="assistant-response-card fade-in">
        <p style="color: var(--color-body); font-style: italic; margin: 0;">
          The indexed commercial court records do not contain sufficient information to answer this question.
        </p>
      </div>
    `;
  }

  // 2. Retrieved Evidence Passages from Database / FAISS
  html += `
    <div class="assistant-response-card fade-in" style="margin-top: 1.5rem;">
      <h3 style="color: var(--color-navy); margin-bottom: 0.5rem;">
        Direct Retrieved Case Authorities (${data.evidence_count || 0} Passages)
      </h3>
      <p style="font-size: 0.875rem; color: var(--color-body); margin-bottom: 1.25rem;">
        Passages retrieved from indexed commercial court judgment records.
      </p>
  `;

  if (data.evidence && data.evidence.length > 0) {
    data.evidence.forEach((item, index) => {
      const isBenchmark = item.case_title && item.case_title.toUpperCase().includes('SYNTHETIC');
      html += `
        <div class="evidence-quote-box">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div>
              <strong style="color: var(--color-navy);">[${index + 1}] ${escapeHtml(item.case_title)}</strong>
              ${isBenchmark ? '<span class="badge" style="background:#fff3cd; color:#856404; font-size:0.68rem; margin-left:0.5rem;">Benchmark Record</span>' : ''}
              <div style="font-size: 0.8rem; color: var(--color-body); margin-top: 0.2rem;">
                ${escapeHtml(item.court_name || "Court not specified")} &bull; ${escapeHtml(item.decision_date || "Date Unspecified")} &bull; Citation: ${escapeHtml(item.citation || "N/A")}
              </div>
            </div>
            ${item.relevance_percentage ? `<span class="badge" style="background:#eaf1f9; color:#315b91;">${item.relevance_percentage}% Match</span>` : ""}
          </div>
          <div style="font-style: italic; color: var(--color-body-dark); margin: 0.5rem 0;">
            "${escapeHtml(item.chunk_text)}"
          </div>
          <div style="text-align: right; margin-top: 0.5rem;">
            <a href="/judgment/${item.judgment_id}" class="btn btn-secondary btn-sm" target="_blank">
              View Case Record &amp; Full Analysis &nearr;
            </a>
          </div>
        </div>
      `;
    });
  } else {
    html += `<p>No relevant judgment passages matched the criteria.</p>`;
  }

  // 3. Mandatory Legal Disclaimer
  html += `
      <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--color-border); font-size: 0.8rem; color: var(--color-body);">
        <strong>Disclaimer:</strong> ${escapeHtml(data.disclaimer || "This tool is an educational research aid and does not constitute formal legal counsel.")}
      </div>
    </div>
  `;

  container.innerHTML = html;
}

/**
 * Utility: HTML Escape
 */
function escapeHtml(text) {
  if (!text) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.toString().replace(/[&<>"']/g, (m) => map[m]);
}
