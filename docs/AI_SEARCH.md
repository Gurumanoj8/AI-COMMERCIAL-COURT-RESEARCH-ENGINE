# AI & Semantic Search: Vector Retrieval and RAG Architecture

## 1. Why Traditional Keyword Search Is Insufficient for Law

In Indian commercial law, judges and counsels frequently express identical legal principles using varied phraseology. For instance:
* A counsel may search: *"can court restrain encashment of performance guarantee?"*
* The judgment text may say: *"injunction against invocation of unconditional bank guarantee is permissible only upon established fraud of egregious nature."*

A strict SQL `ILIKE` keyword search fails to connect these phrases because words like "performance", "restrain", and "encashment" differ lexically from "invocation", "unconditional", and "injunction".

**Vector Semantic Search** resolves this by converting text into high-dimensional numerical coordinates that capture semantic meaning rather than literal spelling.

---

## 2. Embedding Model: Sentence Transformers

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Dimensions**: 384 dimensions
* **Computational Footprint**: ~80 MB RAM, runs natively on CPU without requiring GPU acceleration.
* **Normalization**: Embeddings are L2-normalized upon creation:
  $$\|v\|_2 = \sqrt{\sum_{i=1}^{d} v_i^2} = 1$$
  When vectors are unit-normalized, the inner product between query vector $q$ and document vector $d$ equals their **cosine similarity**:
  $$\text{Cosine Similarity}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} = q \cdot d$$

---

## 3. FAISS (Facebook AI Similarity Search)

* **Index Type**: `faiss.IndexFlatIP` (Flat Inner Product).
* **Storage**:
  * Binary index saved to `uploads/vector_index.faiss`.
  * Vector-to-metadata mapping saved to `uploads/chunk_mapping.json`.
* **Sub-millisecond Retrieval**:
  When a search query is submitted:
  1. The query text is encoded into a 384-dimensional vector.
  2. FAISS compares the query vector against all indexed chunk vectors in memory.
  3. Top candidate chunk IDs and cosine scores are retrieved in under 2 milliseconds.
  4. The candidate IDs are joined with the PostgreSQL database to retrieve the full case metadata.

---

## 4. Text Chunking with Page Citations

Court judgments can span 20 to 100+ pages. Feeding an entire judgment into an embedding model dilutes key sentences.

The application employs a sliding window chunking algorithm (`services/pdf_service.py`):
* **Chunk Size**: 500 words/characters
* **Chunk Overlap**: 100 words/characters
* **Page Preservation**: Each chunk retains the `page_number` from the original PDF parsed by PyMuPDF.
* **Benefit**: When a passage matches a query, the researcher is provided with an exact pinpoint page citation (e.g., *"Matched on Page 3"*).

---

## 5. Retrieval-Augmented Generation (RAG) & Fallback Modes

```
User Query: "What is patent illegality under Section 34?"
     │
     ├──> [ 1. Embed Query with SentenceTransformer ]
     │
     ├──> [ 2. Retrieve Top Passages from FAISS Index ]
     │
     ├──> [ 3. Query PostgreSQL for Official Case Metadata ]
     │
     ▼
Is LLM Provider Configured in .env?
 ├── YES (Gemini or OpenAI API key supplied):
 │     ├── Construct Strict Grounded Prompt with Passages
 │     ├── Model synthesizes answer citing authorities
 │     └── Display AI Synthesis + Verified Sources + Disclaimer
 │
 └── NO (LLM_PROVIDER=none or missing key):
       ├── Transparent Fallback Mode
       ├── Display Exact Retrieved Court Passages Directly
       ├── Provide Full Citations, Court Names, and Dates
       └── Display Explanatory Notice: No Hallucinations Generated
```

### Citation Integrity Rule
To prevent AI hallucinations, the assistant prompt explicitly enforces:
1. No statutory references or legal claims may be made unless directly supported by the retrieved passages.
2. If evidence is missing, the assistant must state: *"The indexed court records do not contain sufficient information to answer this question."*

---

## 6. Jurisprudential Distinction: Semantic Similarity vs. Binding Precedent

A vital distinction required for viva defense:
* **Vector Semantic Similarity**: Measures linguistic and thematic proximity between legal text passages in vector space.
* **Binding Judicial Precedent**: Governed by constitutional doctrine under **Article 141 of the Constitution of India** and judicial hierarchy:
  * Supreme Court judgments bind all High Courts and subordinate commercial courts.
  * A Division Bench judgment binds a Single Bench of the same High Court.
  * A High Court judgment carries persuasive (not binding) value in another High Court.

Semantic similarity indicates that two cases address similar facts or legal topics, but the researcher must evaluate judicial hierarchy and statutory validity to determine legal authority.
