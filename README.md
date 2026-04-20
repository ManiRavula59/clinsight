# Clinsight: Multi-Agent Medical Case Retrieval System ⚕️

Clinsight is an advanced, production-grade clinical decision support prototype inspired by the architecture in *"LLM-augmented multi-agent cooperative framework for medical case retrieval in cardiology."*

It transforms sparse clinical inquiries into deep, evidence-backed diagnostic pipelines by retrieving matching historical patient cases (PMC-Patients), re-ranking them using dense semantic and cross-encoder models, and synthesizing the final diagnosis via LLMs.

---

## I. PROBLEM STATEMENT
Healthcare systems currently suffer from several critical issues, including:
*   Difficulty retrieving clinically similar past cases.
*   Semantic mismatch between diverse medical terminology.
*   LLM hallucination risks and a lack of structured clinical reasoning.
*   High rates of medication errors (dose, interaction, contraindication).
*   Poor medication adherence after hospital discharge.

Current systems are limited, relying on simple keyword search (BM25), dense embeddings, or ungrounded LLM reasoning. None provide structured clinical validation, deterministic safety checks, real-time explainability, or deployable infrastructure on limited hardware.

---

## II. CORE OBJECTIVE
To design and implement a clinically grounded, explainable, and deployable Clinical Decision Support System (CDSS) that effectively:
1. Retrieves truly relevant patient cases.
2. Performs structured medical reasoning and validates against medical knowledge graphs.
3. Ensures medication safety inside hospitals and supports adherence outside hospitals.
4. Provides performance transparency through Information Retrieval (IR) metrics.
5. Is deployable on consumer hardware (e.g., M1 MacBook).

---

## III. RESEARCH CONTRIBUTION & EXISTING WORK
| Dimension | Base Paper | Clinsight |
| :--- | :--- | :--- |
| **Iterative LLM Loop** | Yes | Structured + deterministic KG first |
| **Knowledge Graph** | Conceptual | Deterministic + rule-based validation |
| **Reranking** | LLM heavy | ColBERT + Cross-Encoder + LLM |
| **Privacy** | Not addressed | **PII Shield integrated** |
| **Medication Safety** | No | **Integrated** |
| **Deployability** | Requires A100 GPUs | **Runs on M1 Mac** |

---

## IV. SYSTEM VISION & HIGH-LEVEL ARCHITECTURE
CLINSIGHT is a hybrid Retrieval + Knowledge Graph + Rule Engine + LLM platform designed to assist clinicians safely and responsibly, moving beyond a simple search engine. The system is built upon five integrated layers:

**A. LAYER 1: PRIVACY & QUERY UNDERSTANDING**
*   **Purpose:** To prepare the clinical query for safe and accurate retrieval.
*   **Components:** PII Detection (Presidio), Clinical Named Entity Recognition (NER), Negation Detection, and Concept Linking (UMLS/SNOMED).
*   **Actions:** Removes patient identifiers, extracts structured medical entities, avoids false matches (e.g., "no chest pain"), and normalizes medical vocabulary.

**B. LAYER 2: ORCHESTRATION & KNOWLEDGE GRAPH (MEDICAL LOGIC)**
*   **Purpose:** To reduce synonym mismatch, orchestrate agents, and add deterministic medical structure.
*   **Capabilities:** LangGraph Routing, Concept mapping, synonym expansion, hierarchy traversal, contraindication detection, and causal relationship modeling.

**C. LAYER 3: HYBRID RETRIEVAL ENGINE**
*   **Purpose:** To achieve high recall and precision filtering before LLM reasoning, reducing hallucination risk and computational cost.
*   **Stage 1 – Recall:** Dense Retrieval (FAISS HNSW), Sparse Retrieval (SQLite FTS5 / BM25), and Reciprocal Rank Fusion (RRF).
*   **Stage 2 – Precision:** ColBERT Reranking (token-level interaction) and Cross-Encoder Reranking (joint attention model).

**D. LAYER 4: CLINICAL REASONING & EXPLANATION**
*   **Purpose:** To transform evidence into actionable, explained insights. LLM does not invent facts; it reasons over retrieved and validated evidence.
*   **Components:** Structured Clinical Summary, Evidence-grounded reasoning, Confidence scoring, Multi-model LLM racing, and Streaming Output (SSE) visually rendered on the Frontend UI.

**E. LAYER 5: MEDICATION SAFETY AUTOMATION**
*   **Purpose:** To ensure safety both inside and outside the hospital.
*   **Inside Hospital:** Dose validation, drug–drug interaction checking, allergy conflicts, duplicate therapy detection (Safety Guardian).
*   **Outside Hospital:** Automated medication reminders (IVR/SMS) via Twilio Conversational Agent, adherence tracking, and gentle patient messaging.

---

## V. DEEP DIVE: PROPOSED ARCHITECTURE COMPONENTS

### 1. Privacy Shield (`app/services/privacy_shield.py`)
*   **What it is:** A PHI removal layer placed at the very beginning of the pipeline using Presidio.
*   **What it does:** Detects names, MRN numbers, DOBs, phone numbers, and replaces them (e.g. “John Smith, 65M” → “[PATIENT], 65M”).
*   **What problem it solves:** Legal HIPAA risks and data leakage.
*   **Why it matters:** A real clinical system cannot operate without de-identification.
*   **How it solves base paper limitation:** Base paper ignores privacy; ours adds a mandatory legal safety layer making it deployable.

### 2. The LangGraph Master Orchestrator (`app/services/agent_orchestrator.py`)
*   **What it is:** The "Chief Resident" state machine that routes intents.
*   **What it does:** Automatically routes incoming clinician inputs to the appropriate specialized agent (Case Retrieval RAG, Prescription Upload, or Follow-up Setup).
*   **Why it matters:** Allows Clinsight to perform diverse healthcare tasks from a single chat window without breaking context.

### 3. Clinical Parser (NER + Negation) (`app/services/query_improvement.py`)
*   **What it is:** A deterministic structured extractor of medical features.
*   **What it does:** Extracts diseases, symptoms, labs, detects negations ("no chest pain"), and normalizes units into structured JSON.
*   **Why it matters:** If you index noise → you retrieve noise. Negated terms must be removed before retrieval.
*   **How it solves base paper limitation:** Base paper uses LLMs to refine queries but doesn't handle negation explicitly. We remove negated terms BEFORE retrieval, reducing irrelevant matches and improving precision immediately.

### 4. FAISS Dense Retrieval (`app/scripts/incremental_indexer.py`)
*   **What it is:** Vector-based semantic search engine.
*   **What it does:** Converts cases into embeddings and searches nearest vectors using cosine similarity.
*   **Why it matters:** “Heart attack” and “MI” must match despite character differences.
*   **How it solves base paper limitation:** Base paper evaluated on 50k cases. Our system scales to 250k cases on Mac hardware using memory-efficient HNSW (`all-MiniLM-L6-v2` embeddings).

### 5. FTS5 Lexical Retrieval
*   **What it is:** SQLite full-text search engine.
*   **What it does:** Finds exact term matches for rare drugs, specific pathogens, or precise lab values that Dense models miss.
*   **How it solves base paper limitation:** Base paper uses BM25 in RAM (~6GB). We use a disk-based FTS5 inverted index, meaning no memory explosion on consumer 8GB Macs.

### 6. RRF Fusion
*   **What it is:** Reciprocal Rank Fusion.
*   **What it does:** `Score = 1/(k + rank)`. Adds scores from both FAISS and FTS5 retrieval lists without manual weight tuning (`λ`).
*   **How it solves base paper limitation:** Base paper uses a fixed weighted hybrid. Our RRF is parameter-free, robust, and less sensitive to domain shift.

### 7. ColBERT Reranker (`app/services/colbert_reranker.py`) & Cross-Encoder
*   **What it is:** Precision-first reranking pipeline.
*   **What it does:** ColBERT performs token-level late-interaction matching (Top 100 -> Top 20). Cross-Encoder performs high-precision joint BERT classification (Top 20 -> Top 10).
*   **How it solves base paper limitation:** Base paper jumps directly to an expensive LLM reranker. We insert lightweight precision filters first, resulting in lower LLM costs, cleaner top candidates, and faster deployment.

### 8. Knowledge Graph Validation & Guideline RAG
*   **What it is:** Clinical consistency checker and authoritative guideline retrieval.
*   **What it does:** Checks entity overlap, detects contradictions (e.g. STEMI contraindications), and retrieves ESC/AHA guideline text to penalize mismatches.
*   **How it solves base paper limitation:** Base paper relies purely on LLM judgment. We enforce external authority grounding and deterministic validation BEFORE synthesis.

### 9. Safety Guardian & Prescription Engine (`app/services/safety_guardian.py`)
*   **What it is:** A pre-prescription drug allergy and interaction blocker.
*   **What it does:** Extracts prescriptions via `omni_parser` and validates them against patient profiles to block severe conflicts.

### 10. AI Conversational Follow-up Agent (`app/services/scheduler.py`)
*   **What it is:** Telephone support assistant using Twilio/Bland AI.
*   **What it does:** Queues follow-up calls natively in the patient's language (Telugu, Hindi) via `APScheduler` to monitor drug adherence post-discharge.

### 11. LLM Synthesis & Confidence Gate (`app/services/agent.py`)
*   **What it is:** Final reasoning generator mapping to Information Overload.
*   **What it does:** Combines Cases, Validation, and Guidelines into a structured prompt. Calculates uncertainty: `Confidence = sigmoid((top1 - top2)/T)`.
*   **How it solves base paper limitation:** Base paper output lacks explicit uncertainty scoring and structured workflow synthesis alignment.

### 12. Frontend UI Workflow (`src/app/page.tsx`)
*   **What it is:** An Apple-inspired, animated user interface built with Next.js, Framer Motion, and Tailwind squircle cards.
*   **What it does:** Connects to the backend via Server-Sent Events (SSE) to visually stream the AI's internal traces in real-time before displaying the final structured response.

---

## VI. PERFORMANCE METRICS FOR CLINSIGHT
These metrics are computed deterministically in Python against the PMC-Patients ground truth similar_patients dataset.

**Where Metrics Fit in Clinsight Pipeline:**
*   **Stage 1:** FAISS retrieval (Top 1000) -> Evaluates `Recall@100`, `Recall@1000`
*   **Stage 2:** RRF Fusion
*   **Stage 3:** ColBERT reranking (Top 50) -> Evaluates `Recall@50`, `NDCG@50`
*   **Stage 4 (Final Output):** Cross-Encoder -> Evaluates `Recall@10`, `Precision@10`, `NDCG@10`, `MAP@10`, `MRR`

**1. Definitions**
Let $Q$ = Set of query patients. For a query $q$:
*   $G(q)$ = Ground truth set of relevant patients.
*   $R_K(q)$ = Top-K retrieved patients $[d_1, d_2, \dots, d_K]$.
Relevance scoring: $rel(d_i) = 2$ (highly similar), $1$ (moderately similar), $0$ (otherwise).

**2. Recall@K**
Measures retrieval strength (how many true similar cases were successfully retrieved).
$$\text{Recall@K}(q) = \frac{|R_K(q) \cap G(q)|}{|G(q)|}$$

**3. Precision@K**
Measures Top-K quality.
$$\text{Precision@K}(q) = \frac{|R_K(q) \cap G(q)|}{K}$$

**4. DCG@K & NDCG@K (Normalized Discounted Cumulative Gain)**
Measures ranking order quality.
$$\text{DCG@K} = \Sigma_{i=1}^{K} \frac{2^{\text{rel}(d_i)} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**5. Average Precision (AP) & MAP@K**
Measures overall ranking correctness across queries.
$$\text{AP}(q) = \left(\frac{1}{|G(q)|}\right) \Sigma_{i=1}^{K} \text{Precision@i} \cdot \text{Indicator}(d_i \in G(q))$$

**6. MRR (Mean Reciprocal Rank)**
Measures how quickly a relevant case appears.
$$\text{MRR} = \left(\frac{1}{|Q|}\right) \Sigma_{q \in Q} \frac{1}{\text{rank\_first\_relevant}(q)}$$

*Insight: For research comparison, NDCG@10 is commonly reported. For clinical deployment, Recall@K is critical to avoid missing similar cases.*

---

## VII. CLINICAL SAFETY PRINCIPLES
1. Deterministic first, LLM second.
2. LLM explains, never decides safety.
3. Contraindication rules override similarity.
4. High-risk medications receive stricter logic.
5. Patient messaging must be supportive, not alarming.

---

## VIII. CORE DIRECTORY MAP
**Backend (`/backend/`)**
*   `app/main.py` - FastAPI entrypoint, maps `/api/v1/search`.
*   `app/services/agent_orchestrator.py` - The LangGraph State Orchestrator.
*   `app/services/agent.py` - Main RAG pipeline (FAISS -> ColBERT -> CrossEncoder) & LLM Synthesis.
*   `app/services/llm_manager.py` - Multi-Key LLM Fallback (handles API limits).
*   `app/services/query_improvement.py` - Clinical Parser (Symptom expansion, FTS5 normalization, Negation handling).
*   `app/services/safety_guardian.py` - Pre-prescription drug allergy and interaction blocker.
*   `app/services/scheduler.py` - Voice agent telephony queue manager.
*   `app/api/twilio.py` - Bland AI Twilio conversational hook.
*   `app/scripts/evaluate_ppr_at10.py` - Offline evaluation script generating NDCG/Recall.
*   `app/scripts/incremental_indexer.py` - The FAISS Dense Vector store database engine.

**Frontend (`/frontend/`)**
*   `src/app/page.tsx` - Main UI establishing SSE to the backend.
*   `src/components/ui/` - Contains the squircle components.

---

## IX. SETUP & EXECUTION

**1. Start the Backend server**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --port 8000
```

**2. Start the Frontend server**
```bash
cd frontend
npm run dev
```

---

## X. LONG-TERM VISION (PHASE 2)
*   **True agentic reasoning loop.**
*   **Feedback-driven model retraining:** Introduce a Thumbs Up/Down module on the Frontend. LangGraph will intercept this memory to adapt to the clinician's specific logic (Learning-to-rank using clinician clicks).
*   **Guideline validation node.**
*   **Multimodal integration (ECG, imaging).**

**Final Conclusion**
*   **Base paper:** Research innovation.
*   **Our system:** Research + deployable + safe + adaptive.