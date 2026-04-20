# Clinsight End-to-End Architecture

The following diagram illustrates the complete data flow of the Clinsight Multi-Agent Medical System, tracking how a user input enters the LangGraph Orchestrator and branches into either Clinical Case RAG, Prescription Safety Validation, or Telephony Follow-up.

> **Updated:** Added `Knowledge Graph Structuring` (UMLS/SNOMED ontology expansion) and `Clinical Planner LLM` (Differential Diagnosis + Routing) between Clinical Parser and Retrieval, as defined in the First Review document.

```mermaid
graph TD
    %% Styling
    classDef frontend fill:#3e3e42,stroke:#1e1e1e,stroke-width:2px,color:#fff
    classDef orchestration fill:#005A9C,stroke:#003366,stroke-width:2px,color:#fff
    classDef intent fill:#FF8C00,stroke:#B8860B,stroke-width:2px,color:#fff
    classDef rag fill:#2E8B57,stroke:#006400,stroke-width:2px,color:#fff
    classDef db fill:#8B0000,stroke:#800000,stroke-width:2px,color:#fff
    classDef voice fill:#6A5ACD,stroke:#483D8B,stroke-width:2px,color:#fff
    classDef endgoal fill:#000000,stroke:#32CD32,stroke-width:3px,color:#fff

    %% Frontend
    User([Clinician Input]) --> UI[Client UI / Server-Sent Events]
    UI --> LangGraph{LangGraph Master Orchestrator}
    
    class User,UI frontend
    class LangGraph orchestration

    %% Master Orchestrator Intents
    LangGraph -->|Routes Intent| CaseRetrieval[Case Retrieval Agent]
    LangGraph -->|Routes Intent| RxUpload[Prescription Upload Agent]
    LangGraph -->|Routes Intent| FollowUp[Followup Scheduling Agent]
    LangGraph -->|Routes Intent| Chat[General Chat Agent]
    
    class CaseRetrieval,RxUpload,FollowUp,Chat intent

    %% -------------------------------------------------------------
    %% 1. CLINICAL CASE RETRIEVAL (RAG) PIPELINE
    %% -------------------------------------------------------------
    CaseRetrieval --> Privacy[Privacy Shield / Presidio]
    Privacy --> QueryImprover[Clinical Parser: NER + Negation Removal]
    QueryImprover --> KGStruct[Knowledge Graph Structuring: UMLS + Synonym Expansion]
    KGStruct --> ClinPlanner{Clinical Planner LLM: Differential + Routing}

    ClinPlanner -->|Route: Patient Search| FAISS[(FAISS Dense: 250k S-PubMedBert)]
    ClinPlanner -->|Route: Patient Search| FTS5[(SQLite FTS5 Lexical DB)]
    ClinPlanner -->|Route: Guidelines| GuidelineRAG[Guideline RAG: ESC/AHA Protocols]

    FAISS --> RRF[Reciprocal Rank Fusion RRF]
    FTS5 --> RRF

    RRF --> ColBERT[ColBERT Reranker E1: Top 20]
    ColBERT --> CrossEncoder[Cross-Encoder E2: Top 10]
    CrossEncoder --> KGVal[KG Validation: Entity Overlap Check]
    KGVal --> ConfidenceGate{Confidence Gate: Score > 70%?}

    ConfidenceGate -->|High - Proceed| LLM[LLM Racing / Fallback Synthesis]
    ConfidenceGate -->|Low - Safety Loop| GuidelineRAG
    GuidelineRAG --> LLM

    LLM --> SSE[Stream Output to UI]

    class Privacy,QueryImprover,KGStruct,ClinPlanner,RRF,ColBERT,CrossEncoder,KGVal,ConfidenceGate,LLM,GuidelineRAG rag
    class FAISS,FTS5 db
    class SSE endgoal

    %% -------------------------------------------------------------
    %% 2. PRESCRIPTION ENGINE & SAFETY GUARDIAN
    %% -------------------------------------------------------------
    RxUpload --> Omni[OmniParser Vision Extraction]
    Omni --> Safety[Safety Guardian]
    Safety -->|Queries Patient DB| Profile[(Patient Profile DB)]
    Safety --> Rules{Allergy or Conflict?}
    Rules -->|Severe Conflict| BlockRX[Block & Warn Clinician]
    Rules -->|Safe| ConfirmRX[Confirm & Log Prescription]
    
    class Omni,Safety,Rules,BlockRX,ConfirmRX intent
    class Profile db

    %% -------------------------------------------------------------
    %% 3. MULTI-LINGUAL VOICE AGENT (FOLLOW-UP)
    %% -------------------------------------------------------------
    FollowUp --> Scheduler[APScheduler Background Task]
    ConfirmRX --> Scheduler
    Scheduler --> Twilio[Bland AI / Twilio Hook]
    Twilio --> PhoneOut(((Call Patient in Telugu/Hindi)))
    
    class Scheduler,Twilio,PhoneOut voice

```

### Key Architectural Notes:
1. **LangGraph State Orchestration:** The `LangGraph Master Orchestrator` serves as the front door. It uses an LLM node to classify the exact clinical need before the heavy processing begins.
2. **Knowledge Graph Structuring (NEW):** After the Clinical Parser, the system maps all extracted entities to UMLS/SNOMED ontologies and expands synonyms (e.g., `STEMI → Myocardial Infarction → Heart Attack`). This expanded query is fed into FAISS, drastically improving recall precision.
3. **Clinical Planner LLM (NEW):** Before retrieval begins, a dedicated LLM generates a Differential Diagnosis and decides which route to take — Patient Database Search or Guideline RAG. This makes the system *adaptive* instead of always blindly searching.
4. **Confidence Gate with Safety Loop (NEW):** The Cross-Encoder's empirical scores are used to calculate a confidence percentage. If below 70%, the system automatically fetches Official Guideline RAG before synthesis, preventing hallucinations.
5. **Retrieval Pipeline Standardization:** Stage E2 (Cross-Encoder) strictly filters the Top 10 documents into the LLM context, mapping exactly to the Base Paper's `@10` metric.
6. **Automated Safety Loop (Prescription):** If a user successfully uploads a prescription, it parses, validates against the patient's existing allergies `Profile DB` via the `Safety Guardian`, and automatically flags the `APScheduler` to configure a Twilio checkup call.
