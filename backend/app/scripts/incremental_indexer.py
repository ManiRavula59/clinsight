import os
import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from functools import lru_cache

# Shared DB path — used by indexer and agent.py
DB_PATH = os.environ.get("PMC_DB_PATH", "app/data/pmc_cases.db")


class IncrementalFaissIndexer:
    """
    Hybrid retrieval engine:
      - Dense path : FAISS IndexFlatIP (pritamdeka/S-PubMedBert-MS-MARCO, 768-dim)
      - Sparse path: SQLite FTS5 with BM25 ranking (zero extra RAM)

    The old BM25 pickle approach required ~6 GB RAM on load — catastrophic
    on an 8 GB M1 MacBook.  FTS5 reads directly from the SQLite file that
    already stores all 250,294 case texts, adding no memory overhead.

    Setup (one-time):
        python app/scripts/build_fts5_index.py
    """

    def __init__(
        self,
        index_path: str = "app/data/disease_index.faiss",
        db_path: str = DB_PATH,
    ):
        self.index_path = index_path
        self.db_path = db_path

        # Medical-grade PubMedBert embedding model (768-dim, matches Colab-built index)
        self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index = None
        self._fts5_ready = False  # True once build_fts5_index.py has been run

        self._load_faiss()
        self._check_fts5()

    # ── Initialisation helpers ────────────────────────────────────────────

    def _load_faiss(self):
        """Load FAISS HNSW index from disk (or create an empty one)."""
        if os.path.exists(self.index_path):
            print(f"[FAISS] Loading index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            # INCREASE HYPERPARAMETER FOR ACCURACY (Depth of Graph Search)
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = 128
            print(f"[FAISS] Loaded {self.index.ntotal:,} vectors")
        else:
            print("[FAISS] No index found — creating empty HNSW index")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 128

    def _check_fts5(self):
        """Check whether the FTS5 virtual table has been built yet."""
        if not os.path.exists(self.db_path):
            print(f"[FTS5] DB not found at {self.db_path} — FTS5 disabled")
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT COUNT(*) FROM cases_fts LIMIT 1")
            conn.close()
            self._fts5_ready = True
            print("[FTS5] cases_fts table found — BM25 layer ACTIVE ✅")
        except sqlite3.OperationalError:
            print(
                "[FTS5] cases_fts table NOT found.\n"
                "       Run:  python app/scripts/build_fts5_index.py\n"
                "       Falling back to FAISS-only until then."
            )

    # ── Dense retrieval (FAISS) ───────────────────────────────────────────

    @lru_cache(maxsize=256)
    def _encode_query(self, query: str) -> np.ndarray:
        """Cache the expensive PubMedBert encoding step separately."""
        return self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

    @lru_cache(maxsize=256)
    def search(self, query: str, top_k: int = 100) -> List[int]:
        """Dense semantic search via FAISS. Returns up to top_k integer IDs."""
        if not self.index or self.index.ntotal == 0:
            return []
        query_vec = self._encode_query(query)
        _, indices = self.index.search(query_vec, top_k)
        return [int(i) for i in indices[0] if i >= 0]

    @lru_cache(maxsize=256)
    def search_with_scores(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:
        """Dense search returning (indices, cosine_similarity_scores). Used for calibrated confidence."""
        if not self.index or self.index.ntotal == 0:
            return [], []
        query_vec = self._encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)
        valid = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        return [v[0] for v in valid], [v[1] for v in valid]

    # ── Sparse retrieval (SQLite FTS5 / BM25) ────────────────────────────

    def search_fts5(self, query: str, top_k: int = 100) -> List[int]:
        """
        Lexical BM25 search via SQLite FTS5.

        SQLite FTS5 uses BM25 as its native ranking function.
        The `rank` column directly encodes the BM25 score (more negative = better).

        Memory cost: O(1) — only the matched rows are loaded into RAM.
        Query latency: ~10–50 ms on M1 for 250k documents.

        Returns a list of case integer IDs ordered by BM25 relevance.
        """
        if not self._fts5_ready:
            return []  # Graceful fallback — FAISS-only mode

        # Sanitise the query for FTS5:
        #   - Remove punctuation that FTS5 treats as operators (* " - OR AND NOT)
        #   - Collapse whitespace
        import re
        safe_q = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_q:
            return []

        # FTS5 default is AND (all words must appear) — too strict for long clinical queries.
        # Join tokens with OR so any significant clinical term gets BM25 credit.
        # Tokens shorter than 3 chars (e.g. "a", "of") are dropped to avoid noise.
        tokens = [t for t in safe_q.split() if len(t) >= 3]
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)

        try:
            conn = sqlite3.connect(self.db_path)
            # Memory PRAGMAs for massive latency reduction
            conn.execute("PRAGMA cache_size = -64000;") # 64MB Cache
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA mmap_size = 3000000000;") # Memory map up to 3GB
            
            # FTS5 rank = BM25 score (more negative = more relevant)
            rows = conn.execute(
                """
                SELECT rowid
                FROM   cases_fts
                WHERE  cases_fts MATCH ?
                ORDER  BY rank
                LIMIT  ?
                """,
                (fts_query, top_k),
            ).fetchall()
            conn.close()
            return [int(r[0]) for r in rows]
        except sqlite3.OperationalError as e:
            print(f"[FTS5] Query error ({e}) — falling back to FAISS-only")
            return []

    # ── Legacy alias (keeps agent.py import working without change) ───────

    def search_bm25(self, query: str, top_k: int = 100) -> List[int]:
        """Alias for search_fts5 — drop-in replacement for old BM25 pickle path."""
        return self.search_fts5(query, top_k)

    # ── Index building (used by build_vector_index.py) ────────────────────

    def embed_and_add_chunk(self, texts: List[str]):
        """
        Embeds a batch of texts and appends them to the FAISS index.
        FTS5 is built separately via build_fts5_index.py (one-time migration).
        """
        print(f"[FAISS] Embedding chunk of {len(texts)} documents...")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        print(f"[FAISS] Index now contains {self.index.ntotal:,} vectors")


if __name__ == "__main__":
    idx = IncrementalFaissIndexer()
    print("\n--- Dense FAISS test ---")
    print(idx.search("elderly patient chest pain STEMI myocardial infarction", top_k=5))
    print("\n--- FTS5 BM25 test ---")
    print(idx.search_fts5("chest pain myocardial infarction STEMI", top_k=5))
