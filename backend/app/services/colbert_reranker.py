"""
colbert_reranker.py
====================
ColBERTv2-style Late Interaction Reranker for Clinsight.

Instead of compressing each document into a single vector (bi-encoder),
ColBERT keeps EVERY token's embedding and computes MaxSim:

    Score(Q, D) = Σ   max   cos_sim(q_i, d_j)
                 q_i  d_j∈D

For each query token, find the most similar document token, then sum those
maximum similarities. This captures fine-grained token-level matching that
a single-vector bi-encoder misses.

Example:
    Query:  "anterior STEMI"     → token embeddings: [anterior, STEMI]
    Doc A:  "anterior MI"        → MaxSim("STEMI", best_match) = 0.7
    Doc B:  "posterior NSTEMI"   → MaxSim("anterior", best_match) = 0.3
    → Doc A scores higher because "anterior" token matches exactly.

A regular bi-encoder would average everything into one vector and might
rank Doc B higher if "NSTEMI" is close to "STEMI" in the averaged space.

This module uses the SAME all-MiniLM-L6-v2 model already loaded by the 
indexer — no extra model download, no extra RAM. We just extract token-level
embeddings instead of the pooled [CLS] vector.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


class ColBERTReranker:
    """
    ColBERTv2-style late interaction reranker.
    
    Uses token-level embeddings + MaxSim scoring.
    Reranks a small candidate set (50 docs) — NOT for full corpus search.
    """

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        print(f"[ColBERT] Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Inference mode — no dropout
        print("[ColBERT] Late interaction reranker ready ✅")

    @torch.no_grad()
    def _get_token_embeddings(self, text: str, max_length: int = 256) -> torch.Tensor:
        """
        Returns L2-normalized token-level embeddings for the input text.
        Shape: (num_tokens, hidden_dim)  e.g. (42, 384)
        
        Unlike a bi-encoder which returns shape (1, 384) via mean-pooling,
        this returns one embedding PER token — the core of ColBERT.
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        outputs = self.model(**encoded)

        # outputs.last_hidden_state: (1, seq_len, hidden_dim)
        token_embeds = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

        # Remove [CLS] and [SEP] special tokens — they don't carry content
        # attention_mask tells us which positions are real tokens (1) vs padding (0)
        mask = encoded["attention_mask"].squeeze(0)  # (seq_len,)
        # Keep only real tokens (mask=1), skip first ([CLS]) and last ([SEP])
        real_tokens = token_embeds[1 : mask.sum() - 1]

        if real_tokens.shape[0] == 0:
            # Edge case: empty text → return the [CLS] token as fallback
            real_tokens = token_embeds[:1]

        # L2 normalize each token embedding (unit sphere)
        real_tokens = torch.nn.functional.normalize(real_tokens, p=2, dim=-1)

        return real_tokens

    def maxsim_score(self, query: str, document: str) -> float:
        """
        ColBERT MaxSim score between a query and a document.
        
        For each query token q_i:
            1. Compute cosine similarity with EVERY document token d_j
            2. Take the MAXIMUM similarity
        
        Final score = sum of all max similarities across query tokens.
        
        This is exactly the ColBERTv2 scoring formula.
        """
        q_embeds = self._get_token_embeddings(query)    # (Q, 384)
        d_embeds = self._get_token_embeddings(document)  # (D, 384)

        # Cosine similarity matrix: (Q, D)
        # Each cell [i, j] = cos_sim between query token i and doc token j
        sim_matrix = torch.matmul(q_embeds, d_embeds.T)  # (Q, D)

        # MaxSim: for each query token, take the max similarity across all doc tokens
        max_sims = sim_matrix.max(dim=1).values  # (Q,)

        # Sum of MaxSims = ColBERT score
        return float(max_sims.sum().item())

    def rerank(
        self, query: str, documents: List[str], top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidate documents using ColBERT MaxSim.
        
        Args:
            query: The search query text
            documents: List of candidate document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document_text, maxsim_score) tuples, sorted descending.
        """
        scored = []
        
        # Pre-compute query token embeddings once (reused for every document)
        q_embeds = self._get_token_embeddings(query)  # (Q, 384)

        for doc in documents:
            d_embeds = self._get_token_embeddings(doc)  # (D, 384)
            sim_matrix = torch.matmul(q_embeds, d_embeds.T)  # (Q, D)
            max_sims = sim_matrix.max(dim=1).values  # (Q,)
            score = float(max_sims.sum().item())
            scored.append((doc, score))

        # Sort descending by MaxSim score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
