import math
from typing import List, Dict

# ---------------------------------------------------------------------
# IR Evaluation Metrics for Patient-Patient Retrieval (PPR)
# Based on exact specification from the user's thesis/guide.
# ---------------------------------------------------------------------

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """
    Recall@K = |R_K(q) ∩ G(q)| / |G(q)|
    Did we retrieve the relevant ones at all?
    """
    if not ground_truth_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant_set = set(ground_truth_ids)
    
    hits = len(top_k.intersection(relevant_set))
    return hits / len(relevant_set)

def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """
    Precision@K = |R_K(q) ∩ G(q)| / K
    Of what we retrieved, how many are correct?
    """
    if k == 0:
        return 0.0
        
    top_k = set(retrieved_ids[:k])
    relevant_set = set(ground_truth_ids)
    
    hits = len(top_k.intersection(relevant_set))
    return hits / k

def calculate_dcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, int], k: int) -> float:
    """
    DCG@K = Σ ( (2^rel_i - 1) / log2(i + 1) )
    Where rel_i is 0, 1, or 2 depending on retrieved doc relevance (grade).
    """
    dcg = 0.0
    top_k = retrieved_ids[:k]
    
    for i, doc_id in enumerate(top_k):
        rel = relevance_map.get(doc_id, 0)
        # i is 0-indexed, so rank is i + 1. The formula denominator is log2(rank + 1) = log2((i+1) + 1) = log2(i+2)
        # Wait, standard formula: log2(rank + 1), where rank starts at 1. So rank=1 -> denominator log2(2) = 1.
        # So rank = i + 1, denominator = log2(i + 1 + 1) = log2(i + 2)
        dcg += (2**rel - 1) / math.log2((i + 1) + 1)
        
    return dcg

def calculate_ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, int], k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    The gold metric for graded relevance (1 vs 2).
    """
    dcg = calculate_dcg_at_k(retrieved_ids, relevance_map, k)
    
    # Calculate IDCG by sorting all ground truth elements by relevance descending
    sorted_ideal = sorted(relevance_map.keys(), key=lambda x: relevance_map[x], reverse=True)
    idcg = calculate_dcg_at_k(sorted_ideal, relevance_map, k)
    
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_ap_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """
    AP@K = (1 / |G(q)|) * Σ (Precision@i * Indicator(doc_i ∈ G(q)))
    """
    if not ground_truth_ids:
        return 0.0

    relevant_set = set(ground_truth_ids)
    top_k = retrieved_ids[:k]
    
    ap_sum = 0.0
    hits = 0
    
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            ap_sum += precision_at_i
            
    return ap_sum / len(relevant_set)

def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    MRR = 1 / rank_first_relevant(q)
    """
    relevant_set = set(ground_truth_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

# ---------------------------------------------------------------------
# Example Runner for a Single Query
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Test based on the user's practical example:
    # 3 patients with score 2
    # 7 patients with score 1
    # Total relevant = 10
    
    ground_truth_relevance = {
        "p_score2_a": 2, "p_score2_b": 2, "p_score2_c": 2,
        "p_score1_1": 1, "p_score1_2": 1, "p_score1_3": 1, "p_score1_4": 1, 
        "p_score1_5": 1, "p_score1_6": 1, "p_score1_7": 1
    }
    ground_truth_ids = list(ground_truth_relevance.keys())
    
    # Your top-10 results contain: 2 from score2, 3 from score1, and 5 irrelevant ones
    # Mixed ranking exactly as described in the user prompt example
    mock_retrieved_list = [
        "p_score2_a",   # Hit (score 2) - rank 1
        "irr_1",        # Miss
        "p_score1_1",   # Hit (score 1) - rank 3
        "irr_2",        # Miss
        "p_score2_b",   # Hit (score 2) - rank 5
        "irr_3",        # Miss
        "p_score1_2",   # Hit (score 1) - rank 7
        "p_score1_3",   # Hit (score 1) - rank 8
        "irr_4",        # Miss
        "irr_5"         # Miss
    ]
    
    k = 10
    recall_10 = calculate_recall_at_k(mock_retrieved_list, ground_truth_ids, k)
    precision_10 = calculate_precision_at_k(mock_retrieved_list, ground_truth_ids, k)
    ndcg_10 = calculate_ndcg_at_k(mock_retrieved_list, ground_truth_relevance, k)
    map_10 = calculate_ap_at_k(mock_retrieved_list, ground_truth_ids, k)
    mrr = calculate_mrr(mock_retrieved_list, ground_truth_ids)
    
    print("--- Clinsight Patient-Patient Retrieval (PPR) Metrics ---")
    print(f"Recall@{k}:    {recall_10:.2f}  (Expected ~0.50 based on example)")
    print(f"Precision@{k}: {precision_10:.2f}  (Expected ~0.50 based on example)")
    print(f"NDCG@{k}:      {ndcg_10:.4f}")
    print(f"MAP@{k}:       {map_10:.4f}")
    print(f"MRR:         {mrr:.4f}  (First hit is at rank 1, so 1.0)")
