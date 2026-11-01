from collections import defaultdict
import numpy as np


def precision_recall_ndcg_at_k(predictions, k=10, threshold=3.5):
    """Calculates Precision@K, Recall@K, and NDCG@K for predictions"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    ndcgs = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        dcg = sum(1 / np.log2(idx + 2) for idx, (est, true_r) in enumerate(user_ratings[:k]) if true_r >= threshold)
        sorted_by_true = sorted(user_ratings, key=lambda x: x[1], reverse=True)
        idcg = sum(1 / np.log2(idx + 2) for idx, (est, true_r) in enumerate(sorted_by_true[:k]) if true_r >= threshold)

        ndcgs[uid] = dcg / idcg if idcg > 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    avg_ndcg = sum(ndcgs.values()) / len(ndcgs)

    return avg_precision, avg_recall, avg_ndcg
