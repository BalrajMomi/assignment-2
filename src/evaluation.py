from collections import defaultdict
import numpy as np
import random
from surprise import accuracy

def evaluate_predictions(predictions):
    """Computes basic error metrics"""
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    return rmse, mae

def precision_recall_ndcg_at_k_flow_2(model, rated_df, trainset, testset, k=5, threshold=3.5, num_unrated_candidates=100):
    """
    Calculates Precision@K, Recall@K, and NDCG@K using Flow #2:
    Candidates for each test user = (Books liked in Test Set) + (Random pool of Uninteracted Books)
    """
    # Get all unique book IDs in the dataset
    all_books = set(rated_df['book_id'].unique())
    
    # Group test set interactions by user
    test_user_ratings = defaultdict(list)
    for uid, iid, true_r in testset:
        test_user_ratings[uid].append((iid, true_r))
        
    # Group train set interactions by user to know what they already interacted with
    train_user_books = defaultdict(set)
    for u, i, _ in trainset.all_ratings():
        raw_uid = trainset.to_raw_uid(u)
        raw_iid = trainset.to_raw_iid(i)
        train_user_books[raw_uid].add(raw_iid)

    precisions = {}
    recalls = {}
    ndcgs = {}

    # Evaluate metrics for each user who has actual liked books in the test set
    for uid, user_test_items in test_user_ratings.items():
        # 1. Find the books in the test set that the user actually liked (rating >= threshold)
        liked_test_books = {iid for iid, r in user_test_items if r >= threshold}
        if not liked_test_books:
            continue

        # 2. Get a collection of other books the user has NOT interacted with (not in train, not in test)
        interacted_books = train_user_books[uid].union({iid for iid, _ in user_test_items})
        uninteracted_books = list(all_books - interacted_books)

        random.seed(42)
        # Sample a subset of uninteracted books to keep computation fast & standardized
        sampled_unrated = random.sample(uninteracted_books, min(len(uninteracted_books), num_unrated_candidates))
        
        # Combined candidates: liked test books + sampled uninteracted books
        candidate_pool = list(liked_test_books) + sampled_unrated
        
        # 3. Predict ratings for all these candidate books
        predictions = []
        for book_id in candidate_pool:
            est_rating = model.predict(uid, book_id).est
            predictions.append((book_id, est_rating))
            
        # 4. Sort ALL candidates by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Take the Top K
        top_k_recs = {book_id for book_id, _ in predictions[:k]}
        
        # 6. Check overlaps and compute metrics
        hits = top_k_recs.intersection(liked_test_books)
        n_rel = len(liked_test_books)
        n_rec_k = k
        n_rel_and_rec_k = len(hits)
        
        # Precision@K
        precisions[uid] = n_rel_and_rec_k / n_rec_k
        
        # Recall@K
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel > 0 else 0
        
        # NDCG@K
        dcg = 0
        for idx, (book_id, _) in enumerate(predictions[:k]):
            if book_id in liked_test_books:
                dcg += 1 / np.log2(idx + 2)
                
        idcg = sum(1 / np.log2(idx + 2) for idx in range(min(n_rel, k)))
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0

    # Average over all evaluated users
    avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0
    avg_ndcg = sum(ndcgs.values()) / len(ndcgs) if ndcgs else 0

    return avg_precision, avg_recall, avg_ndcg
