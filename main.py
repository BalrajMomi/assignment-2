import random
from src.data_loader import download_and_load_data, preprocess_data
from src.recommender import train_svd_model, get_top_n_recommendations, save_prediction_details
from src.evaluation import evaluate_predictions, precision_recall_ndcg_at_k_flow_2

def main():
    url = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_children.json.gz"
    local_filename = "goodreads_interactions_children.json.gz"

    # 1. Load data
    df = download_and_load_data(url, local_filename)
    rated_df = preprocess_data(df)

    # 2. Train SVD Recommender
    model, predictions, testset = train_svd_model(rated_df)

    # 3. Save Actual vs. Predicted Details
    print("\n--- Saving Prediction Details ---")
    save_prediction_details(predictions)

    # 4. Evaluate Error Metrics
    print("\n--- Model Evaluation (Error Metrics) ---")
    evaluate_predictions(predictions)

    # 5. Evaluate Ranking Metrics (Flow #2)
    print("\n--- Model Evaluation (Ranking Metrics - Flow #2) ---")
    # SVD training sets need to be accessed from our trained surprise datasets
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(rated_df[['user_id', 'book_id', 'rating']], reader)
    trainset, _ = train_test_split(surprise_data, test_size=0.2, random_state=42)
    
    random.seed(42)
    prec, rec, ndcg = precision_recall_ndcg_at_k_flow_2(
        model, rated_df, trainset, testset, k=5, threshold=3.5, num_unrated_candidates=100
    )
    print(f"Precision@5: {prec:.4f}")
    print(f"Recall@5:    {rec:.4f}")
    print(f"NDCG@5:      {ndcg:.4f}")

    # 6. Show Top 5 recommendations for 3 random test users
    test_users = list(set([pred.uid for pred in predictions]))
    random.seed(42)
    selected_users = random.sample(test_users, k=3)

    for i, user in enumerate(selected_users, 1):
        print(f"\n--- User {i}: {user} ---")
        recs = get_top_n_recommendations(user, model, rated_df, n=5)
        print("Top 5 Unread Book Recommendations:")
        print(recs.to_string(index=False))

if __name__ == '__main__':
    main()
