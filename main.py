import random
from src.data_loader import download_and_load_data, preprocess_data
from src.recommender import train_svd_model, get_top_n_recommendations
from src.evaluation import evaluate_predictions, precision_recall_ndcg_at_k

SOURCE_DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_children.json.gz"
LOCAL_FILE_NAME = "temp/goodreads_interactions_children.json.gz"

def main():
    # 1. Load data
    df = download_and_load_data(SOURCE_DATA_URL, LOCAL_FILE_NAME)
    rated_df = preprocess_data(df)
    
    # 2. Train SVD Recommender
    model, predictions, test_set = train_svd_model(rated_df)
    
    # 3. Evaluate Error Metrics
    print("\n--- Model Evaluation (Error Metrics) ---")
    evaluate_predictions(predictions)
    
    # 4. Evaluate Ranking Metrics
    print("\n--- Model Evaluation (Ranking Metrics) ---")
    prec, rec, ndcg = precision_recall_ndcg_at_k(predictions, k=5, threshold=3.5)
    print(f"Precision@5: {prec:.4f}")
    print(f"Recall@5:    {rec:.4f}")
    print(f"NDCG@5:      {ndcg:.4f}")
    
    # 5. Show Top 5 recommendations for 3 random test users (not yet read)
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
