import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

from src.data_loader import store_data_set


def train_svd_model(rated_df):
    """Loads preprocessed data into Surprise and trains an SVD model"""
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(rated_df[['user_id', 'book_id', 'rating']], reader)

    trainset, test_set = train_test_split(surprise_data, test_size=0.2, random_state=42)

    store_data_set(trainset, test_set)

    model = SVD(random_state=42)
    print("Training SVD Matrix Factorization model...")
    model.fit(trainset)

    predictions = model.test(test_set)
    return model, predictions, test_set

def get_top_n_recommendations(user_id, model, original_df, n=5):
    """Generates top N recommendations for a user containing books they have not read"""
    all_books = original_df['book_id'].unique()

    if 'is_read' in original_df.columns:
        read_books = original_df[(original_df['user_id'] == user_id) & (original_df['is_read'] == True)]['book_id'].unique()
    else:
        read_books = original_df[original_df['user_id'] == user_id]['book_id'].unique()

    unread_books = [book for book in all_books if book not in read_books]

    predictions = [model.predict(user_id, book_id) for book_id in unread_books]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]

    recommendations = pd.DataFrame([{
        'Book ID': pred.iid,
        'Estimated Rating': round(pred.est, 3)
    } for pred in top_n])

    return recommendations

def save_prediction_details(predictions, output_filename='svd_predictions_results.csv'):
    """Extracts detailed comparisons between actual and predicted ratings and saves them to a CSV"""
    predictions_df = pd.DataFrame([{
        'User ID': pred.uid,
        'Book ID': pred.iid,
        'Actual Rating': pred.r_ui,
        'Predicted Rating': round(pred.est, 3),
        'Prediction Error': round(abs(pred.r_ui - pred.est), 3)
    } for pred in predictions])

    predictions_df.to_csv(output_filename, index=False)
    print(f"Saved prediction details to '{output_filename}'.")
    return predictions_df
