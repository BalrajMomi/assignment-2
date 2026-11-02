import gzip
import json
import urllib.request
import pandas as pd


def download_and_load_data(url, local_filename, max_records=100000):
    """Downloads the compressed dataset and loads it into a Pandas DataFrame"""
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, local_filename)
    print("Download completed.")

    data = []
    print(f"Loading first {max_records} records...")
    with gzip.open(local_filename, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= max_records:
                break

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows successfully.")
    return df


def preprocess_data(df):
    """Filters out inactive interactions and prepares standard user ratings dataset"""
    rated_df = df[df['rating'] > 0][['user_id', 'book_id', 'rating', 'is_read']]
    return rated_df


def store_data_set(train_set, test_set):
    # 1. Reconstruct the training set from the surprise trainset object
    train_data_list = []
    for u, i, r in train_set.all_ratings():
        train_data_list.append({
            'user_id': train_set.to_raw_uid(u),
            'book_id': train_set.to_raw_iid(i),
            'rating': r
        })
    train_export_df = pd.DataFrame(train_data_list)

    # 2. Reconstruct the testing set from the testset object
    test_export_df = pd.DataFrame(test_set, columns=['user_id', 'book_id', 'rating'])

    # Save both to CSV files in the workspace
    train_export_df.to_csv('recommender_train_set.csv', index=False)
    test_export_df.to_csv('recommender_test_set.csv', index=False)

    print("=== Export Completed Successfully ===")
    print(f"Saved {len(train_export_df)} training records to 'recommender_train_set.csv'")
    print(f"Saved {len(test_export_df)} testing records to 'recommender_test_set.csv'")
