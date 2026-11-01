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
