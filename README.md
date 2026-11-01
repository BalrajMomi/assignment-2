# Book Recommender Project

This is a modularized Python project featuring a Collaborative Filtering Book Recommender System built using Singular Value Decomposition (SVD) on a Goodreads ratings subset.

## Project Structure
```
book_recommender_project/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Downloads and prepares Goodreads JSON ratings data
│   ├── recommender.py    # Builds SVD matrix factorization and unread recommendation filtering
│   └── evaluation.py     # Calculates predictive error and ranking metrics (Precision, Recall, NDCG)
│
├── main.py               # Orchestrator running pipeline and recommending books to 3 test users
├── requirements.txt      # List of dependencies
└── README.md             # This guide
```

## Setup and Installation

1. Ensure you have Python installed (3.8+ recommended).
2. Navigate to the project root directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note for Windows users:* If building `scikit-surprise` from scratch fails on your system, please ensure you have Visual Studio C++ build tools installed, or install it pre-compiled via Anaconda:
   ```bash
   conda install -c conda-forge scikit-surprise
   ```

## How to Run
Execute the central pipeline driver:
```bash
python main.py
