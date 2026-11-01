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
├── run_project.sh        # Bash script to run on macOS/Linux
├── run_project.bat       # Command script to run on Windows
└── README.md             # This guide
```

## Quick Run Scripts

We provide convenient automation scripts that handle virtual environment setup, package installation, program execution, and environment cleanup out-of-the-box. These guarantee that no leftover global libraries remain on your system.

### For macOS / Linux:
1. Open your terminal and navigate to the project directory.
2. Ensure the script has execute permissions:
   ```bash
   chmod +x run_project.sh
   ```
3. Run the script:
   ```bash
   ./run_project.sh
   ```

### For Windows:
1. Open Command Prompt (`cmd`) and navigate to the project directory.
2. Execute the batch file:
   ```cmd
   run_project.bat
   ```

---

## Manual Setup and Installation

If you prefer to configure and run the project manually:

1. Ensure you have Python installed (3.8+ recommended).
2. Navigate to the project root directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note for Windows users:* If building `scikit-surprise` from scratch fails on your system, please ensure you have Visual Studio C++ build tools installed, or install it pre-compiled via Anaconda:
   ```bash
   conda install -c conda-forge scikit-surprise
   ```

3. Execute the central pipeline driver manually:
   ```bash
   python main.py
   ```
