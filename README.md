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
├── main.py               # Orchestrator running pipeline and recommending books to test users
├── requirements.txt      # List of dependencies
├── run_project.sh        # Bash script to run on macOS/Linux
├── run_project.bat       # Command script to run on Windows
└── README.md             # This guide
```

---

## Step-by-Step Installation for Novices

To run this program, you need **Python** installed on your system. Follow these steps:

### Step 1: Install Python
1. Ensure you have **Python 3.8 or newer** installed (Python 3.10 or 3.11 is highly recommended).
2. **How to check:** Open your Terminal (macOS/Linux) or Command Prompt (Windows) and type:
   ```bash
   python --version
   ```
3. If you do not have Python installed, download it from the official site: [python.org/downloads](https://www.python.org/downloads/) and make sure to check the box that says **"Add Python to PATH"** during installation.

### Step 2: Run the Automatic Launch Scripts
We provide automatic launcher files that configure everything for you (creating a clean, isolated virtual environment, installing libraries, running the recommender, and cleaning up afterwards).

#### On macOS / Linux:
1. Open your terminal application.
2. Navigate to the project folder:
   ```bash
   cd path/to/book_recommender_project
   ```
3. Give the automation script run permission:
   ```bash
   chmod +x run_project.sh
   ```
4. Run the script:
   ```bash
   ./run_project.sh
   ```

#### On Windows:
1. Open **Command Prompt** (press `Windows Key + R`, type `cmd`, and hit Enter).
2. Navigate to the folder:
   ```cmd
   cd path\to\book_recommender_project
   ```
3. Run the batch file:
   ```cmd
   run_project.bat
   ```

---

## Manual Installation (Alternative)

If you prefer to run the steps manually instead of using the automatic helper scripts:

1. Open your terminal or command prompt inside the project folder.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate it:
   * **macOS/Linux:** `source venv/bin/activate`
   * **Windows:** `venv\Scripts\activate`
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note for Windows: If installing scikit-surprise fails, please install Visual Studio C++ Build Tools or run `conda install -c conda-forge scikit-surprise` inside an Anaconda environment.)*
5. Run the recommender:
   ```bash
   python main.py
   ```
