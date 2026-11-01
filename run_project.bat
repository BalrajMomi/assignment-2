@echo off
echo === Initializing Temporary Environment ===
python -m venv temp_env
call temp_env\Scripts\activate

echo === Installing Dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo === Executing Recommender System ===
python main.py

echo === Cleaning Up Environment ===
call deactivate
rd /s /q temp_env
echo === Done ===
