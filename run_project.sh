#!/bin/bash

echo "=== Initializing Temporary Environment ==="
python3 -m venv temp_env
source temp_env/bin/activate

echo "=== Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Executing Recommender System ==="
python main.py

echo "=== Cleaning Up Environment ==="
deactivate
rm -rf temp_env
echo "=== Done ==="
