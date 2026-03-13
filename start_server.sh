#!/bin/bash

# Activate virtual environment and run Flask app
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please create one first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
echo "Installing Flask dependencies..."
pip install Flask flask-cors opencv-python Pillow

# Run the Flask application
echo "Starting Flask server..."
echo "Open your browser and navigate to: http://localhost:5000"
python app.py
