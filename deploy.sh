#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python could not be found. Please ensure Python is installed."
    exit 1
fi

echo "Python is installed."

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Set Flask application and environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask application on port 8080
flask run --host=0.0.0.0 --port=8080
