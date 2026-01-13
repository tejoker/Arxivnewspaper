#!/bin/bash
# Activates venv and runs the newspaper generator
# Usage: ./run.sh [date]

SOURCE_DIR=$(dirname "$0")
cd "$SOURCE_DIR"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

source venv/bin/activate

# Check if first valid argument is a date-like string (YYYY-MM-DD)
DATE_ARG=""
if [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    DATE_ARG="--date $1"
    shift # Remove the date argument so it's not passed twice
fi

python main.py $DATE_ARG "$@"
