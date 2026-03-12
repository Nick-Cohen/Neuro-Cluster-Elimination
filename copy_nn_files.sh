#!/bin/bash

# Source and destination directories (hardcoded)
SOURCE_DIR="/home/cohenn1/NCE/NCE/neural_networks"
DEST_DIR="/home/cohenn1/NCE/venv/lib/python3.11/site-packages/NCE/neural_networks"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Copy all .py files from source to destination
echo "Copying .py files from '$SOURCE_DIR' to '$DEST_DIR'..."

# Use find to locate all .py files and copy them
find "$SOURCE_DIR" -name "*.py" -type f -exec cp {} "$DEST_DIR/" \;

# Check if any .py files were found
PY_COUNT=$(find "$SOURCE_DIR" -name "*.py" -type f | wc -l)

if [ $PY_COUNT -eq 0 ]; then
    echo "No .py files found in '$SOURCE_DIR'"
else
    echo "Successfully copied $PY_COUNT .py file(s) to '$DEST_DIR'"
fi

echo "Copy operation completed."