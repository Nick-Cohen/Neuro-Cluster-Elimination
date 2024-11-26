#!/bin/bash

# Check if a file path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <file_path>"
  exit 1
fi

file_path="$1"

# Read the file and reformat the string
read -r line < "$file_path"
formatted=$(echo "#"; echo "$line" | tr ' ' '\n')

# Update the file with the new format
echo "$formatted" > "$file_path"
