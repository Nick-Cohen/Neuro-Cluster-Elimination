#!/bin/bash
# Create a new prompt file in prompts/ and open it in VS Code.
# Naming: prompt-<month>-<day>-<YY>-<instance>.txt

dir="/home/cohenn1/NCE/prompts"
mkdir -p "$dir"

month=$(date +%-m)
day=$(date +%-d)
year=$(date +%y)

instance=1
while [ -f "$dir/prompt-${month}-${day}-${year}-${instance}.txt" ]; do
    ((instance++))
done

file="$dir/prompt-${month}-${day}-${year}-${instance}.txt"
touch "$file"
code "$file"
echo "$file"
