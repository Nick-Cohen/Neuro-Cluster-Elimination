#!/bin/bash

# Repository File Packager
# Packages specified files into a zip for easy analysis
# Usage: ./package_files.sh

# ========================================
# CONFIGURATION - EDIT THIS SECTION
# ========================================

# List of files to include in the package
# Add or remove files as needed - use relative or absolute paths
FILES_TO_PACKAGE=(
    "/home/cohenn1/NCE/nce/inference/bucket.py"
    "/home/cohenn1/NCE/nce/inference/factor.py"
    "/home/cohenn1/NCE/nce/inference/factor_nn.py"
    "/home/cohenn1/NCE/nce/inference/graphical_model.py"
    "/home/cohenn1/NCE/nce/data/data_loader.py"
    "/home/cohenn1/NCE/nce/data/data_preprocessor.py"
    "/home/cohenn1/NCE/nce/neural_networks/losses.py"
    "/home/cohenn1/NCE/nce/neural_networks/net.py"
    "/home/cohenn1/NCE/nce/neural_networks/simple_net.py"
    "/home/cohenn1/NCE/nce/neural_networks/train.py"
    "/home/cohenn1/NCE/nce/neural_networks/linear_mse_solver.py"
    "/home/cohenn1/NCE/nce/neural_networks/decision_tree.py"
    "/home/cohenn1/NCE/nce/sampling/sample_generator.py"
    "/home/cohenn1/NCE/notebooks/_September-2025/decision_tree_test.py"
)

# Default output folder name (will be overwritten if it exists)
OUTPUT_FOLDER="nce_package_files"

# ========================================
# SCRIPT LOGIC - NO NEED TO EDIT BELOW
# ========================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_FOLDER}"

print_status "Starting file packaging process..."
print_status "Output directory: $OUTPUT_DIR"

# Remove existing output directory if it exists
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
    print_status "Removed existing output directory"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for organizing files
TEMP_DIR=$(mktemp -d)
print_status "Created temporary directory: $TEMP_DIR"

# Track files found and missing
FOUND_FILES=()
MISSING_FILES=()

# Function to add file header comment
add_file_header() {
    local source_file="$1"
    local temp_file="$2"
    local original_path="$3"
    
    # Determine comment style based on file extension
    local comment_prefix=""
    if [[ "$source_file" == *.py ]]; then
        comment_prefix="# "
    elif [[ "$source_file" == *.ipynb ]]; then
        comment_prefix="# "
    else
        comment_prefix="# "
    fi
    
    # Create header
    echo "${comment_prefix}=================================================================================" > "$temp_file"
    echo "${comment_prefix}FILE LOCATION: $original_path" >> "$temp_file"
    echo "${comment_prefix}EXTRACTED BY: NCE Package Script" >> "$temp_file"
    if [[ "$source_file" == *.ipynb ]]; then
        echo "${comment_prefix}NOTE: Jupyter notebook converted to Python format (code cells only)" >> "$temp_file"
    fi
    echo "${comment_prefix}=================================================================================" >> "$temp_file"
    echo "" >> "$temp_file"
    
    # Handle Jupyter notebooks specially
    if [[ "$source_file" == *.ipynb ]]; then
        extract_notebook_content "$source_file" "$temp_file"
    else
        # Append original file content for regular files
        cat "$source_file" >> "$temp_file"
    fi
}

# Function to extract code and markdown from Jupyter notebooks
extract_notebook_content() {
    local notebook_file="$1"
    local output_file="$2"
    
    # Use Python to extract only cell source content
    python3 << EOF >> "$output_file"
import json

try:
    with open('$notebook_file', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cell_number = 1
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        if not source:  # Skip empty cells
            continue
            
        if cell_type == 'code':
            print(f"# ===== CODE CELL {cell_number} =====")
            if isinstance(source, list):
                for line in source:
                    # Print the line without extra newlines (strip trailing newline, print will add one)
                    print(line.rstrip('\n'))
            else:
                print(source.rstrip('\n'))
            print()  # Add blank line after cell
            
        elif cell_type == 'markdown':
            print(f"# ===== MARKDOWN CELL {cell_number} =====")
            if isinstance(source, list):
                for line in source:
                    # Convert markdown to Python comments
                    cleaned_line = line.rstrip('\n')
                    if cleaned_line.strip():  # Only print non-empty lines
                        print(f"# {cleaned_line}")
            else:
                cleaned_source = source.rstrip('\n')
                if cleaned_source.strip():
                    print(f"# {cleaned_source}")
            print()  # Add blank line after cell
        
        cell_number += 1
        
except Exception as e:
    print(f"# ERROR: Could not parse notebook - {e}")
    print(f"# File may be corrupted or in an unsupported format")
EOF
}

# Copy files to output directory with headers and flattened names
print_status "Collecting and processing files..."
for file in "${FILES_TO_PACKAGE[@]}"; do
    if [ -f "$file" ]; then
        # Create a safe filename by replacing extension for notebooks
        if [[ "$file" == *.ipynb ]]; then
            safe_filename=$(basename "$file" .ipynb).py
        else
            safe_filename=$(basename "$file")
        fi
        
        # Full path for output file
        output_file="$OUTPUT_DIR/$safe_filename"
        
        # Add header and copy content
        add_file_header "$file" "$output_file" "$file"
        
        FOUND_FILES+=("$file")
        print_status "✓ Processed: $file -> $safe_filename"
    else
        MISSING_FILES+=("$file")
        print_warning "✗ Not found: $file"
    fi
done

# Check if we found any files
if [ ${#FOUND_FILES[@]} -eq 0 ]; then
    print_error "No files found to package!"
    rm -rf "$TEMP_DIR"
    exit 1
fi

print_success "File processing complete!"
print_status "All files are ready in: $OUTPUT_DIR"

# Clean up
rm -rf "$TEMP_DIR"

# Print summary
echo
print_success "PACKAGING COMPLETE!"
echo "===================="
print_status "Files packaged: ${#FOUND_FILES[@]}"
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    print_warning "Files missing: ${#MISSING_FILES[@]}"
fi
print_status "Output directory: $OUTPUT_DIR"
print_status "Select all files in this directory for upload!"

# Show directory contents
print_status "Files ready for upload:"
ls -la "$OUTPUT_DIR"

echo
print_status "Files included in package:"
for file in "${FOUND_FILES[@]}"; do
    echo "  ✓ $file"
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo
    print_warning "Files not found (update script to remove or fix paths):"
    for file in "${MISSING_FILES[@]}"; do
        echo "  ✗ $file"
    done
fi