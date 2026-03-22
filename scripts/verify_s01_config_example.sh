#!/bin/bash
# Verification script for S01: Complete Config Example in BENCHMARK_USAGE.md
# Extracts YAML from BENCHMARK_USAGE.md, runs benchmark on one hard bucket, validates outputs

set -e  # Exit on error

# Configuration
TEMP_CONFIG="/tmp/benchmark_config_verify.yaml"
TEMP_BUCKET_LIST="/tmp/benchmark_single_bucket.json"
BENCHMARK_USAGE="BENCHMARK_USAGE.md"
NCE_DIR="/home/cohenn1/NCE"
BUCKET_LIST="${NCE_DIR}/data/hard_buckets/bucket_list.json"
OUTPUT_BASE="${NCE_DIR}/data/benchmark_output"
BUCKET_LIST_BACKUP="${BUCKET_LIST}.backup_verify"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0

# Cleanup function to restore bucket_list.json on exit
cleanup() {
    if [ -f "$BUCKET_LIST_BACKUP" ]; then
        mv "$BUCKET_LIST_BACKUP" "$BUCKET_LIST"
        check_info "Restored original bucket_list.json"
    fi
    if [ -f "$TEMP_BUCKET_LIST" ]; then
        rm -f "$TEMP_BUCKET_LIST"
    fi
    if [ -f "$TEMP_CONFIG" ]; then
        rm -f "$TEMP_CONFIG"
    fi
}

trap cleanup EXIT

# Helper function to print check results
check_pass() {
    echo -e "${GREEN}PASS:${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}FAIL:${NC} $1 — $2"
    ((CHECKS_FAILED++))
}

check_info() {
    echo -e "${YELLOW}INFO:${NC} $1"
}

echo "================================================="
echo "S01 Config Example Verification"
echo "================================================="
echo ""

# Step 1: Extract YAML from BENCHMARK_USAGE.md
check_info "Step 1: Extracting YAML from BENCHMARK_USAGE.md"

# Extract the first YAML code block (between ```yaml and ```)
awk '/^```yaml$/,/^```$/ {if (!/^```/) print}' "$BENCHMARK_USAGE" > "$TEMP_CONFIG"

YAML_LINES=$(wc -l < "$TEMP_CONFIG")
if [ "$YAML_LINES" -gt 0 ]; then
    check_pass "YAML extraction (${YAML_LINES} lines extracted)"
    check_info "Extracted YAML written to: ${TEMP_CONFIG}"
else
    check_fail "YAML extraction" "No YAML content found in ${BENCHMARK_USAGE}"
    exit 1
fi

echo ""

# Step 2: Validate YAML syntax
check_info "Step 2: Validating YAML syntax"

python3 << 'EOF'
import yaml
import sys

try:
    with open('/tmp/benchmark_config_verify.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for required top-level sections
    required_sections = ['inference', 'nn', 'training', 'sampling', 'backward', 'output']
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            sys.exit(1)
    
    print("YAML_VALID")
    sys.exit(0)
except yaml.YAMLError as e:
    print(f"YAML parsing error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Validation error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    check_pass "YAML syntax validation"
else
    check_fail "YAML syntax validation" "See error above"
    exit 1
fi

echo ""

# Step 3: Get first bucket from bucket_list.json
check_info "Step 3: Selecting first bucket from bucket_list.json"

FIRST_BUCKET=$(python3 << EOF
import json
with open('${BUCKET_LIST}', 'r') as f:
    buckets = json.load(f)
if len(buckets) == 0:
    print("ERROR: No buckets in bucket_list.json")
    exit(1)
print(buckets[0]['bucket_id'])
EOF
)

if [ -n "$FIRST_BUCKET" ]; then
    check_pass "Bucket selection (selected: ${FIRST_BUCKET})"
else
    check_fail "Bucket selection" "Failed to read bucket from ${BUCKET_LIST}"
    exit 1
fi

echo ""

# Step 4: Create a temporary modified config that only processes one bucket
check_info "Step 4: Creating single-bucket config"

# Create a temporary bucket_list.json with only the first bucket
TEMP_BUCKET_LIST="/tmp/benchmark_single_bucket.json"
python3 << EOF
import json
with open('${BUCKET_LIST}', 'r') as f:
    buckets = json.load(f)
with open('${TEMP_BUCKET_LIST}', 'w') as f:
    json.dump([buckets[0]], f, indent=2)
EOF

check_pass "Single-bucket config created"

echo ""

# Step 5: Run benchmark
check_info "Step 5: Running benchmark on single bucket"

cd "$NCE_DIR"

# Since bucket_benchmark.py doesn't have --max-buckets flag, we temporarily
# replace bucket_list.json with our single-bucket version
BUCKET_LIST_BACKUP="${BUCKET_LIST}.backup_verify"
cp "$BUCKET_LIST" "$BUCKET_LIST_BACKUP"
cp "$TEMP_BUCKET_LIST" "$BUCKET_LIST"

check_info "Command: cd ${NCE_DIR} && python scripts/bucket_benchmark.py ${TEMP_CONFIG} fast --gpus 0 --output-dir ${OUTPUT_BASE}/verify_s01"

# Run benchmark (capture output for parsing)
BENCHMARK_OUTPUT=$(python scripts/bucket_benchmark.py "$TEMP_CONFIG" fast --gpus 0 --output-dir "${OUTPUT_BASE}/verify_s01" 2>&1)
BENCHMARK_EXIT=$?

# Restore original bucket_list.json (also happens in trap, but do it here for clarity)
mv "$BUCKET_LIST_BACKUP" "$BUCKET_LIST"
check_info "Restored original bucket_list.json"

echo "$BENCHMARK_OUTPUT"

if [ $BENCHMARK_EXIT -eq 0 ]; then
    check_pass "Benchmark execution (exit code: 0)"
else
    check_fail "Benchmark execution" "Exit code: ${BENCHMARK_EXIT}"
fi

echo ""

# Step 6: Validate output artifacts
check_info "Step 6: Validating output artifacts"

# Find the output directory (should be under OUTPUT_BASE/verify_s01/)
OUTPUT_DIR="${OUTPUT_BASE}/verify_s01/${FIRST_BUCKET}"

check_info "Expected output directory: ${OUTPUT_DIR}"

if [ -d "$OUTPUT_DIR" ]; then
    check_pass "Output directory exists"
else
    check_fail "Output directory existence" "Directory not found: ${OUTPUT_DIR}"
    echo ""
    echo "================================================="
    echo "Summary: ${CHECKS_PASSED} passed, ${CHECKS_FAILED} failed"
    echo "================================================="
    exit 1
fi

# Check loss.png
if [ -f "${OUTPUT_DIR}/loss.png" ]; then
    FILE_SIZE=$(stat -c%s "${OUTPUT_DIR}/loss.png" 2>/dev/null || stat -f%z "${OUTPUT_DIR}/loss.png" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 0 ]; then
        check_pass "loss.png exists and has content (${FILE_SIZE} bytes)"
    else
        check_fail "loss.png size" "File is empty"
    fi
else
    check_fail "loss.png existence" "File not found at ${OUTPUT_DIR}/loss.png"
fi

# Check local_error.png
if [ -f "${OUTPUT_DIR}/local_error.png" ]; then
    FILE_SIZE=$(stat -c%s "${OUTPUT_DIR}/local_error.png" 2>/dev/null || stat -f%z "${OUTPUT_DIR}/local_error.png" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 0 ]; then
        check_pass "local_error.png exists and has content (${FILE_SIZE} bytes)"
    else
        check_fail "local_error.png size" "File is empty"
    fi
else
    check_fail "local_error.png existence" "File not found at ${OUTPUT_DIR}/local_error.png"
fi

# Check metrics.json
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    FILE_SIZE=$(stat -c%s "${OUTPUT_DIR}/metrics.json" 2>/dev/null || stat -f%z "${OUTPUT_DIR}/metrics.json" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 0 ]; then
        check_pass "metrics.json exists and has content (${FILE_SIZE} bytes)"
    else
        check_fail "metrics.json size" "File is empty"
    fi
else
    check_fail "metrics.json existence" "File not found at ${OUTPUT_DIR}/metrics.json"
fi

echo ""

# Step 7: Validate metrics.json schema
check_info "Step 7: Validating metrics.json schema"

python3 << EOF
import json
import sys

try:
    with open('${OUTPUT_DIR}/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Check for required keys
    required_keys = ['epochs_completed', 'final_loss', 'final_local_error']
    missing_keys = [key for key in required_keys if key not in metrics]
    
    if missing_keys:
        print(f"Missing required keys: {', '.join(missing_keys)}")
        sys.exit(1)
    
    print(f"epochs_completed: {metrics['epochs_completed']}")
    print(f"final_loss: {metrics['final_loss']}")
    print(f"final_local_error: {metrics['final_local_error']}")
    
    sys.exit(0)
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Validation error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    check_pass "metrics.json schema validation"
else
    check_fail "metrics.json schema validation" "See error above"
fi

echo ""

# Step 8: Summary
check_info "Step 8: Cleanup complete (via trap handler)"

echo ""
echo "================================================="
echo "Summary: ${CHECKS_PASSED} passed, ${CHECKS_FAILED} failed"
echo "================================================="

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed.${NC}"
    exit 1
fi
