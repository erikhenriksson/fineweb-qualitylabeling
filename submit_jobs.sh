#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROCESS_BATCH="$SCRIPT_DIR/process_batch.sh"

source "$SCRIPT_DIR/common.sh"

# Common has:
# ROOT_DIR, DATA_DIR, PREDICT_DIR, SAMPLE_DIR, TEST_MODE

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 SUBSET" >&2
    echo >&2
    echo "example: $0 CC-MAIN-2013-20" >&2
    exit 1
fi

SUBSET="$1"

# Add debug statements
echo "SUBSET is: $SUBSET"
echo "Testing condition..."

if [[ "$SUBSET" == "100BT" || "$SUBSET" == "350BT" || "$SUBSET" == "10BT" ]]; then
    echo "Condition matched! Should use SAMPLE_DIR"
    DATA_DIR=$SAMPLE_DIR
fi

PARQUET_DIR="$ROOT_DIR/$DATA_DIR/$SUBSET"

echo "PARQUET_DIR is: $PARQUET_DIR"

PREDICT_DIR="$ROOT_DIR/$PREDICT_DIR/$DATA_DIR/$SUBSET"
LOG_DIR="$ROOT_DIR/$LOG_DIR/$DATA_DIR/$SUBSET"

# Check if directory exists
if [ ! -d "$PARQUET_DIR" ]; then
    echo "Error: Directory $PARQUET_DIR does not exist"
    exit 1
fi

# Get list of all parquet files (just the filenames, not the full path)
cd "$PARQUET_DIR"
files=(*.parquet)

# Check if any parquet files were found
if [ ! -e "${files[0]}" ]; then
    echo "Error: No parquet files found in $PARQUET_DIR"
    exit 1
fi

mkdir -p "$PREDICT_DIR"
mkdir -p "$LOG_DIR"

total_files=${#files[@]}
batch_size=8  # Number of GPUs/files to process in parallel

# Calculate number of batches needed
num_batches=$(( (total_files + batch_size - 1) / batch_size ))

echo "Found $total_files files in $PARQUET_DIR"
echo "Will process in $num_batches batches of up to $batch_size files each"
echo "Logs will be stored in: $LOG_DIR"
echo "------------------------"

for ((batch=0; batch<num_batches; batch++)); do
    # Calculate start and end indices for this batch
    start=$((batch * batch_size))
    end=$((start + batch_size - 1))

    # Ensure we don't exceed the array bounds
    if [ $end -ge $total_files ]; then
        end=$((total_files - 1))
    fi

    # Create a comma-separated list of files for this batch
    file_list=""
    echo "Batch $batch would process:"
    for ((i=start; i<=end; i++)); do
        echo "  - ${files[i]} (would use GPU $((i-start)))"
        if [ -n "$file_list" ]; then
            file_list="${file_list},"
        fi
        file_list="${file_list}${files[i]}"
    done

    echo "Command that would be executed:"
    echo "------------------------"
    echo "sbatch process_batch.sh \"$SUBSET\" \"$file_list\""
    echo "------------------------"
    echo ""

    if [ "$TEST_MODE" = false ]; then
        sbatch  "$PROCESS_BATCH" "$SUBSET" "$file_list" "$SCRIPT_DIR"
        echo "Submitted batch $batch (files $start to $end)"
    fi
done

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: No jobs were actually submitted"
fi