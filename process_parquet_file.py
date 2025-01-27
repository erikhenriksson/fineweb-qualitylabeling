import time

import joblib
import pandas as pd
import pyarrow.parquet as pq
import torch
from fastparquet import write
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

# Set up model for speed and precision
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def process_chunk(chunk, model, tokenizer, platt_scaler, target_class_index=8):
    text_batch = [item["text"] for item in chunk]
    text_ids = [item["id"] for item in chunk]

    all_lines_with_idx = []
    for doc_idx, text in enumerate(text_batch):
        for line_idx, line in enumerate(text.splitlines()):
            all_lines_with_idx.append((len(line.split()), doc_idx, line_idx, line))

    # Sort lines by their length for efficient padding
    all_lines_with_idx.sort(key=lambda x: x[0])

    # Process each group of lines by length
    all_scaled_probs = [None] * len(all_lines_with_idx)
    for i in range(0, len(all_lines_with_idx), 128):
        batch = all_lines_with_idx[i : i + 128]
        line_batch = [x[3] for x in batch]

        inputs = tokenizer(
            line_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()

        positive_logits = logits[:, target_class_index]
        scaled_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[:, 1]

        for j, (_, doc_idx, line_idx, _) in enumerate(batch):
            all_scaled_probs[i + j] = (doc_idx, line_idx, round(scaled_probs[j], 4))

    # Organize scaled probabilities back into the structure of the original text_batch
    doc_scaled_probs = [[] for _ in text_batch]
    for doc_idx, line_idx, prob in sorted(all_scaled_probs, key=lambda x: (x[0], x[1])):
        doc_scaled_probs[doc_idx].append(prob)

    # Create results while preserving ALL original data
    results = []
    for doc_idx, original_doc in enumerate(chunk):
        # Create a new dict with all original data
        augmented_doc = original_doc.copy()
        # Add our new field
        augmented_doc["line_quality"] = doc_scaled_probs[doc_idx]
        results.append(augmented_doc)

    return results


def process_large_file(input_file, chunk_size):
    """Reads a large parquet file in chunks efficiently."""

    pf = pq.ParquetFile(input_file)

    # Read in our specified chunk sizes
    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        # Convert chunk to list of dicts
        chunk = []
        for _, row in df.iterrows():
            document = row.to_dict()
            chunk.append(document)

        yield chunk
        del df


def write_incremental_parquet(results, output_file, first_write=False):
    """Write results incrementally to parquet file, preserving all data types."""
    df = pd.DataFrame(results)

    if first_write:
        df.to_parquet(output_file, index=False, engine="fastparquet")
    else:
        write(output_file, df, append=True, file_scheme="simple")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "TurkuNLP/finerweb-quality-classifier", num_labels=9
    )
    scaler_path = hf_hub_download(
        repo_id="TurkuNLP/finerweb-quality-classifier", filename="platt_scaler.joblib"
    )
    platt_scaler = joblib.load(scaler_path)
    model.to(device)
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True,
        backend="inductor",
    )
    model.eval()

    total_items = 0
    total_time = 0.0
    first_write = True

    for chunk_idx, chunk in enumerate(process_large_file(args.input_file, 10000)):
        start_time = time.perf_counter()

        results = process_chunk(chunk, model, tokenizer, platt_scaler)

        # Write results immediately instead of storing them
        write_incremental_parquet(results, args.output_file, first_write)
        first_write = False

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        throughput = len(chunk) / elapsed_time if elapsed_time > 0 else float("inf")

        total_items += len(chunk)
        total_time += elapsed_time
        average_throughput = (
            total_items / total_time if total_time > 0 else float("inf")
        )

        if chunk_idx % 100 == 0:
            print(
                f"Chunk {chunk_idx}: Throughput = {throughput:.2f} items/s, "
                f"Average Throughput = {average_throughput:.2f} items/s"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input parquet file.")
    parser.add_argument(
        "output_file", type=str, help="Path to the output parquet file."
    )
    args = parser.parse_args()
    main(args)
