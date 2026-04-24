import glob
import sys
import time
import tracemalloc

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from patent.modeling.iforest.train import evaluate_params
from patent.utils import get_vectors_from_files


def format_mb(bytes_size):
    return f"{bytes_size / (1024 * 1024):.2f} MB"


def profile_single_run(file_paths):
    # Start tracking memory allocations
    tracemalloc.start()
    total_start_time = time.perf_counter()

    # 1. Profile Loading Embeddings
    print(f"--- 1. Loading Embeddings from {len(file_paths)} files ---")
    step_start = time.perf_counter()

    X_train = get_vectors_from_files(file_paths)

    step_time = time.perf_counter() - step_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"Time Taken:  {step_time:.2f}s")
    print(f"Current Mem: {format_mb(current_mem)}")
    print(f"Peak Mem:    {format_mb(peak_mem)}\n")

    if len(X_train) == 0:
        print("No embeddings loaded. Exiting before model fitting.")
        tracemalloc.stop()
        return

    # 2. Profile Model Fitting
    print("--- 2. Fitting Isolation Forest ---")
    step_start = time.perf_counter()

    # Using baseline params for test
    model = IsolationForest(max_samples=256, n_estimators=100, random_state=42)
    model.fit(X_train)

    step_time = time.perf_counter() - step_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"Time Taken:  {step_time:.2f}s")
    print(f"Current Mem: {format_mb(current_mem)}")
    print(f"Peak Mem:    {format_mb(peak_mem)}\n")

    # 3. Profile Calculating Stability
    print("--- 3. Calculating Stability ---")
    step_start = time.perf_counter()

    X_train_split, X_val_split = train_test_split(X_train, test_size=0.02, random_state=42, shuffle=True)
    baseline_params = {"max_samples": 256, "n_estimators": 100, "contamination": 0.08}
    stability_score = evaluate_params(baseline_params, X_train_split, X_val_split)

    step_time = time.perf_counter() - step_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"Stability Score: {stability_score:.6f}")
    print(f"Time Taken:  {step_time:.2f}s")
    print(f"Current Mem: {format_mb(current_mem)}")
    print(f"Peak Mem:    {format_mb(peak_mem)}\n")

    # Clean up
    tracemalloc.stop()
    print(f"Total Time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    # If the user passed arguments, use them as file paths
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    else:
        # Otherwise, try finding processed parquet files by default
        test_files = glob.glob("data/processed/*.parquet")

    if not test_files:
        print("No Parquet files provided or found in data/processed/")
        print("Usage: python profile_run.py <path_to_parquet_file1> <path_to_parquet_file2>")
        sys.exit(1)

    profile_single_run(test_files)
