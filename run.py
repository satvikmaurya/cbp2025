#!/usr/bin/env python3

import os
import glob
import subprocess
import multiprocessing
import time
from pathlib import Path
import argparse

def run_cbp_on_trace(trace_file):
    """Run cbp executable on a single trace file."""
    # Get base filename for output naming
    trace_basename = os.path.basename(trace_file)
    output_dir = "./logs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the cbp executable
    print(f"Processing {trace_basename}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["./cbp", trace_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Write stdout to a log file
        log_file = os.path.join(output_dir, f"{trace_basename}.log")
        with open(log_file, "w") as f:
            f.write(result.stdout)
        
        elapsed = time.time() - start_time
        print(f"Completed {trace_basename} in {elapsed:.2f} seconds")
        return (trace_basename, True, elapsed)
    
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"Error processing {trace_basename}: {e}")
        # Write error output to log
        error_log = os.path.join(output_dir, f"{trace_basename}.error.log")
        with open(error_log, "w") as f:
            f.write(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
        return (trace_basename, False, elapsed)

def find_trace_files():
    """Find all trace files in the specified directories."""
    trace_dirs = [
        "../traces/compress",
        "../traces/fp",
        "../traces/int",
        "../traces/infra",
        "../traces/media",
        "../traces/web"
    ]
    
    all_traces = []
    for directory in trace_dirs:
        if os.path.exists(directory):
            # Find all *.gz files in the directory
            traces = glob.glob(os.path.join(directory, "*.gz"))
            all_traces.extend(traces)
            print(f"Found {len(traces)} traces in {directory}")
        else:
            print(f"Warning: Directory {directory} not found")
    
    return all_traces

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CBP on multiple traces using multiprocessing')
    parser.add_argument('--cores', type=int, default=multiprocessing.cpu_count(),
                        help='Number of cores to use (default: all available)')
    parser.add_argument('--max-traces', type=int, default=None,
                        help='Maximum number of traces to process (default: all)')
    args = parser.parse_args()
    
    # Find all trace files
    all_traces = find_trace_files()
    print(f"Found a total of {len(all_traces)} trace files")
    
    # Limit the number of traces if specified
    if args.max_traces:
        all_traces = all_traces[:args.max_traces]
        print(f"Limited to {len(all_traces)} traces")
    
    # Create a multiprocessing pool
    print(f"Using {args.cores} CPU cores")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(run_cbp_on_trace, all_traces)
    
    # Summarize results
    total_time = time.time() - start_time
    success_count = sum(1 for _, success, _ in results if success)
    
    print(f"\nProcessing Summary:")
    print(f"Successfully processed {success_count} out of {len(results)} traces")
    print(f"Total elapsed time: {total_time:.2f} seconds")
    
    # List failed traces
    failed = [(name, elapsed) for name, success, elapsed in results if not success]
    if failed:
        print("\nFailed traces:")
        for name, elapsed in failed:
            print(f"  - {name} (attempted for {elapsed:.2f} seconds)")

if __name__ == "__main__":
    main()