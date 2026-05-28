"""Rayon thread-count sweep — how does pigeon scale with core count?

Spawns a fresh subprocess per thread count with `RAYON_NUM_THREADS=N`
so each measurement uses an isolated rayon pool. This is the only
reliable way to vary rayon parallelism from Python; the rayon global
thread pool is initialized once per process and can't be resized.

Run with (from repo root):
    python3 code/demo/bench_threads.py

Optional flags:
    --n N         (default 1_000_000)    rows
    --d_bits D    (default 256)          d in bits
    --k K         (default 2)            Hamming radius
    --reps R      (default 3)            timing repetitions per thread count
    --threads T   (default 1,2,4,6,8,10) comma-separated thread counts

Output:
    logs/demo/<timestamp>_bench_threads.json
    plus a printed table.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "demo")
)
os.makedirs(LOG_DIR, exist_ok=True)


WORKER_SCRIPT = """
import time, sys, json, numpy as np
import hammingbird

n        = int(sys.argv[1])
d_bytes  = int(sys.argv[2])
k        = int(sys.argv[3])
reps     = int(sys.argv[4])

rng = np.random.default_rng(0)
A = np.ascontiguousarray(rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8))

# Warm up the rayon thread pool + caches.
hammingbird.find_pairs_self(A, k)

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    hammingbird.find_pairs_self(A, k)
    times.append(time.perf_counter() - t0)

print(json.dumps({
    "n": n, "d_bytes": d_bytes, "k": k,
    "version": hammingbird.__version__,
    "times_s": times,
}))
"""


def run_one(n: int, d_bits: int, k: int, n_threads: int, reps: int) -> dict:
    d_bytes = d_bits // 8
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(n_threads)
    py = sys.executable
    t0 = time.perf_counter()
    out = subprocess.check_output(
        [py, "-c", WORKER_SCRIPT, str(n), str(d_bytes), str(k), str(reps)],
        env=env,
    )
    total = time.perf_counter() - t0
    payload = json.loads(out.decode().strip())
    times = payload["times_s"]
    return {
        "threads": n_threads,
        "n": n,
        "d_bits": d_bits,
        "k": k,
        "reps": reps,
        "median_s": sorted(times)[len(times) // 2],
        "best_s": min(times),
        "worst_s": max(times),
        "subprocess_overhead_s": round(total - sum(times), 2),
        "pigeon_version": payload["version"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--d_bits", type=int, default=256)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=str, default="1,2,4,6,8,10")
    args = parser.parse_args()

    thread_counts = [int(x.strip()) for x in args.threads.split(",")]
    print(f"Rayon thread sweep — n={args.n}, d_bits={args.d_bits}, k={args.k}, reps={args.reps}")
    print(f"Thread counts: {thread_counts}")
    print(f"Hardware: {platform.platform()} • {platform.processor() or platform.machine()}")
    print()

    results = []
    base_median = None
    print(f"{'threads':>8}  {'median':>10}  {'best':>10}  {'worst':>10}  {'speedup':>8}  {'efficiency':>10}")
    print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
    for nt in thread_counts:
        r = run_one(args.n, args.d_bits, args.k, nt, args.reps)
        results.append(r)
        if base_median is None:
            base_median = r["median_s"]
        speedup = base_median / r["median_s"]
        efficiency = (speedup / nt) * 100
        print(
            f"{nt:>8}  {r['median_s']*1000:>8.1f}ms  "
            f"{r['best_s']*1000:>8.1f}ms  {r['worst_s']*1000:>8.1f}ms  "
            f"{speedup:>7.2f}×  {efficiency:>9.1f}%"
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset_id": "rayon_thread_sweep",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "config": {
            "n": args.n,
            "d_bits": args.d_bits,
            "k": args.k,
            "reps": args.reps,
            "thread_counts": thread_counts,
        },
        "results": results,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{stamp}_bench_threads.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Saved → {os.path.relpath(path)}")
    print()
    if results:
        best = min(results, key=lambda r: r["median_s"])
        worst = max(results, key=lambda r: r["median_s"])
        max_speedup = worst["median_s"] / best["median_s"]
        ideal = best["threads"] / 1
        print(f"Headline: {best['threads']}-thread run is {max_speedup:.2f}× faster than 1-thread.")
        print(f"Ideal linear scaling would give {ideal:.0f}×; "
              f"we're at {max_speedup / ideal * 100:.0f}% efficiency.")


if __name__ == "__main__":
    main()
