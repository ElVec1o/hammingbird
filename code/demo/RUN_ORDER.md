# Suggested run order

Each preset answers one specific question. Run in this order — they get
progressively more expensive in time/memory, and the early ones validate
that the demo itself is working before you commit to the long ones.

After each preset, a JSON file lands in `logs/demo/`. To have me interpret
the results later, just point me at that folder ("read logs/demo and tell
me what's interesting") and I'll synthesize.

---

## 1. Sanity (~1 min)

**Preset:** `1. Baseline sanity (≈1 min)`
**Question:** Does the demo work end-to-end?
**Cost:** 1 min, <100 MB

All 8 methods on n=20k uniform random, k=2, d=256. The exact methods
should agree (usually 0 pairs found — that's correct on uniform random
at this size). LSH/HNSW/Annoy should show recall=1.0 too (trivially,
since there's nothing to miss).

If anything breaks here, fix it before continuing. If recall on the
approximate methods comes back < 1.0 at this size, something is off.

---

## 2. Real data (~1–2 min)

**Preset:** `2. Real CIFAR-10 pHashes (≈1 min)`
**Question:** Does pigeon's advantage hold on actual perceptual hashes?
**Cost:** 1–2 min, <500 MB
**Prereq:** `code/experiments/_real_data/phashes_cifar10_50000.bin` must
exist. If missing, run `python3 code/experiments/gen_real_phashes.py` (~30s).

Real 64-bit pHashes from CIFAR-10 train. Two k values (2 and 4). The
expected story:
- At k=2: pigeon ~10–50× over FAISS Flat, full recall.
- At k=4: lead narrows to ~2×.
- usearch/Annoy: should hit recall close to 1.0 here; compare wall-clock.

This is the most honest external-defensibility check we have.

---

## 3. Adaptive showcase (~30 sec but with big numbers)

**Preset:** `3. Adaptive on structured data (≈30 sec)`
**Question:** Does entropy-aware chunking really win on non-uniform data?
**Cost:** ~30 sec, ~500 MB (the bit/byte paths take 5–10× their usual time)

Half the bits constant. The byte and bit chunkers should produce
order-of-magnitude longer wall-clocks than usual because their chunks
falling in the zero region make every row hash to bucket 0 → all-pairs
candidate set. Adaptive should be ~3 orders of magnitude faster. **The
huge ratio is the headline number** for non-uniform / structured
signature use cases.

---

## 4. Bit-level fix at high k (~2 min)

**Preset:** `4. High-k regime: did bit-level fix the historic collapse? (≈2 min)`
**Question:** v0.1 was 5× slower than FAISS Flat at k=16. Did the bit-level
chunker (added in v0.4.0) actually fix it?
**Cost:** 2 min, <500 MB

Three k values (12, 16, 20) on n=50k uniform. Expected: bit beats byte
2-5× at k≥16, and is competitive with or beats FAISS Flat across all
three. If byte is still much slower than bit at k=20, that confirms the
v0.4 change. If FAISS Flat is now faster than bit at k=20, that
quantifies the small-k operating envelope.

---

## 5. Scaling curve (~5 min default, hours optional)

**Preset:** `5. Scaling curve at k=2, d=256 (≈5–15 min)`
**Question:** Does the gap over FAISS Flat really keep growing with n?
**Cost:** Default 3 points (100k, 500k, 1M) ≈ 5–10 min. n=2M ≈ +15 min FAISS Flat. n=5M ≈ hours.

After running the preset, the app shows a scaling curve. The known v0.4.3
result is that pigeon is roughly linear in n while FAISS Flat is O(n²)
— so the ratio should grow ~linearly with n itself.

If you want to push beyond n=1M, just keep clicking Run with new n
values (the curve auto-updates from session history). The current
records are n=2M (992×) and n=10M (extrapolated, never run live). This
is where you can settle the "yes it really scales" question with your
own eyes.

---

## 6. Approximate libraries comparison (~5 min)

**Preset:** `6. Approximate libraries on clustered data (≈5 min)`
**Question:** Does pigeon (exact) actually beat the ANN libraries
(usearch / Annoy / FAISS-LSH) on wall-clock at meaningful recall?
**Cost:** ~5 min, ~500 MB. Annoy build is the bottleneck.

Clustered dataset at n=100k. All 8 methods run. Approximate methods will
show recall < 1.0 — quantify it. The honest comparison: at recall ≥99%,
who wins on wall-clock?

If pigeon dominates: differentiator confirmed, exact + fast is a real
product position.
If usearch/HNSW beats pigeon at recall ≥99%: we need to pitch as
"exact-recall guarantee," not "fastest." Still a position, just different.

---

## 7. (legacy stress, kept for compatibility)

**Preset:** `7. Stress test — push past n=1M (open-ended)`
**Question:** Where does it break?
**Cost:** Whatever you want.

Default config is n=2M, k=2, just pigeon + FAISS Flat. Adjust on the
sidebar after picking this preset. Memory estimate updates live.

**Suggested ladder:**
1. n=2M (Flat: ~15 min, pigeon: ~1 sec) — should match the 992× v0.4.3 figure.
2. n=5M (Flat: ~90 min, pigeon: ~3 sec) — never run live before.
3. n=10M pigeon-only (Flat would be ~6 hours, skip it) — does pigeon hit ~5 sec as extrapolated?
4. n=20M pigeon-only — does anything degrade?

Stop whenever you want. The session history retains everything; the
JSON logs in `logs/demo/` preserve it across browser closes.

---

---

## Round 2 — new presets (run in this order for fresh territory)

The first seven presets established the baseline story. The six below
expand into territory we haven't tested.

### 8. Clustered scaling sweep (~15-25 min)

**Preset:** `8. Clustered data scaling — does the lead grow on real-shape data?`
**Question:** Preset 6 showed strict dominance at n=100k. Does it hold at
n=500k and n=1M, or do the ANN libraries close the gap with scale?
**Cost:** 15-25 min. FAISS Flat at n=1M is ~3 min. Annoy at n=1M is the
long pole (slow trees).

This is the most pitch-relevant new data point: "we beat every ANN
library across the entire size range, not just a cherry-picked n."

### 9. Wide hashes — 512-bit (~3-5 min)

**Preset:** `9. Wide hashes (512-bit) — does the algorithm hold up?`
**Question:** Many real workloads use wider signatures (face embeddings
at 512 bits, learned hashes at 1024). Does pigeon still dominate?
**Cost:** 3-5 min at n=200k.

Wider chunks at d=512 mean tighter filtering at low k, so we expect
pigeon's lead to *widen*, not narrow. Annoy is skipped (slow on wide
vectors); usearch is included.

### 10. Streaming Index — sub-µs query latency (~2 min) 🔥

**Preset:** `10. Streaming Index — per-query latency`
**Question:** Pigeon's Index API claims sub-µs query latency. Verify it
on YOUR hardware and compare against FAISS Flat range_search + usearch.
**Cost:** 2 min for 1000 queries per method.

DIFFERENT MODE — this preset measures per-query latency, not all-pairs.
The `extra` column will show `{median_us, mean_us, p99_us, build_s}`.
The `time_s` column is the TOTAL for all 1000 queries (so divide by 1000
for per-query mean).

Expected:
- pigeon Index: median <1 µs
- FAISS Flat: ~100 µs (1000× slower per query)
- usearch HNSW: ~10-50 µs (10-50× slower, approximate)

This is the result that justifies the "real-time content-moderation
primitive" framing.

### 11. Exact dedup (k=0) at scale (~5-10 min)

**Preset:** `11. Exact dedup (k=0) at scale`
**Question:** Most production dedup is byte-identical (k=0). How fast at
n=2M?
**Cost:** 5-10 min. FAISS Flat at n=2M k=0 is the slow one.

Validates the v0.4.1 k=0 fix works at scale. Also shows pigeon at its
peak operating regime — k=0 candidate sets are essentially zero on real
data.

### 12. ANN libraries TUNED for high recall (~10 min) 🔥

**Preset:** `12. Approximate libraries TUNED for high recall`
**Question:** Default settings give 55-78% recall — unfair pitch. What
happens when usearch/Annoy/FAISS-LSH are configured for ≥95% recall?
**Cost:** ~10 min. Annoy with 50 trees is slow but fair.

This is THE honest comparison. If the tuned ANN libraries match pigeon's
recall AND beat its wall-clock, the pitch must be "exact-recall
guarantee", not "fastest". If pigeon still wins at full recall, it's
both. Either way, you have a defensible position you can take into a
technical review without getting blindsided.

### 13. Push pigeon to n=10M (open-ended) 🔥🔥

**Preset:** `13. Push pigeon to the limit — n=5M and beyond`
**Question:** How far does pigeon scale before something gives?
**Cost:** ~30 sec at n=5M, ~2 min at n=10M. Pigeon-only — FAISS Flat at
n=10M would be ~6 hours.

We never ran above n=2M in any agent session. This is where you can
nail down "yes, the algorithm holds at 10M, 20M, beyond." Vary `n_sweep`
on the sidebar after picking the preset. Watch Activity Monitor for RSS
— that's the constraint.

---

## Updated run order (full list)

```
Round 1 — baseline story (done)
  1. sanity                ~1 min
  2. real_phash            ~1-2 min  ← re-run after MultiHash fix
  3. adaptive_structured   ~30 sec
  4. high_k_fix            ~2 min
  5. scaling_curve         ~5-15 min ← re-run for full sweep
  6. ann_libraries         ~5 min    ← re-run after MultiHash fix
  7. (legacy stress, optional)

Round 2 — new territory
  8. clustered_scaling     ~15-25 min ← clustered version of 5
  9. wide_hashes           ~3-5 min   ← d=512
  10. streaming_index      ~2 min     ← sub-µs latency claim
  11. exact_dedup          ~5-10 min  ← k=0 at scale
  12. ann_tuned            ~10 min    ← honest ANN comparison
  13. pigeon_at_the_limit  ~2-5 min   ← n=10M and beyond
  14. face_embedding_width ~5-10 min  ← d=1024

Round 3 — thoroughness round
  15. cross_join_clustered ~3-5 min   ← dedupe B against blacklist A
  16. cross_join_phash     ~1-2 min   ← real-data cross-join

Separate CLI script (NOT a Streamlit preset)
  bench_threads.py         ~3 min     ← rayon thread-count scaling
    python3 code/demo/bench_threads.py
    (defaults: n=1M, d=256, k=2, threads=1,2,4,6,8,10)
    writes a JSON to logs/demo/
```

---

## After running

Ask me:

> "Read logs/demo and tell me what the numbers say across all presets:
> does the clustered scaling hold up, does the streaming Index actually
> hit sub-µs, can tuned ANN libraries match pigeon's recall at competitive
> speed, and where does pigeon's wall hit at large n?"

I'll synthesize every JSON file into a head-to-head report you can use
for an external pitch.
