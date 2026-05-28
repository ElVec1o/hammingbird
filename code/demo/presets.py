"""Pre-defined benchmark combos for the pigeon demo.

Each preset is a focused answer to ONE question. Run them in the order
listed in `RUN_ORDER.md` for the most useful narrative.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Canonical method names (must match those in methods.py).
PIGEON_BYTE     = "pigeon — find_pairs_self (byte-aligned)"
PIGEON_BIT      = "pigeon — find_pairs_self_bit (bit-aligned)"
PIGEON_ADAPTIVE = "pigeon — find_pairs_self_adaptive (entropy)"
FAISS_FLAT      = "FAISS IndexBinaryFlat (brute force, exact)"
FAISS_MH        = "FAISS IndexBinaryMultiHash (exact, nflip=0)"
FAISS_HASH      = "FAISS IndexBinaryHash (LSH, approximate)"
USEARCH         = "usearch HNSW (binary, approximate)"
ANNOY           = "Annoy (random projection, approximate)"
# Cross-join variants
PIGEON_CROSS_BYTE = "pigeon — find_pairs_cross (byte)"
PIGEON_CROSS_BIT  = "pigeon — find_pairs_cross_bit (bit)"
FAISS_FLAT_CROSS  = "FAISS IndexBinaryFlat (cross)"
FAISS_MH_CROSS    = "FAISS IndexBinaryMultiHash (cross, nflip=0)"
USEARCH_CROSS     = "usearch HNSW (cross, approximate)"

ALL_EXACT  = [PIGEON_BYTE, PIGEON_BIT, PIGEON_ADAPTIVE, FAISS_FLAT, FAISS_MH]
ALL_APPROX = [FAISS_HASH, USEARCH, ANNOY]
ALL_METHODS = ALL_EXACT + ALL_APPROX


@dataclass
class Preset:
    """A reproducible benchmark configuration."""
    id: str                              # short label for the log filename
    label: str                           # display name in the UI
    question: str                        # one-line "what does this answer"
    dataset: str                         # key into datasets.DATASETS
    n: int
    d_bits: int                          # display value; d_bytes = d_bits // 8
    k_values: list[int]                  # sweep one or more k values
    methods: list[str]
    rough_runtime: str                   # for the UI
    notes: str = ""                      # extra interpretation guidance
    # When non-None, multiple n values to sweep (replaces the single n above).
    n_sweep: Optional[list[int]] = None


PRESETS: list[Preset] = [
    Preset(
        id="01_sanity",
        label="1. Baseline sanity (≈1 min)",
        question="Does the demo work? Do exact methods agree and approximate ones show realistic recall?",
        dataset="Synthetic — uniform random",
        n=20_000,
        d_bits=256,
        k_values=[2],
        methods=ALL_METHODS,
        rough_runtime="~1 minute (mostly the Annoy build)",
        notes=(
            "All five exact methods should return the same pair count "
            "(usually 0 on this uniform random data). LSH/HNSW/Annoy should "
            "show recall=1.0 too because true pairs are rare. This is the "
            "'does the harness work' check before the real experiments."
        ),
    ),
    Preset(
        id="02_real_phash",
        label="2. Real CIFAR-10 pHashes (≈1 min)",
        question="Does pigeon's lead hold on real 64-bit perceptual hashes from a public dataset?",
        dataset="Real — CIFAR-10 pHashes (64-bit, fixed)",
        n=50_000,
        d_bits=64,
        k_values=[2, 4],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH, USEARCH, ANNOY],
        rough_runtime="~1–2 minutes",
        notes=(
            "Real pHashes from CIFAR-10. We skip k=8 (= d_bytes) because the "
            "byte-aligned pigeon path would refuse and the comparison is "
            "narrower at high k/d ratio. Expect pigeon to dominate at k=2 "
            "(~10–50× over FAISS Flat) and the lead to narrow at k=4."
        ),
    ),
    Preset(
        id="03_adaptive_structured",
        label="3. Adaptive on structured data (≈30 sec)",
        question="Does entropy-aware chunking really win when bits are non-uniform?",
        dataset="Synthetic — low-entropy (adversarial)",
        n=20_000,
        d_bits=256,
        k_values=[4],
        methods=[PIGEON_BYTE, PIGEON_BIT, PIGEON_ADAPTIVE, FAISS_FLAT],
        rough_runtime="~30 seconds, BUT pigeon_bit/byte may take 5–10× longer than usual here",
        notes=(
            "Half the bits constant. The naive (byte/bit) chunkers will "
            "degenerate to all-pairs candidate sets — exactly the pathological "
            "case the adaptive planner was designed for. Expect adaptive to "
            "beat the naive ones by 2–3 orders of magnitude. Compare to FAISS "
            "Flat as the universal baseline."
        ),
    ),
    Preset(
        id="04_high_k_fix",
        label="4. High-k regime: did bit-level fix the historic collapse? (≈2 min)",
        question="At k≥16 the v0.1 algorithm was 5× SLOWER than FAISS Flat. Did bit-level chunks fix it?",
        dataset="Synthetic — uniform random",
        n=50_000,
        d_bits=256,
        k_values=[12, 16, 20],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT],
        rough_runtime="~2 minutes",
        notes=(
            "At k≥16 the byte path produces a huge candidate set; the bit path "
            "uses finer chunks. Per round 5: bit was 2.5× faster than byte at "
            "k=16 and 5.2× faster at k=20. We want to confirm and also see "
            "where each crosses FAISS Flat."
        ),
    ),
    Preset(
        id="05_scaling_curve",
        label="5. Scaling curve at k=2, d=256 (≈5–15 min depending on max n)",
        question="Does pigeon's lead over FAISS Flat really grow with n?",
        dataset="Synthetic — uniform random",
        n=100_000,
        n_sweep=[100_000, 500_000, 1_000_000],
        d_bits=256,
        k_values=[2],
        methods=[PIGEON_BYTE, FAISS_FLAT, FAISS_MH],
        rough_runtime="~5 min at n=1M (FAISS Flat is the slow one); add ~30 min for n=2M, multiple hours for n=10M",
        notes=(
            "This is THE headline. Pigeon should be roughly linear in n while "
            "FAISS Flat is O(n²). The ratio grows. If you have RAM and "
            "patience, push to n=2M (≈ 15 min FAISS Flat) or n=5M (≈90 min)."
        ),
    ),
    Preset(
        id="06_ann_libraries_clustered",
        label="6. Approximate libraries on clustered data (≈5 min)",
        question="How does pigeon (exact) stack up against usearch/Annoy/FAISS-LSH (approximate) on a realistic clustered workload?",
        dataset="Synthetic — clustered (real-world shape)",
        n=100_000,
        d_bits=256,
        k_values=[4],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH, FAISS_HASH, USEARCH, ANNOY],
        rough_runtime="~5 minutes (Annoy build is the slowest piece)",
        notes=(
            "The honest comparison vs the ANN world. Approximate methods will "
            "show recall < 1.0. The question: do they actually beat pigeon's "
            "wall-clock at meaningful recall (≥99%)? If they do, our pitch "
            "needs to qualify; if they don't, pigeon's exact + fast is a real "
            "differentiator."
        ),
    ),
    Preset(
        id="07_stress",
        label="7. Stress test — push past n=1M (open-ended)",
        question="At what n does the speedup curve plateau or break?",
        dataset="Synthetic — uniform random",
        n=2_000_000,
        d_bits=256,
        k_values=[2],
        methods=[PIGEON_BYTE, FAISS_FLAT],  # FAISS Flat takes ~15 min at n=2M
        rough_runtime="~15 min at n=2M; jumps fast — n=10M FAISS Flat is hours",
        notes=(
            "Open-ended: vary n on the sidebar after picking this preset. "
            "We could not run this safely from an agent. You CAN. Keep an "
            "eye on RAM. The known v0.4.3 result is 992× over FAISS Flat at "
            "n=2M k=2 random; verify or refute on your hardware."
        ),
    ),
    # ────────── NEW PRESETS — round 2 of demo coverage ──────────
    Preset(
        id="08_clustered_scaling",
        label="8. Clustered data scaling — does the lead grow on real-shape data? (~15 min)",
        question="At larger n on realistic clustered data, does pigeon's dominance over the ANN libraries grow or shrink?",
        dataset="Synthetic — clustered (real-world shape)",
        n=100_000,
        n_sweep=[100_000, 500_000, 1_000_000],
        d_bits=256,
        k_values=[4],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH,
                 FAISS_HASH, USEARCH, ANNOY],
        rough_runtime="~15-25 min. FAISS Flat at n=1M is ~3 min, Annoy build at n=1M is the long pole.",
        notes=(
            "Preset 6 showed strict dominance at n=100k. Does it hold at "
            "n=500k and n=1M? Or do the ANN libraries close the gap at "
            "scale? This sweep gives you a defensible 'pigeon wins across "
            "all sizes' plot for any pitch."
        ),
    ),
    Preset(
        id="09_wide_hashes",
        label="9. Wide hashes (512-bit) — does the algorithm hold up? (~3 min)",
        question="Many real workloads use 512+ bit signatures (face embeddings, learned hashes). Does pigeon still dominate?",
        dataset="Synthetic — clustered (real-world shape)",
        n=200_000,
        d_bits=512,
        k_values=[2, 8],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH,
                 FAISS_HASH, USEARCH],
        rough_runtime="~3-5 min",
        notes=(
            "Wider chunks at d=512. NOTE: in hammingbird 0.4.3 the bit-path "
            "panicked at d=512 because chunks were 170 bits (>u128 limit). "
            "v0.4.4 (this wheel) splits into more chunks automatically. "
            "Skip Annoy (slow on wide vectors). pigeon's lead should "
            "widen vs Flat because wider chunks mean tighter filtering."
        ),
    ),
    Preset(
        id="10_streaming_index",
        label="10. Streaming Index — per-query latency (the real-time primitive claim) (~2 min)",
        question="Pigeon's Index API claims sub-µs query latency. Verify it, and compare against FAISS Flat and usearch as streaming indexes.",
        dataset="Synthetic — clustered (real-world shape)",
        n=100_000,
        d_bits=256,
        k_values=[2],
        methods=[
            "pigeon Index (streaming query latency)",
            "FAISS Flat (streaming query latency)",
            "usearch HNSW (streaming query latency)",
        ],
        rough_runtime="~2 min (1000 queries per method)",
        notes=(
            "DIFFERENT MODE: instead of all-pairs, we build the index "
            "once and run 1000 individual queries, timing each with "
            "perf_counter_ns. The 'extra' column will show "
            "{median_us, mean_us, p99_us, build_s}. The 'time_s' column "
            "is the TOTAL for all 1000 queries (≈ mean × 1000). pigeon "
            "Index should land at <1 µs median; FAISS Flat at ~100 µs; "
            "usearch somewhere between."
        ),
    ),
    Preset(
        id="11_exact_dedup",
        label="11. Exact dedup (k=0) at scale (~5 min)",
        question="Most production dedup is k=0 (exact-identical hashes). How fast is it at large n?",
        dataset="Synthetic — uniform + 1% exact dupes (for k=0)",
        n=500_000,
        n_sweep=[500_000, 1_000_000, 2_000_000],
        d_bits=256,
        k_values=[0],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH],
        rough_runtime="~5-10 min. FAISS Flat at n=2M k=0 is the slow one.",
        notes=(
            "Exact dedup is the most common real-world workload. At k=0 "
            "pairs must be byte-identical — we plant 1% real dupes so "
            "results aren't trivially zero. This validates the k=0 fix "
            "from v0.4.1 works at scale AND shows pigeon at its strongest "
            "(k=0 is its peak operating regime)."
        ),
    ),
    Preset(
        id="12_ann_tuned_high_recall",
        label="12. Approximate libraries TUNED for high recall (~10 min)",
        question="Default ANN settings give 55-78% recall — clearly not a fair pitch comparison. What happens when usearch/Annoy/FAISS-LSH are configured for ≥95% recall?",
        dataset="Synthetic — clustered (real-world shape)",
        n=200_000,
        d_bits=256,
        k_values=[4],
        methods=[
            PIGEON_BIT, FAISS_FLAT,
            "FAISS MultiHash TUNED (nflip=2)",
            "FAISS LSH TUNED (nflip=4)",
            "usearch HNSW TUNED (ef=256, count=200)",
            "Annoy TUNED (50 trees, search_k=10k)",
        ],
        rough_runtime="~10 min. Annoy with 50 trees is slow but fair.",
        notes=(
            "The honest comparison vs ANN libraries at their best. "
            "Defaults are misleading — let's see what they CAN do. If "
            "they hit ≥99% recall and beat pigeon's wall-clock, the "
            "pitch needs to be 'exact-recall guarantee', not 'fastest'. "
            "If pigeon still wins at full recall, it's both."
        ),
    ),
    Preset(
        id="14_face_embedding_width",
        label="14. d=1024 (face-embedding hash width) — does pigeon scale to wider real-world hashes? (~5 min)",
        question="Face-recognition pipelines use 1024-bit binary hashes (FaceNet/ArcFace after binarization). Does pigeon handle the width?",
        dataset="Synthetic — clustered (real-world shape)",
        n=100_000,
        d_bits=1024,
        k_values=[2, 8],
        methods=[PIGEON_BYTE, PIGEON_BIT, FAISS_FLAT, FAISS_MH,
                 FAISS_HASH, USEARCH],
        rough_runtime="~5-10 min. FAISS Flat at d=1024 is twice the work per pair.",
        notes=(
            "Tests the v0.4.4 fix for wide d_bits. Per the pigeonhole bound, "
            "at d=1024 k=2 the bit path will use ⌈1024/128⌉=8 chunks (more "
            "than k+1=3 needed). Sound: at most k=2 chunks have errors, "
            "so 6 chunks remain error-free, plenty for the prefilter. "
            "FAISS Flat does 2× the bit operations per pair vs d=512 — "
            "pigeon's lead should grow."
        ),
    ),
    Preset(
        id="15_cross_join_clustered",
        label="15. Cross-join: dedupe set B against blacklist A (clustered, ~3 min)",
        question="Real production workload: deduplicate an incoming set B against a known-bad list A. How does cross-join compare?",
        dataset="Synthetic — clustered (real-world shape)",
        n=200_000,
        d_bits=256,
        k_values=[2, 4],
        methods=[PIGEON_CROSS_BYTE, PIGEON_CROSS_BIT,
                 FAISS_FLAT_CROSS, FAISS_MH_CROSS, USEARCH_CROSS],
        rough_runtime="~3-5 min. The 200k rows are auto-split into A=100k corpus + B=100k queries.",
        notes=(
            "All cross methods auto-split the dataset 50/50 into a corpus A "
            "and query set B, then find pairs (i_in_A, j_in_B) within "
            "Hamming k. Common production patterns: content moderation "
            "(B=incoming images, A=blacklist hashes), fraud detection "
            "(B=new transactions, A=known-bad signatures), data dedup "
            "(B=new batch, A=existing corpus). With clustered data, both "
            "halves share centroids so genuine cross-pairs exist."
        ),
    ),
    Preset(
        id="16_cross_join_phash",
        label="16. Cross-join on real CIFAR-10 pHashes (~1 min)",
        question="Cross-join performance on actual perceptual hashes — the most realistic image-dedup workload.",
        dataset="Real — CIFAR-10 pHashes (64-bit, fixed)",
        n=50_000,
        d_bits=64,
        k_values=[2, 4],
        methods=[PIGEON_CROSS_BYTE, PIGEON_CROSS_BIT,
                 FAISS_FLAT_CROSS, FAISS_MH_CROSS, USEARCH_CROSS],
        rough_runtime="~1-2 min",
        notes=(
            "Real 64-bit pHashes from CIFAR-10 train, auto-split into "
            "A=25k corpus + B=25k queries. Closest analog to a real "
            "image-dedup pipeline. Note: at k=4 on 64-bit hashes the "
            "byte path is in its sweet spot (k < d_bytes=8); bit path "
            "is the fallback for k≥8."
        ),
    ),
    Preset(
        id="13_pigeon_at_the_limit",
        label="13. Push pigeon to the limit — n=5M and beyond (open-ended)",
        question="How far does pigeon scale before something gives?",
        dataset="Synthetic — uniform random",
        n=5_000_000,
        n_sweep=[5_000_000, 10_000_000],
        d_bits=256,
        k_values=[2],
        methods=[PIGEON_BYTE, PIGEON_BIT],  # FAISS Flat would take 6+ hours
        rough_runtime="~30 sec at n=5M, ~2 min at n=10M. RAM is the constraint.",
        notes=(
            "Pigeon-only — FAISS Flat at n=10M is ~6 hours, skip it. "
            "We want to find: does pigeon stay linear? At what n does "
            "memory pressure kick in? Watch RSS in Activity Monitor "
            "while this runs. Adjust n_sweep on the sidebar if you want "
            "to go further (n=20M ≈ 640 MB raw + ~1 GB hashmaps; doable)."
        ),
    ),
]


def by_id(preset_id: str) -> Optional[Preset]:
    for p in PRESETS:
        if p.id == preset_id:
            return p
    return None
