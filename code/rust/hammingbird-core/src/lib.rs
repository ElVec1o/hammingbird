//! Pigeonhole prefilter for exact Hamming-distance pair search on packed
//! binary vectors. Port of `code/python/hammingbird_ref/core.py`. Chunk boundaries
//! match the Python library (np.linspace truncated to int).
//!
//! v0.2 engineering:
//!   * `find_pairs_self_st`   — original single-threaded baseline (preserved).
//!   * `find_pairs_self_par`  — rayon-parallel bucket build, one task per
//!                              chunk position. No shared HashMap → no
//!                              contention. Verify step also parallelized.
//!   * `find_pairs_self_par_prefetch` — same as par + explicit prefetch in
//!                              the verify loop.
//!   * `find_pairs_self`      — public default; currently aliases the
//!                              parallel-with-prefetch version.

use hashbrown::HashMap;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

/// Inclusive chunk byte ranges, matching np.linspace(0, n_bytes, k+2).astype(int).
pub fn chunk_boundaries(n_bytes: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n_chunks + 1);
    for i in 0..=n_chunks {
        edges.push(((i as u64) * (n_bytes as u64) / (n_chunks as u64)) as usize);
    }
    (0..n_chunks)
        .filter_map(|i| {
            let (lo, hi) = (edges[i], edges[i + 1]);
            if hi > lo { Some((lo, hi)) } else { None }
        })
        .collect()
}

/// Like `hamming` but bails out as soon as the running popcount exceeds `k`.
/// Returns `Some(d)` if `hamming(a,b) <= k`, else `None`. Processes the row
/// in 8-byte (u64) chunks so the early-exit granularity is 64 bits.
#[inline]
pub fn hamming_le_k(a: &[u8], b: &[u8], k: u32) -> Option<u32> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let n8 = n / 8;
    let mut acc: u32 = 0;
    let pa = a.as_ptr() as *const u64;
    let pb = b.as_ptr() as *const u64;
    for i in 0..n8 {
        unsafe {
            let xa = pa.add(i).read_unaligned();
            let xb = pb.add(i).read_unaligned();
            acc += (xa ^ xb).count_ones();
        }
        if acc > k {
            return None;
        }
    }
    for i in (n8 * 8)..n {
        acc += (a[i] ^ b[i]).count_ones();
        if acc > k {
            return None;
        }
    }
    Some(acc)
}

/// Popcount the XOR of two equal-length byte slices.
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let n8 = n / 8;
    let mut acc: u32 = 0;
    let pa = a.as_ptr() as *const u64;
    let pb = b.as_ptr() as *const u64;
    for i in 0..n8 {
        unsafe {
            let xa = pa.add(i).read_unaligned();
            let xb = pb.add(i).read_unaligned();
            acc += (xa ^ xb).count_ones();
        }
    }
    for i in (n8 * 8)..n {
        acc += (a[i] ^ b[i]).count_ones();
    }
    acc
}

#[inline]
fn emit_pairs<K>(buckets: &HashMap<K, Vec<u32>, FxBuildHasher>, out: &mut Vec<(u32, u32)>) {
    for idxs in buckets.values() {
        if idxs.len() < 2 {
            continue;
        }
        for a in 0..idxs.len() {
            let ia = idxs[a];
            for b in (a + 1)..idxs.len() {
                let ib = idxs[b];
                let (i, j) = if ia < ib { (ia, ib) } else { (ib, ia) };
                out.push((i, j));
            }
        }
    }
}

/// Build candidate (i,j) pairs for one chunk position. No dedup yet.
fn candidates_for_chunk(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    lo: usize,
    hi: usize,
) -> Vec<(u32, u32)> {
    let chunk_len = hi - lo;
    let mut out: Vec<(u32, u32)> = Vec::new();
    if chunk_len <= 16 {
        let mut buckets: HashMap<u128, Vec<u32>, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for i in 0..n {
            let row = &data[i * d_bytes..i * d_bytes + d_bytes];
            let mut buf = [0u8; 16];
            buf[..chunk_len].copy_from_slice(&row[lo..hi]);
            let key = u128::from_le_bytes(buf);
            buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
        }
        emit_pairs(&buckets, &mut out);
    } else {
        let mut buckets: HashMap<Box<[u8]>, Vec<u32>, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for i in 0..n {
            let row = &data[i * d_bytes..i * d_bytes + d_bytes];
            let key: Box<[u8]> = row[lo..hi].into();
            buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
        }
        emit_pairs(&buckets, &mut out);
    }
    out
}

/// Single-threaded baseline. Preserved for benchmarking.
pub fn find_pairs_self_st(data: &[u8], n: usize, d_bytes: usize, k: u32) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(n);
    for (lo, hi) in &chunks {
        let mut c = candidates_for_chunk(data, n, d_bytes, *lo, *hi);
        candidates.append(&mut c);
    }
    if candidates.is_empty() {
        return Vec::new();
    }
    candidates.sort_unstable();
    candidates.dedup();

    let mut out = Vec::new();
    for (i, j) in candidates {
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
        let d = hamming(ra, rb);
        if d <= k {
            out.push((i, j, d));
        }
    }
    out
}

/// Rayon-parallel version: each chunk position runs on its own thread,
/// then sort/dedup/verify in parallel.
pub fn find_pairs_self_par(data: &[u8], n: usize, d_bytes: usize, k: u32) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);

    // Per-chunk candidate vectors built in parallel (no shared state).
    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_chunk(data, n, d_bytes, *lo, *hi))
        .collect();

    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk {
        candidates.append(&mut v);
    }
    candidates.par_sort_unstable();
    candidates.dedup();

    // Parallel verify.
    let out: Vec<(u32, u32, u32)> = candidates
        .par_iter()
        .filter_map(|(i, j)| {
            let ra = &data[(*i as usize) * d_bytes..(*i as usize + 1) * d_bytes];
            let rb = &data[(*j as usize) * d_bytes..(*j as usize + 1) * d_bytes];
            let d = hamming(ra, rb);
            if d <= k { Some((*i, *j, d)) } else { None }
        })
        .collect();
    out
}

/// Software prefetch helper. On aarch64 uses the PRFM instruction via the
/// stable `core::arch::aarch64::_prefetch` intrinsic; elsewhere a no-op.
#[inline(always)]
fn prefetch_row(_data: &[u8], _row: usize, _d_bytes: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1KEEP, [x] — prefetch for load into L1, retain.
        let p = _data.as_ptr().add(_row * _d_bytes);
        core::arch::asm!(
            "prfm pldl1keep, [{x}]",
            x = in(reg) p,
            options(nostack, preserves_flags, readonly),
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let p = _data.as_ptr().add(_row * _d_bytes) as *const i8;
        _mm_prefetch(p, _MM_HINT_T0);
    }
}

/// Verify a slice of candidates with prefetch lookahead.
fn verify_with_prefetch(
    cand: &[(u32, u32)],
    data: &[u8],
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    const LOOKAHEAD: usize = 8;
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    let n_cand = cand.len();
    for idx in 0..n_cand {
        if idx + LOOKAHEAD < n_cand {
            let (pi, pj) = cand[idx + LOOKAHEAD];
            prefetch_row(data, pi as usize, d_bytes);
            prefetch_row(data, pj as usize, d_bytes);
        }
        let (i, j) = cand[idx];
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
        if let Some(d) = hamming_le_k(ra, rb, k) {
            out.push((i, j, d));
        }
    }
    out
}

/// Verify with prefetch and the *old* full-popcount path (no early exit).
/// Used only by the Task-3 bench harness for an apples-to-apples comparison.
fn verify_with_prefetch_full(
    cand: &[(u32, u32)],
    data: &[u8],
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    const LOOKAHEAD: usize = 8;
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    let n_cand = cand.len();
    for idx in 0..n_cand {
        if idx + LOOKAHEAD < n_cand {
            let (pi, pj) = cand[idx + LOOKAHEAD];
            prefetch_row(data, pi as usize, d_bytes);
            prefetch_row(data, pj as usize, d_bytes);
        }
        let (i, j) = cand[idx];
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
        let d = hamming(ra, rb);
        if d <= k {
            out.push((i, j, d));
        }
    }
    out
}

/// Variant of `find_pairs_self_no_dedup_par_prefetch` that uses the *old*
/// full-popcount verify path. Exposed only for Task 3 bench comparison.
pub fn find_pairs_self_no_dedup_par_prefetch_full(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");
    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);
    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_chunk(data, n, d_bytes, *lo, *hi))
        .collect();
    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 { return Vec::new(); }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk { candidates.append(&mut v); }
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (candidates.len() / (n_threads * 4)).max(1024);
    let parts: Vec<Vec<(u32, u32, u32)>> = candidates
        .par_chunks(chunk_size)
        .map(|slice| verify_with_prefetch_full(slice, data, d_bytes, k))
        .collect();
    let mut verified: Vec<(u32, u32, u32)> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for mut p in parts { verified.append(&mut p); }
    verified.par_sort_unstable();
    verified.dedup();
    verified
}

/// Rayon-parallel + prefetch verify.
pub fn find_pairs_self_par_prefetch(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);

    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_chunk(data, n, d_bytes, *lo, *hi))
        .collect();

    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk {
        candidates.append(&mut v);
    }
    candidates.par_sort_unstable();
    candidates.dedup();

    // Parallel verify over chunks of the sorted candidate list, with
    // per-chunk prefetch lookahead. Sorted-by-i means adjacent candidates
    // share row `i`, so prefetching row j of the next candidate is the win.
    let chunk_size = (candidates.len() / (rayon::current_num_threads() * 4)).max(1024);
    let parts: Vec<Vec<(u32, u32, u32)>> = candidates
        .par_chunks(chunk_size)
        .map(|slice| verify_with_prefetch(slice, data, d_bytes, k))
        .collect();
    let mut out: Vec<(u32, u32, u32)> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for mut p in parts {
        out.append(&mut p);
    }
    // Candidates were sorted by (i,j), so per-chunk results are already in order.
    out
}

/// Error returned when the caller asks for a regime the byte-aligned chunk
/// pigeonhole cannot satisfy. The pigeonhole guarantee requires at least
/// `k + 1` disjoint byte-aligned chunks; when `k >= d_bytes`, only
/// `d_bytes` chunks exist and the algorithm can silently miss pairs
/// (see real_phash_report.md — CIFAR-10 at d_bytes=8, k=8 misses 2/1347).
///
/// In that regime callers should fall back to FAISS `IndexBinaryFlat`
/// (exact O(n²) brute force) which has no such limitation.
#[derive(Debug)]
pub struct UnsupportedRegime {
    pub k: u32,
    pub d_bytes: usize,
}

impl std::fmt::Display for UnsupportedRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hammingbird: byte-aligned chunk pigeonhole requires k < d_bytes (got k={}, d_bytes={}). \
             For this regime use FAISS IndexBinaryFlat or wait for bit-level chunks in a later release.",
            self.k, self.d_bytes
        )
    }
}

impl std::error::Error for UnsupportedRegime {}

/// Returns `true` iff the given `(k, d_bytes)` is in the regime where the
/// byte-aligned chunk pigeonhole gives an exact result. See
/// `UnsupportedRegime` for the math.
#[inline]
pub fn is_supported(k: u32, d_bytes: usize) -> bool {
    (k as usize) < d_bytes
}

/// Bit-level analogue of `is_supported`: the bit-aligned chunk pigeonhole
/// requires `k < d_bits = 8 * d_bytes`, which is essentially always true
/// for d_bytes >= 8.
#[inline]
pub fn is_supported_bit(k: u32, d_bytes: usize) -> bool {
    (k as usize) < 8 * d_bytes
}

// ---------------------------------------------------------------------------
// Bit-level chunking (v0.4, Task 1).
//
// Byte-aligned chunks break the pigeonhole guarantee when k >= d_bytes: we
// only have d_bytes possible disjoint chunks but need k+1 of them. Bit-level
// chunking partitions the d_bits bits into approximately equal half-open
// ranges, so the guarantee holds whenever k < d_bits = 8 * d_bytes.
// ---------------------------------------------------------------------------

/// Inclusive bit ranges. Each chunk is a half-open `[lo_bit, hi_bit)` range
/// over the row's bit positions. Edges are `(i * d_bits) / n_chunks`
/// truncated (matches the byte boundary formula).
pub fn chunk_bit_boundaries(d_bits: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n_chunks + 1);
    for i in 0..=n_chunks {
        edges.push(((i as u64) * (d_bits as u64) / (n_chunks as u64)) as usize);
    }
    (0..n_chunks)
        .filter_map(|i| {
            let (lo, hi) = (edges[i], edges[i + 1]);
            if hi > lo { Some((lo, hi)) } else { None }
        })
        .collect()
}

/// Pick a bit-chunk layout for `(d_bits, k)` that satisfies BOTH the
/// pigeonhole guarantee (≥ k+1 disjoint chunks) AND the u128 chunk-key width
/// limit (≤ 128 bits per chunk).
///
/// For wide d (e.g. d=512 with k=2), `k+1=3` chunks of 170 bits each would
/// exceed u128. We bump `n_chunks` up to `ceil(d_bits / 128)` to keep every
/// chunk under the limit. Pigeonhole still holds: with m chunks and ≤ k
/// errors, at least m-k chunks are error-free, and since m ≥ k+1 we are
/// guaranteed at least one error-free chunk.
///
/// At k=0 the single-chunk shortcut (first 128 bits as prefix prefilter) is
/// also handled here for consistency.
pub fn bit_chunks_for(d_bits: usize, k: u32) -> Vec<(usize, usize)> {
    if k == 0 && d_bits > 128 {
        return vec![(0usize, 128usize)];
    }
    let by_pigeonhole = (k as usize) + 1;
    let by_width = (d_bits + 127) / 128; // ceil(d_bits / 128)
    let n_chunks = by_pigeonhole.max(by_width).max(1);
    chunk_bit_boundaries(d_bits, n_chunks)
}

/// Extract bits at positions `[lo_bit, hi_bit)` from `row`, returned as a
/// u128 with bit 0 of the result corresponding to the bit at `lo_bit`.
///
/// Bit indexing is MSB-first inside each byte: bit position `p` of the row
/// is `(row[p / 8] >> (7 - p % 8)) & 1`. This matches the convention used
/// in `unpack_bits`.
///
/// Panics if `hi_bit - lo_bit > 128`.
#[inline]
pub fn extract_bits(row: &[u8], lo_bit: usize, hi_bit: usize) -> u128 {
    assert!(hi_bit >= lo_bit, "extract_bits: hi_bit < lo_bit");
    let width = hi_bit - lo_bit;
    assert!(width <= 128, "extract_bits: width {} > 128", width);
    if width == 0 {
        return 0;
    }
    let mut acc: u128 = 0;
    let mut nbits: u32 = 0;
    let mut p = lo_bit;
    while p < hi_bit {
        let byte_idx = p / 8;
        let bit_in_byte = p % 8;
        // How many bits we can take from this byte (MSB-first), bounded by
        // remaining width.
        let take = std::cmp::min(8 - bit_in_byte, hi_bit - p);
        // Extract `take` bits starting at MSB position `bit_in_byte` in this byte.
        let shift_right = 8 - bit_in_byte - take;
        let mask = if take == 8 { 0xFFu8 } else { (1u8 << take) - 1 };
        let bits = (row[byte_idx] >> shift_right) & mask;
        acc |= (bits as u128) << nbits;
        nbits += take as u32;
        p += take;
    }
    acc
}

/// Build candidate pairs for one bit-level chunk position.
fn candidates_for_bit_chunk(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    lo_bit: usize,
    hi_bit: usize,
) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    let mut buckets: HashMap<u128, Vec<u32>, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(n, FxBuildHasher);
    for i in 0..n {
        let row = &data[i * d_bytes..i * d_bytes + d_bytes];
        let key = extract_bits(row, lo_bit, hi_bit);
        buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
    }
    emit_pairs(&buckets, &mut out);
    out
}

/// Like `find_pairs_self_no_dedup_par_prefetch` but uses bit-level chunks,
/// so the safe constraint is `k < 8 * d_bytes` (always true in practice).
pub fn find_pairs_self_bit(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");
    assert!(
        is_supported_bit(k, d_bytes),
        "hammingbird: k ({}) must be < 8*d_bytes ({})",
        k, 8 * d_bytes
    );

    let d_bits = 8 * d_bytes;
    let chunks = bit_chunks_for(d_bits, k);
    // Invariant guaranteed by bit_chunks_for; assert kept as defense in depth.
    for &(lo, hi) in &chunks {
        debug_assert!(hi - lo <= 128, "bit chunk wider than 128 bits");
    }

    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_bit_chunk(data, n, d_bytes, *lo, *hi))
        .collect();

    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk {
        candidates.append(&mut v);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (candidates.len() / (n_threads * 4)).max(1024);
    let parts: Vec<Vec<(u32, u32, u32)>> = candidates
        .par_chunks(chunk_size)
        .map(|slice| verify_with_prefetch(slice, data, d_bytes, k))
        .collect();
    let mut verified: Vec<(u32, u32, u32)> =
        Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for mut p in parts {
        verified.append(&mut p);
    }
    verified.par_sort_unstable();
    verified.dedup();
    verified
}

/// Public default: parallel no-dedup + prefetch (Round-4 winner, 3-4× over
/// the prior `par_prefetch` default at k>=16). Panics with a clear message
/// if `k >= d_bytes` — that regime can silently miss pairs; use
/// `try_find_pairs_self` for a non-panicking variant or call FAISS Flat.
pub fn find_pairs_self(data: &[u8], n: usize, d_bytes: usize, k: u32) -> Vec<(u32, u32, u32)> {
    assert!(
        is_supported(k, d_bytes),
        "hammingbird: k ({}) must be < d_bytes ({}). At k >= d_bytes the byte-aligned \
         chunk pigeonhole can miss pairs. Use FAISS IndexBinaryFlat for that regime.",
        k, d_bytes
    );
    find_pairs_self_no_dedup_par_prefetch(data, n, d_bytes, k)
}

/// Non-panicking variant of `find_pairs_self`. Returns `Err` when the
/// regime is unsupported (`k >= d_bytes`).
pub fn try_find_pairs_self(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Result<Vec<(u32, u32, u32)>, UnsupportedRegime> {
    if !is_supported(k, d_bytes) {
        return Err(UnsupportedRegime { k, d_bytes });
    }
    Ok(find_pairs_self_no_dedup_par_prefetch(data, n, d_bytes, k))
}

// ---------------------------------------------------------------------------
// Dedup-attack variants (Round 3, Task 1).
//
// Round-2 profiling showed sort+dedup is 74-82% of total at k>=16. These
// variants attack that phase directly.
// ---------------------------------------------------------------------------

/// Single-threaded baseline equivalent to `find_pairs_self_st`, but the
/// candidate pairs are packed into u64 before sort/dedup. This gives pdqsort
/// a contiguous primitive type to work with (smaller compares, no tuple
/// branches), which is the cheap experiment.
pub fn find_pairs_self_radix(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);
    // Build candidates as packed u64.
    let mut packed: Vec<u64> = Vec::new();
    for (lo, hi) in &chunks {
        let c = candidates_for_chunk(data, n, d_bytes, *lo, *hi);
        packed.reserve(c.len());
        for (i, j) in c {
            packed.push(((i as u64) << 32) | (j as u64));
        }
    }
    if packed.is_empty() {
        return Vec::new();
    }
    packed.sort_unstable();
    packed.dedup();

    let mut out = Vec::new();
    for p in packed {
        let i = (p >> 32) as u32;
        let j = (p & 0xFFFF_FFFF) as u32;
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
        let d = hamming(ra, rb);
        if d <= k {
            out.push((i, j, d));
        }
    }
    out
}

/// Skip candidate dedup entirely. Verify every duplicate candidate; the
/// output (which is much smaller than the candidate set at large k since
/// the vast majority fail the Hamming bound) is deduped at the end.
pub fn find_pairs_self_no_dedup(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);
    let mut candidates: Vec<(u32, u32)> = Vec::new();
    for (lo, hi) in &chunks {
        let mut c = candidates_for_chunk(data, n, d_bytes, *lo, *hi);
        candidates.append(&mut c);
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    // Verify every candidate (with duplicates).
    let mut verified: Vec<(u32, u32, u32)> = Vec::new();
    for (i, j) in candidates {
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
        let d = hamming(ra, rb);
        if d <= k {
            verified.push((i, j, d));
        }
    }
    // Final dedup pass on the (much smaller) verified set.
    verified.sort_unstable();
    verified.dedup();
    verified
}

/// Parallel no-dedup: per-chunk candidate generation in parallel, no global
/// sort/dedup on the candidate set; instead verify every candidate (with
/// duplicates) in parallel and dedup the (small) verified output at the end.
///
/// Round-3 showed `no_dedup` (single-thread) beats sort+dedup at high k. This
/// adds rayon parallelism plus prefetch in the verify pass, since verify is
/// now the dominant cost.
pub fn find_pairs_self_no_dedup_par_prefetch(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);

    // Per-chunk candidate vectors in parallel — no shared state.
    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_chunk(data, n, d_bytes, *lo, *hi))
        .collect();

    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 {
        return Vec::new();
    }

    // Concatenate (no sort, no dedup).
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk {
        candidates.append(&mut v);
    }

    // Parallel verify with prefetch lookahead inside each chunk.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (candidates.len() / (n_threads * 4)).max(1024);
    let parts: Vec<Vec<(u32, u32, u32)>> = candidates
        .par_chunks(chunk_size)
        .map(|slice| verify_with_prefetch(slice, data, d_bytes, k))
        .collect();
    let mut verified: Vec<(u32, u32, u32)> =
        Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for mut p in parts {
        verified.append(&mut p);
    }

    // Final dedup on the (small) verified output.
    verified.par_sort_unstable();
    verified.dedup();
    verified
}

/// Alias without prefetch lookahead (single-pass verify per chunk), retained
/// for completeness — folds into the prefetch variant when called.
pub fn find_pairs_self_no_dedup_par(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    find_pairs_self_no_dedup_par_prefetch(data, n, d_bytes, k)
}

// ---------------------------------------------------------------------------
// Cross-join: find pairs (a_id, b_id) with hamming(A[a_id], B[b_id]) <= k.
// ---------------------------------------------------------------------------

/// For one chunk position, bucket A and B by the chunk bytes; emit all
/// (a_id, b_id) pairs where the chunk bytes match exactly.
fn candidates_for_chunk_cross(
    a_data: &[u8],
    n_a: usize,
    b_data: &[u8],
    n_b: usize,
    d_bytes: usize,
    lo: usize,
    hi: usize,
) -> Vec<(u32, u32)> {
    let chunk_len = hi - lo;
    let mut out: Vec<(u32, u32)> = Vec::new();
    if chunk_len <= 16 {
        let mut a_buckets: HashMap<u128, Vec<u32>, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(n_a, FxBuildHasher);
        for i in 0..n_a {
            let row = &a_data[i * d_bytes..i * d_bytes + d_bytes];
            let mut buf = [0u8; 16];
            buf[..chunk_len].copy_from_slice(&row[lo..hi]);
            let key = u128::from_le_bytes(buf);
            a_buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
        }
        for j in 0..n_b {
            let row = &b_data[j * d_bytes..j * d_bytes + d_bytes];
            let mut buf = [0u8; 16];
            buf[..chunk_len].copy_from_slice(&row[lo..hi]);
            let key = u128::from_le_bytes(buf);
            if let Some(v) = a_buckets.get(&key) {
                for &ai in v {
                    out.push((ai, j as u32));
                }
            }
        }
    } else {
        let mut a_buckets: HashMap<Box<[u8]>, Vec<u32>, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(n_a, FxBuildHasher);
        for i in 0..n_a {
            let row = &a_data[i * d_bytes..i * d_bytes + d_bytes];
            let key: Box<[u8]> = row[lo..hi].into();
            a_buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
        }
        for j in 0..n_b {
            let row = &b_data[j * d_bytes..j * d_bytes + d_bytes];
            let key: Box<[u8]> = row[lo..hi].into();
            if let Some(v) = a_buckets.get(&key) {
                for &ai in v {
                    out.push((ai, j as u32));
                }
            }
        }
    }
    out
}

/// Find all (a_id, b_id, dist) with hamming(A[a_id], B[b_id]) <= k.
/// Returns sorted by (a_id, b_id), each pair appearing once.
/// Cross-join, bit-level chunks. Same algorithm as `find_pairs_cross` but uses
/// bit-aligned chunks so `k < 8*d_bytes` is the constraint (essentially always
/// true). Closes the `k >= d_bytes` correctness gap that `find_pairs_cross`
/// has on small-d inputs (e.g. 64-bit pHashes at k=8).
pub fn find_pairs_cross_bit(
    a_data: &[u8],
    n_a: usize,
    b_data: &[u8],
    n_b: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(a_data.len(), n_a * d_bytes, "a_data length mismatch");
    assert_eq!(b_data.len(), n_b * d_bytes, "b_data length mismatch");
    assert!(
        is_supported_bit(k, d_bytes),
        "hammingbird: k ({}) must be < 8*d_bytes ({})",
        k, 8 * d_bytes
    );
    assert!(n_a <= u32::MAX as usize, "n_a exceeds u32::MAX");
    assert!(n_b <= u32::MAX as usize, "n_b exceeds u32::MAX");

    let d_bits = 8 * d_bytes;
    let chunks = bit_chunks_for(d_bits, k);
    for &(lo, hi) in &chunks {
        debug_assert!(hi - lo <= 128, "bit chunk wider than 128 bits");
    }

    // Rayon-parallel per-chunk candidate generation.
    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|&(lo, hi)| {
            let mut buckets_a: HashMap<u128, Vec<u32>, FxBuildHasher> =
                HashMap::with_capacity_and_hasher(n_a, FxBuildHasher);
            for i in 0..n_a {
                let row = &a_data[i * d_bytes..i * d_bytes + d_bytes];
                buckets_a.entry(extract_bits(row, lo, hi)).or_default().push(i as u32);
            }
            let mut buckets_b: HashMap<u128, Vec<u32>, FxBuildHasher> =
                HashMap::with_capacity_and_hasher(n_b, FxBuildHasher);
            for j in 0..n_b {
                let row = &b_data[j * d_bytes..j * d_bytes + d_bytes];
                buckets_b.entry(extract_bits(row, lo, hi)).or_default().push(j as u32);
            }
            let mut local: Vec<(u32, u32)> = Vec::new();
            for (key, ais) in &buckets_a {
                if let Some(bjs) = buckets_b.get(key) {
                    for &ai in ais {
                        for &bj in bjs {
                            local.push((ai, bj));
                        }
                    }
                }
            }
            local
        })
        .collect();
    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 { return Vec::new(); }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk { candidates.append(&mut v); }
    candidates.par_sort_unstable();
    candidates.dedup();

    let out: Vec<(u32, u32, u32)> = candidates
        .par_iter()
        .filter_map(|(ai, bj)| {
            let ra = &a_data[(*ai as usize) * d_bytes..(*ai as usize + 1) * d_bytes];
            let rb = &b_data[(*bj as usize) * d_bytes..(*bj as usize + 1) * d_bytes];
            hamming_le_k(ra, rb, k).map(|d| (*ai, *bj, d))
        })
        .collect();
    out
}

pub fn find_pairs_cross(
    a_data: &[u8],
    n_a: usize,
    b_data: &[u8],
    n_b: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(a_data.len(), n_a * d_bytes, "a_data length mismatch");
    assert_eq!(b_data.len(), n_b * d_bytes, "b_data length mismatch");
    assert!(
        is_supported(k, d_bytes),
        "hammingbird: k ({}) must be < d_bytes ({}). At k >= d_bytes the byte-aligned \
         chunk pigeonhole can miss pairs. Use FAISS IndexBinaryFlat for that regime.",
        k, d_bytes
    );
    assert!(n_a <= u32::MAX as usize, "n_a exceeds u32::MAX");
    assert!(n_b <= u32::MAX as usize, "n_b exceeds u32::MAX");

    let chunks = chunk_boundaries(d_bytes, (k + 1) as usize);
    // Rayon-parallel per-chunk candidate generation; sequential dedup + verify
    // afterward (verify is cheap thanks to hamming_le_k early exit).
    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|(lo, hi)| candidates_for_chunk_cross(a_data, n_a, b_data, n_b, d_bytes, *lo, *hi))
        .collect();
    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 { return Vec::new(); }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk { candidates.append(&mut v); }
    candidates.par_sort_unstable();
    candidates.dedup();

    let out: Vec<(u32, u32, u32)> = candidates
        .par_iter()
        .filter_map(|(ai, bj)| {
            let ra = &a_data[(*ai as usize) * d_bytes..(*ai as usize + 1) * d_bytes];
            let rb = &b_data[(*bj as usize) * d_bytes..(*bj as usize + 1) * d_bytes];
            hamming_le_k(ra, rb, k).map(|d| (*ai, *bj, d))
        })
        .collect();
    out
}

// ---------------------------------------------------------------------------
// Positional q-gram counting filter (Jokinen-Ukkonen lower bound).
//
// Pair within Hamming k must share >= t = (d_bits - q + 1) - k*q positional
// q-grams. We compute, for every row, its q-gram value at every starting bit
// position; build per-position sorted (gram, row) arrays; then for each query
// row i, look up its gram at each position, and increment a per-other-row
// counter. Candidates are rows j with count[j] >= t; verify with popcount.
// ---------------------------------------------------------------------------

/// Unpack packed MSB-first bits into a (n, d_bits) flat row-major u8 array of 0/1.
fn unpack_bits(data: &[u8], n: usize, d_bytes: usize) -> Vec<u8> {
    let d_bits = d_bytes * 8;
    let mut out = vec![0u8; n * d_bits];
    for i in 0..n {
        let row_in = &data[i * d_bytes..(i + 1) * d_bytes];
        let row_out = &mut out[i * d_bits..(i + 1) * d_bits];
        for b in 0..d_bytes {
            let byte = row_in[b];
            for k in 0..8 {
                row_out[b * 8 + k] = (byte >> (7 - k)) & 1;
            }
        }
    }
    out
}

/// Compute per-row q-gram values for one starting position p.
/// Returns Vec<u32> of length n.
#[inline]
fn grams_at_position(bits: &[u8], n: usize, d_bits: usize, p: usize, q: u32) -> Vec<u32> {
    let mut out = vec![0u32; n];
    for i in 0..n {
        let row = &bits[i * d_bits + p..i * d_bits + p + q as usize];
        let mut v: u32 = 0;
        for j in 0..q as usize {
            v = (v << 1) | (row[j] as u32);
        }
        out[i] = v;
    }
    out
}

/// Per-position index: sorted by gram value. `grams[idx]` and `rows[idx]` are
/// parallel. `query[i]` is the gram value of row i at this position (for O(1)
/// lookup of "what value does row i have here?").
struct PosIndex {
    grams_sorted: Vec<u32>,
    rows_sorted: Vec<u32>,
    query: Vec<u32>,
}

fn build_pos_index(bits: &[u8], n: usize, d_bits: usize, p: usize, q: u32) -> PosIndex {
    let grams = grams_at_position(bits, n, d_bits, p, q);
    // Sort row indices by gram value.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&r| grams[r as usize]);
    let grams_sorted: Vec<u32> = order.iter().map(|&r| grams[r as usize]).collect();
    PosIndex { grams_sorted, rows_sorted: order, query: grams }
}

/// Find all pairs in `data` within Hamming distance <= k using the positional
/// q-gram counting filter. Result is the same set as `find_pairs_self_st`.
pub fn find_pairs_qgram_self(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
    q: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");
    assert!(q >= 1 && q <= 16, "q must be in [1, 16]");
    let d_bits = d_bytes * 8;
    assert!(d_bits >= q as usize, "q exceeds d_bits");

    let n_pos = d_bits - q as usize + 1;
    let t_i = n_pos as i64 - (k as i64) * (q as i64);
    if t_i <= 0 {
        // Filter is vacuous: fall back to chunk pigeonhole for correctness.
        return find_pairs_self_st(data, n, d_bytes, k);
    }
    let t = t_i as u32;

    // Unpack bits once.
    let bits = unpack_bits(data, n, d_bytes);

    // Build per-position sorted indices.
    let mut indices: Vec<PosIndex> = Vec::with_capacity(n_pos);
    for p in 0..n_pos {
        indices.push(build_pos_index(&bits, n, d_bits, p, q));
    }

    // For each query row i, accumulate counts of co-occurring positions with
    // every other row j > i. We only count j > i to avoid double-counting; this
    // is correct because the count is symmetric: # shared positions(i,j) ==
    // shared positions(j,i), and we will see this count when we iterate i and
    // look at all bucket-mates (which include both j>i and j<i; we only
    // increment j>i so that the bound check is performed once per pair).
    //
    // However: if we only increment j>i, we may *miss* the pair if all the
    // bucket positions place i strictly after j in the bucket. That's not a
    // problem: for ANY position where rows i and j share a gram value, the
    // pair (min(i,j), max(i,j)) is the same. We need >=t such positions. So:
    // when processing query row i, for each position, find i's gram value,
    // then iterate the contiguous bucket of rows sharing that value; for each
    // bucket member j != i, if j > i increment count[j].
    //
    // Then for j > i where count[j] >= t, verify with popcount.

    // count[j] : how many positions share the gram with row i (j>i only).
    // Use u16 — n_pos <= 256*8 = ~2048 positions max for d_bytes <= 256;
    // for safety use u16 (fits up to 65535).
    let mut count: Vec<u16> = vec![0u16; n];
    // Track which j's were touched, so we can clear only those.
    let mut touched: Vec<u32> = Vec::with_capacity(1024);

    let mut out: Vec<(u32, u32, u32)> = Vec::new();

    for i in 0..n as u32 {
        // Reset previously touched counts.
        for &j in &touched {
            count[j as usize] = 0;
        }
        touched.clear();

        for p in 0..n_pos {
            let idx = &indices[p];
            let v = idx.query[i as usize];
            // Find contiguous range in grams_sorted equal to v via partition_point.
            let lo = idx.grams_sorted.partition_point(|&x| x < v);
            let hi = idx.grams_sorted.partition_point(|&x| x <= v);
            // Iterate bucket; only increment j > i.
            for k_in in lo..hi {
                let j = idx.rows_sorted[k_in];
                if j > i {
                    let c = &mut count[j as usize];
                    if *c == 0 {
                        touched.push(j);
                    }
                    *c += 1;
                }
            }
        }

        // Verify candidates.
        let ra = &data[(i as usize) * d_bytes..(i as usize + 1) * d_bytes];
        for &j in &touched {
            if count[j as usize] >= t as u16 {
                let rb = &data[(j as usize) * d_bytes..(j as usize + 1) * d_bytes];
                let d = hamming(ra, rb);
                if d <= k {
                    out.push((i, j, d));
                }
            }
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// Entropy-aware adaptive chunk planning (v0.4, Task 4).
//
// Real-world binary data (perceptual hashes, SimHash signatures) is NOT
// uniform: some bits have low entropy. A uniform chunker wastes those bits
// as part of chunk keys; an entropy-aware planner partitions only the
// informative bits, balanced by entropy, into k+1 chunks. Disjoint
// non-contiguous bit chunks still satisfy pigeonhole: with k+1 disjoint
// chunks, a pair within Hamming k must share ALL bits in at least one chunk.
// ---------------------------------------------------------------------------

/// Per-bit entropy on a random sample of rows. Returns Vec<f64> of length
/// `8 * d_bytes`. Bit indexing matches `extract_bits` (MSB-first within each
/// byte).
pub fn estimate_bit_entropies(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    sample_size: usize,
) -> Vec<f64> {
    let d_bits = 8 * d_bytes;
    let m = sample_size.min(n);
    let mut counts = vec![0u64; d_bits];
    // Deterministic stride sampling — no rng dependency, reproducible.
    let stride = if m == 0 { 1 } else { (n / m).max(1) };
    let mut sampled: u64 = 0;
    let mut row = 0;
    while sampled < m as u64 && row < n {
        let r = &data[row * d_bytes..(row + 1) * d_bytes];
        for byte_idx in 0..d_bytes {
            let byte = r[byte_idx];
            for b in 0..8 {
                let bit = (byte >> (7 - b)) & 1;
                if bit == 1 {
                    counts[byte_idx * 8 + b] += 1;
                }
            }
        }
        sampled += 1;
        row += stride;
    }
    let denom = sampled as f64;
    counts
        .iter()
        .map(|&c| {
            if denom == 0.0 { return 0.0; }
            let p = c as f64 / denom;
            if p == 0.0 || p == 1.0 { return 0.0; }
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        })
        .collect()
}

/// Plan `n_chunks` chunks of bit positions, each balanced by summed entropy,
/// each ≤ `max_bits_per_chunk` bits. Only bits with positive entropy are
/// allocated; zero-entropy bits are dropped. Greedy round-robin: sort bits
/// by descending entropy, assign each to the chunk with the smallest current
/// summed entropy (subject to the size cap).
pub fn plan_adaptive_chunks(
    entropies: &[f64],
    n_chunks: usize,
    max_bits_per_chunk: usize,
) -> Vec<Vec<usize>> {
    let mut idxs: Vec<usize> = (0..entropies.len())
        .filter(|&i| entropies[i] > 0.0)
        .collect();
    idxs.sort_by(|&a, &b| entropies[b].partial_cmp(&entropies[a]).unwrap());

    let mut chunks: Vec<Vec<usize>> = (0..n_chunks).map(|_| Vec::new()).collect();
    let mut sums: Vec<f64> = vec![0.0; n_chunks];

    for bit in idxs {
        // Pick chunk with smallest sum that still has capacity.
        let mut best: Option<usize> = None;
        let mut best_sum = f64::INFINITY;
        for c in 0..n_chunks {
            if chunks[c].len() >= max_bits_per_chunk {
                continue;
            }
            if sums[c] < best_sum {
                best_sum = sums[c];
                best = Some(c);
            }
        }
        if let Some(c) = best {
            chunks[c].push(bit);
            sums[c] += entropies[bit];
        } else {
            // All chunks full — drop the bit.
            break;
        }
    }
    // Keep bit positions sorted within each chunk for stable extraction.
    for c in &mut chunks {
        c.sort_unstable();
    }
    chunks
}

/// Extract the bits at the given (sorted) positions from `row`, packed into
/// a u128 with bit i of the output corresponding to position positions[i].
/// Panics if `positions.len() > 128`.
#[inline]
pub fn extract_bits_at(row: &[u8], positions: &[usize]) -> u128 {
    assert!(positions.len() <= 128, "extract_bits_at: too many positions");
    let mut acc: u128 = 0;
    for (out_bit, &p) in positions.iter().enumerate() {
        let byte_idx = p / 8;
        let bit_in_byte = p % 8;
        let bit = (row[byte_idx] >> (7 - bit_in_byte)) & 1;
        acc |= (bit as u128) << out_bit;
    }
    acc
}

fn candidates_for_position_chunk(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    positions: &[usize],
) -> Vec<(u32, u32)> {
    let mut buckets: HashMap<u128, Vec<u32>, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(n, FxBuildHasher);
    for i in 0..n {
        let row = &data[i * d_bytes..i * d_bytes + d_bytes];
        let key = extract_bits_at(row, positions);
        buckets.entry(key).or_insert_with(|| Vec::with_capacity(1)).push(i as u32);
    }
    let mut out: Vec<(u32, u32)> = Vec::new();
    emit_pairs(&buckets, &mut out);
    out
}

/// Entropy-aware adaptive variant of `find_pairs_self_bit`.
///
/// Algorithm: sample bits to estimate per-bit entropies; partition only the
/// positive-entropy bits into k+1 disjoint chunks balanced by entropy; run
/// the standard pigeonhole on those chunks; verify with full popcount so the
/// result is exact.
pub fn find_pairs_self_adaptive(
    data: &[u8],
    n: usize,
    d_bytes: usize,
    k: u32,
) -> Vec<(u32, u32, u32)> {
    assert_eq!(data.len(), n * d_bytes, "data length mismatch");
    assert!(n <= u32::MAX as usize, "n exceeds u32::MAX");

    // At k=0 we want exact duplicates. Adaptive planning provides no benefit
    // (verify must touch every bit anyway) and the single-chunk plan can
    // exceed u128. Delegate to the bit path, which special-cases k=0 safely.
    if k == 0 {
        return find_pairs_self_bit(data, n, d_bytes, k);
    }

    let sample_size = n.min(10_000).max(1);
    let entropies = estimate_bit_entropies(data, n, d_bytes, sample_size);
    let n_chunks = (k + 1) as usize;
    let chunks = plan_adaptive_chunks(&entropies, n_chunks, 128);

    // If any chunk is empty, the plan failed; fall back to bit-level chunks.
    if chunks.iter().any(|c| c.is_empty()) {
        return find_pairs_self_bit(data, n, d_bytes, k);
    }

    let per_chunk: Vec<Vec<(u32, u32)>> = chunks
        .par_iter()
        .map(|positions| candidates_for_position_chunk(data, n, d_bytes, positions))
        .collect();
    let total: usize = per_chunk.iter().map(|v| v.len()).sum();
    if total == 0 { return Vec::new(); }
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(total);
    for mut v in per_chunk { candidates.append(&mut v); }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (candidates.len() / (n_threads * 4)).max(1024);
    let parts: Vec<Vec<(u32, u32, u32)>> = candidates
        .par_chunks(chunk_size)
        .map(|slice| verify_with_prefetch(slice, data, d_bytes, k))
        .collect();
    let mut verified: Vec<(u32, u32, u32)> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for mut p in parts { verified.append(&mut p); }
    verified.par_sort_unstable();
    verified.dedup();
    verified
}

// ---------------------------------------------------------------------------
// Streaming Index (v0.4, Task 2).
//
// State + add/query interface for content-moderation pipelines that maintain a
// growing database of hashes and query each incoming hash against it.
// Bit-level chunks (Task 1) so the Index supports any k < 8*d_bytes.
// Append-only; remove() is deferred to v0.5.
// ---------------------------------------------------------------------------

/// Append-only near-duplicate index over packed binary vectors.
pub struct Index {
    d_bytes: usize,
    k: u32,
    rows: Vec<u8>, // row-major, length n * d_bytes
    n: u32,
    /// One hashmap per bit-chunk. Key = chunk bits packed as u128.
    buckets: Vec<HashMap<u128, Vec<u32>, FxBuildHasher>>,
    chunks_bit: Vec<(usize, usize)>,
}

impl Index {
    pub fn new(d_bytes: usize, k: u32) -> Self {
        let d_bits = 8 * d_bytes;
        assert!(
            (k as usize) < d_bits,
            "Index: k ({}) must be < 8*d_bytes ({})",
            k, d_bits
        );
        let chunks_bit = bit_chunks_for(d_bits, k);
        for &(lo, hi) in &chunks_bit {
            debug_assert!(hi - lo <= 128, "Index: bit chunk wider than 128 bits");
        }
        let buckets: Vec<HashMap<u128, Vec<u32>, FxBuildHasher>> = (0..chunks_bit.len())
            .map(|_| HashMap::with_hasher(FxBuildHasher))
            .collect();
        Self { d_bytes, k, rows: Vec::new(), n: 0, buckets, chunks_bit }
    }

    /// Append one row, return its assigned id.
    pub fn add(&mut self, row: &[u8]) -> u32 {
        assert_eq!(row.len(), self.d_bytes, "Index::add row length mismatch");
        let id = self.n;
        self.rows.extend_from_slice(row);
        self.n += 1;
        for (chunk_pos, &(lo, hi)) in self.chunks_bit.iter().enumerate() {
            let key = extract_bits(row, lo, hi);
            self.buckets[chunk_pos]
                .entry(key)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(id);
        }
        id
    }

    /// Append a batch of rows; return `Some(first_id)` for the first row
    /// added, or `None` if the batch was empty. Distinguishing empty from
    /// "added exactly one row whose id is 0" matters for downstream
    /// pipelines that filter DataFrames into possibly-empty batches.
    pub fn add_batch(&mut self, batch: &[u8]) -> Option<u32> {
        assert_eq!(batch.len() % self.d_bytes, 0, "Index::add_batch length not multiple of d_bytes");
        let m = batch.len() / self.d_bytes;
        if m == 0 {
            return None;
        }
        let first_id = self.n;
        self.rows.reserve(batch.len());
        self.rows.extend_from_slice(batch);
        for i in 0..m {
            let id = self.n;
            let row = &batch[i * self.d_bytes..(i + 1) * self.d_bytes];
            for (chunk_pos, &(lo, hi)) in self.chunks_bit.iter().enumerate() {
                let key = extract_bits(row, lo, hi);
                self.buckets[chunk_pos]
                    .entry(key)
                    .or_insert_with(|| Vec::with_capacity(1))
                    .push(id);
            }
            self.n += 1;
            let _ = id;
        }
        Some(first_id)
    }

    /// Batched query: run `query` for each row in `queries`, in order.
    /// Returns one result vector per input row. Memory-cheap version of
    /// "loop over rows and call query" — avoids re-allocating the
    /// candidate-set HashSet per query.
    pub fn query_batch(&self, queries: &[u8]) -> Vec<Vec<(u32, u32)>> {
        assert_eq!(
            queries.len() % self.d_bytes,
            0,
            "Index::query_batch length not multiple of d_bytes"
        );
        let m = queries.len() / self.d_bytes;
        let mut out: Vec<Vec<(u32, u32)>> = Vec::with_capacity(m);
        let mut seen: hashbrown::HashSet<u32, FxBuildHasher> =
            hashbrown::HashSet::with_hasher(FxBuildHasher);
        for i in 0..m {
            let row = &queries[i * self.d_bytes..(i + 1) * self.d_bytes];
            seen.clear();
            for (chunk_pos, &(lo, hi)) in self.chunks_bit.iter().enumerate() {
                let key = extract_bits(row, lo, hi);
                if let Some(ids) = self.buckets[chunk_pos].get(&key) {
                    for &id in ids { seen.insert(id); }
                }
            }
            let mut hits: Vec<(u32, u32)> = Vec::new();
            for &id in seen.iter() {
                let other = &self.rows[(id as usize) * self.d_bytes..((id as usize) + 1) * self.d_bytes];
                let d = hamming(row, other);
                if d <= self.k {
                    hits.push((id, d));
                }
            }
            hits.sort_unstable();
            out.push(hits);
        }
        out
    }

    /// Return all (other_id, hamming_dist) where hamming(row, rows[other_id]) <= k.
    pub fn query(&self, row: &[u8]) -> Vec<(u32, u32)> {
        assert_eq!(row.len(), self.d_bytes, "Index::query row length mismatch");
        // Collect candidate ids via bit-chunk buckets. Use a HashSet to dedup;
        // for small candidate sets a SmallVec + sort would be cheaper, but
        // HashSet is simpler and robust.
        let mut seen: hashbrown::HashSet<u32, FxBuildHasher> =
            hashbrown::HashSet::with_hasher(FxBuildHasher);
        for (chunk_pos, &(lo, hi)) in self.chunks_bit.iter().enumerate() {
            let key = extract_bits(row, lo, hi);
            if let Some(ids) = self.buckets[chunk_pos].get(&key) {
                for &id in ids {
                    seen.insert(id);
                }
            }
        }
        let k = self.k;
        let d_bytes = self.d_bytes;
        let mut out: Vec<(u32, u32)> = Vec::new();
        for id in seen {
            let other = &self.rows[(id as usize) * d_bytes..((id as usize) + 1) * d_bytes];
            let d = hamming(row, other);
            if d <= k {
                out.push((id, d));
            }
        }
        out.sort_unstable();
        out
    }

    pub fn len(&self) -> usize { self.n as usize }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn d_bytes(&self) -> usize { self.d_bytes }
    pub fn k(&self) -> u32 { self.k }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_match_python_linspace_truncate() {
        assert_eq!(chunk_boundaries(32, 3), vec![(0, 10), (10, 21), (21, 32)]);
        assert_eq!(chunk_boundaries(32, 2), vec![(0, 16), (16, 32)]);
        assert_eq!(chunk_boundaries(16, 4), vec![(0, 4), (4, 8), (8, 12), (12, 16)]);
    }

    #[test]
    fn hamming_le_k_matches_full() {
        // Random equal-length slices; for several k values check the relation.
        let n = 100;
        let d_bytes = 32;
        for seed in [1u64, 2, 3, 4, 5] {
            let data = deterministic_random(n, d_bytes, 0xE0E0 ^ seed);
            for i in 0..(n - 1) {
                for j in (i + 1)..n {
                    let ra = &data[i * d_bytes..(i + 1) * d_bytes];
                    let rb = &data[j * d_bytes..(j + 1) * d_bytes];
                    let d_full = hamming(ra, rb);
                    for k in [0u32, 2, 8, 16, 64, 128, 256] {
                        let r = hamming_le_k(ra, rb, k);
                        if d_full <= k {
                            assert_eq!(r, Some(d_full));
                        } else {
                            assert_eq!(r, None);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hamming_basic() {
        assert_eq!(hamming(&[0u8; 32], &[0u8; 32]), 0);
        assert_eq!(hamming(&[0xFFu8; 32], &[0u8; 32]), 256);
        let a = [0b1010_1010u8; 32];
        let b = [0b0101_0101u8; 32];
        assert_eq!(hamming(&a, &b), 256);
    }

    #[test]
    fn finds_identical_pairs() {
        let mut data = vec![0u8; 4 * 32];
        for i in 0..32 {
            data[0 * 32 + i] = (i as u8).wrapping_mul(17);
            data[1 * 32 + i] = (i as u8).wrapping_mul(31).wrapping_add(7);
            data[2 * 32 + i] = (i as u8).wrapping_mul(17);
            data[3 * 32 + i] = (i as u8).wrapping_mul(53).wrapping_add(99);
        }
        let pairs = find_pairs_self(&data, 4, 32, 2);
        assert_eq!(pairs, vec![(0, 2, 0)]);
        let pairs_st = find_pairs_self_st(&data, 4, 32, 2);
        assert_eq!(pairs_st, vec![(0, 2, 0)]);
    }

    #[test]
    fn one_bit_flip_found() {
        let mut data = vec![0u8; 3 * 32];
        for i in 0..32 {
            data[0 * 32 + i] = (i as u8).wrapping_mul(17);
            data[1 * 32 + i] = (i as u8).wrapping_mul(17);
        }
        data[1 * 32 + 5] ^= 0b0000_0001;
        let pairs = find_pairs_self(&data, 3, 32, 2);
        assert_eq!(pairs, vec![(0, 1, 1)]);
    }

    #[test]
    fn parallel_matches_st() {
        // Build a small random-ish corpus and assert par == st.
        let n = 500usize;
        let d_bytes = 32usize;
        let mut data = vec![0u8; n * d_bytes];
        for i in 0..(n * d_bytes) {
            data[i] = ((i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493) >> 32) as u8;
        }
        // Inject a few near-duplicates.
        for r in 1..5 {
            let src = (r * 97) % n;
            let dst = (r * 199) % n;
            if src != dst {
                let (a, b) = data.split_at_mut(dst.max(src) * d_bytes);
                let (lo_row, hi_row) = if src < dst {
                    (&a[src * d_bytes..(src + 1) * d_bytes], &mut b[..d_bytes])
                } else {
                    (&b[..d_bytes], &mut a[dst * d_bytes..(dst + 1) * d_bytes])
                };
                hi_row.copy_from_slice(lo_row);
            }
        }
        let mut a = find_pairs_self_st(&data, n, d_bytes, 2);
        let mut b = find_pairs_self_par(&data, n, d_bytes, 2);
        let mut c = find_pairs_self_par_prefetch(&data, n, d_bytes, 2);
        a.sort_unstable();
        b.sort_unstable();
        c.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    fn deterministic_random(n: usize, d_bytes: usize, seed: u64) -> Vec<u8> {
        let mut data = vec![0u8; n * d_bytes];
        let mut s = seed;
        for byte in data.iter_mut() {
            // xorshift64*
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *byte = (s.wrapping_mul(2685821657736338717) >> 56) as u8;
        }
        data
    }

    fn pair_set(v: &[(u32, u32, u32)]) -> std::collections::BTreeSet<(u32, u32, u32)> {
        v.iter().copied().collect()
    }

    #[test]
    fn qgram_matches_chunk_k8_q12() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xDEADBEEF);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 8));
        let b = pair_set(&find_pairs_qgram_self(&data, n, d_bytes, 8, 12));
        assert_eq!(a, b, "qgram differs from chunk at k=8 q=12");
    }

    #[test]
    fn qgram_matches_chunk_k2_q16() {
        let n = 200;
        let d_bytes = 32;
        // Inject some near-duplicates so pair set is non-empty.
        let mut data = deterministic_random(n, d_bytes, 0xCAFE);
        for r in 1..6 {
            let src = (r * 31) % n;
            let dst = (r * 71) % n;
            if src != dst {
                let (lo_r, hi_r) = if src < dst { (src, dst) } else { (dst, src) };
                let (left, right) = data.split_at_mut(hi_r * d_bytes);
                let src_row = &left[lo_r * d_bytes..(lo_r + 1) * d_bytes].to_vec();
                right[..d_bytes].copy_from_slice(src_row);
                // flip one bit
                right[3] ^= 0b0000_0010;
            }
        }
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 2));
        let b = pair_set(&find_pairs_qgram_self(&data, n, d_bytes, 2, 16));
        assert_eq!(a, b, "qgram differs from chunk at k=2 q=16");
    }

    #[test]
    fn radix_matches_chunk_k2() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xAA01);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 2));
        let b = pair_set(&find_pairs_self_radix(&data, n, d_bytes, 2));
        assert_eq!(a, b);
    }

    #[test]
    fn radix_matches_chunk_k4() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xAA02);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 4));
        let b = pair_set(&find_pairs_self_radix(&data, n, d_bytes, 4));
        assert_eq!(a, b);
    }

    #[test]
    fn radix_matches_chunk_k16() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xAA03);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 16));
        let b = pair_set(&find_pairs_self_radix(&data, n, d_bytes, 16));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_matches_chunk_k2() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB01);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 2));
        let b = pair_set(&find_pairs_self_no_dedup(&data, n, d_bytes, 2));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_matches_chunk_k4() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB02);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 4));
        let b = pair_set(&find_pairs_self_no_dedup(&data, n, d_bytes, 4));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_matches_chunk_k16() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB03);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 16));
        let b = pair_set(&find_pairs_self_no_dedup(&data, n, d_bytes, 16));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_par_matches_chunk_k2() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB11);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 2));
        let b = pair_set(&find_pairs_self_no_dedup_par_prefetch(&data, n, d_bytes, 2));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_par_matches_chunk_k8() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB12);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 8));
        let b = pair_set(&find_pairs_self_no_dedup_par_prefetch(&data, n, d_bytes, 8));
        assert_eq!(a, b);
    }

    #[test]
    fn no_dedup_par_matches_chunk_k16() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xBB13);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 16));
        let b = pair_set(&find_pairs_self_no_dedup_par_prefetch(&data, n, d_bytes, 16));
        assert_eq!(a, b);
    }

    #[test]
    fn cross_matches_brute_force() {
        let n_a = 50;
        let n_b = 60;
        let d_bytes = 32;
        let a_data = deterministic_random(n_a, d_bytes, 0xC101);
        let mut b_data = deterministic_random(n_b, d_bytes, 0xC102);
        // Plant ~5 near-duplicates of A[0..5] into B[0..5] (1 bit flip each).
        for r in 0..5 {
            let src = &a_data[r * d_bytes..(r + 1) * d_bytes].to_vec();
            b_data[r * d_bytes..(r + 1) * d_bytes].copy_from_slice(src);
            b_data[r * d_bytes + (r % d_bytes)] ^= 1 << (r % 8);
        }
        // Brute force reference at k=2.
        let mut expected: Vec<(u32, u32, u32)> = Vec::new();
        for i in 0..n_a {
            for j in 0..n_b {
                let ra = &a_data[i * d_bytes..(i + 1) * d_bytes];
                let rb = &b_data[j * d_bytes..(j + 1) * d_bytes];
                let d = hamming(ra, rb);
                if d <= 2 {
                    expected.push((i as u32, j as u32, d));
                }
            }
        }
        let mut got = find_pairs_cross(&a_data, n_a, &b_data, n_b, d_bytes, 2);
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn bit_boundaries_basic() {
        // d_bits=64, k+1=3 chunks: should be [0,21) [21,42) [42,64)
        assert_eq!(chunk_bit_boundaries(64, 3), vec![(0, 21), (21, 42), (42, 64)]);
        assert_eq!(chunk_bit_boundaries(256, 4), vec![(0, 64), (64, 128), (128, 192), (192, 256)]);
        // d_bits=64, 9 chunks (e.g. k=8).
        let c = chunk_bit_boundaries(64, 9);
        assert_eq!(c.len(), 9);
        assert_eq!(c[0].0, 0);
        assert_eq!(c[8].1, 64);
        // disjoint, contiguous, total = d_bits.
        let total: usize = c.iter().map(|(lo, hi)| hi - lo).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn extract_bits_known() {
        // The exact packing of bits into the u128 result is an internal
        // contract: bytes-aligned extracts return the original byte value, and
        // ranges that span bytes pack the more-significant (lower-index) chunk
        // into the low bits of the next byte of the result. What matters is
        // that the function is a deterministic, collision-free mapping from
        // bit-range contents to u128. These tests pin the current contract.
        let row = [0b1111_0000u8, 0b0000_1111u8];
        // Bits [4, 12) — low nibble of byte 0 (= 0000) + high nibble of byte 1 (= 0000) = 0.
        assert_eq!(extract_bits(&row, 4, 12), 0);
        // Bits [0, 4) — high nibble of byte 0 = 0b1111 -> 0x0F.
        assert_eq!(extract_bits(&row, 0, 4), 0x0F);
        // Bits [12, 16) — low nibble of byte 1 = 0b1111 -> 0x0F.
        assert_eq!(extract_bits(&row, 12, 16), 0x0F);
        // Zero width.
        assert_eq!(extract_bits(&row, 4, 4), 0);
        // Cross-byte 4-bit fetch on zero bits.
        assert_eq!(extract_bits(&row, 6, 10), 0);
        // Whole-byte extracts return the byte itself.
        let row2 = [0xABu8, 0xCDu8];
        assert_eq!(extract_bits(&row2, 0, 8), 0xAB);
        assert_eq!(extract_bits(&row2, 8, 16), 0xCD);
        // 16-bit extract: bytes are packed in the "next-chunk goes to higher bits"
        // direction. With the current implementation, the result is
        // (byte0_chunk) | (byte1_chunk << 8) = 0xAB | (0xCD << 8) = 0xCDAB.
        assert_eq!(extract_bits(&row2, 0, 16), 0xCDAB);
        // Different rows that share the bit range produce the same key (the
        // critical invariant for chunk-bucketing).
        let row_a = [0xFFu8, 0x00u8];
        let row_b = [0xFFu8, 0xFFu8];
        assert_eq!(extract_bits(&row_a, 0, 8), extract_bits(&row_b, 0, 8));
        // Different bit contents produce different keys.
        assert_ne!(extract_bits(&row_a, 0, 16), extract_bits(&row_b, 0, 16));
    }

    #[test]
    fn bit_matches_byte_for_supported() {
        // For k < d_bytes the bit-level path should be a superset (and actually equal)
        // of the byte-level set in this regime, since both compute exact pairs.
        let n = 200;
        let d_bytes = 32;
        for &kk in &[2u32, 4, 8] {
            let data = deterministic_random(n, d_bytes, 0xB17 ^ (kk as u64));
            let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, kk));
            let b = pair_set(&find_pairs_self_bit(&data, n, d_bytes, kk));
            assert_eq!(a, b, "bit vs byte mismatch at k={}", kk);
        }
    }

    /// Regression: wide d (d>=512) panicked on the bit / Index paths at small
    /// k because chunk_bit_boundaries(d_bits, k+1) produced chunks > 128 bits.
    /// `bit_chunks_for` now picks max(k+1, ceil(d_bits/128)) chunks so the
    /// u128 chunk-key invariant always holds.
    #[test]
    fn bit_path_wide_d_no_panic() {
        // d=512 k=2 → was 3 chunks of 170 bits each; needs 4 chunks now.
        let n = 100;
        for &d_bytes in &[64usize, 128] {  // 512, 1024 bits
            for &k in &[0u32, 2, 4, 8] {
                let mut data = deterministic_random(n, d_bytes, 0xD512 + k as u64);
                // Plant 2 exact-duplicate pairs
                let src_row = data[0..d_bytes].to_vec();
                data[1 * d_bytes..2 * d_bytes].copy_from_slice(&src_row);
                let src_row = data[10 * d_bytes..11 * d_bytes].to_vec();
                data[11 * d_bytes..12 * d_bytes].copy_from_slice(&src_row);

                let from_st  = pair_set(&find_pairs_self_st (&data, n, d_bytes, k));
                let from_bit = pair_set(&find_pairs_self_bit(&data, n, d_bytes, k));
                assert_eq!(from_st, from_bit,
                           "wide-d bit path disagrees with byte at d_bytes={}, k={}",
                           d_bytes, k);

                // Index should also work at wide d
                let mut idx = Index::new(d_bytes, k);
                idx.add_batch(&data);
                let q = data[0..d_bytes].to_vec();
                let _hits = idx.query(&q);  // just verify no panic
            }
        }
    }

    /// Regression: `k=0` previously panicked on bit / adaptive / Index paths
    /// when `d_bytes >= 17`, because the single chunk would span all d_bits
    /// and exceed the u128 chunk-key width assertion. At k=0 we now use a
    /// 128-bit prefix as a sound prefilter and rely on popcount verify for
    /// exactness.
    #[test]
    fn k_zero_no_panic_and_finds_exact_dupes() {
        let n = 50;
        let d_bytes = 32; // d_bits = 256, triggers the old panic
        let mut data = deterministic_random(n, d_bytes, 0xDEADBEEF);
        // Plant 3 exact-duplicate pairs.
        for r in 0..3 {
            let src_row = data[r * 2 * d_bytes..(r * 2 + 1) * d_bytes].to_vec();
            data[(r * 2 + 1) * d_bytes..(r * 2 + 2) * d_bytes].copy_from_slice(&src_row);
        }

        let from_st       = pair_set(&find_pairs_self_st       (&data, n, d_bytes, 0));
        let from_bit      = pair_set(&find_pairs_self_bit      (&data, n, d_bytes, 0));
        let from_adaptive = pair_set(&find_pairs_self_adaptive (&data, n, d_bytes, 0));
        assert_eq!(from_st, from_bit,      "k=0: bit path disagrees with byte path");
        assert_eq!(from_st, from_adaptive, "k=0: adaptive path disagrees with byte path");
        assert!(from_st.len() >= 3, "should find at least the 3 planted dupes");

        // Index at k=0 with d_bytes=32 also must not panic.
        let mut idx = Index::new(d_bytes, 0);
        idx.add_batch(&data);
        // Query the first planted-pair source row; should find it AND its dup.
        let q = data[0..d_bytes].to_vec();
        let hits = idx.query(&q);
        assert!(hits.iter().any(|(id, d)| *id == 0 && *d == 0));
        assert!(hits.iter().any(|(id, d)| *id == 1 && *d == 0));
    }

    /// Cross-join bit-level: closes the same `k >= d_bytes` correctness gap
    /// that bit-level self-join closed in Round 5.
    #[test]
    fn cross_bit_finds_phash_regime_pairs_byte_misses() {
        let n_a = 30usize;
        let n_b = 30usize;
        let d_bytes = 8;
        let mut a = deterministic_random(n_a, d_bytes, 0xA0A0);
        let mut b = deterministic_random(n_b, d_bytes, 0xB0B0);
        // Plant 5 cross pairs at exactly Hamming 8: b[i] = a[i] with one bit flipped per byte.
        for i in 0..5 {
            let src = a[i * d_bytes..(i + 1) * d_bytes].to_vec();
            b[i * d_bytes..(i + 1) * d_bytes].copy_from_slice(&src);
            for byte in 0..d_bytes {
                b[i * d_bytes + byte] ^= 1u8 << (byte % 8);
            }
        }
        // Ground truth via O(n_a * n_b) brute force.
        let mut truth: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for i in 0..n_a {
            for j in 0..n_b {
                let ra = &a[i * d_bytes..(i + 1) * d_bytes];
                let rb = &b[j * d_bytes..(j + 1) * d_bytes];
                if hamming(ra, rb) <= 8 {
                    truth.insert((i as u32, j as u32));
                }
            }
        }
        let from_bit: std::collections::HashSet<(u32, u32)> =
            find_pairs_cross_bit(&a, n_a, &b, n_b, d_bytes, 8)
                .into_iter().map(|(i, j, _)| (i, j)).collect();
        assert_eq!(from_bit, truth, "cross_bit must match brute force at k=d_bytes");
        assert!(from_bit.len() >= 5, "should find at least the 5 planted pairs");
    }

    #[test]
    fn bit_finds_phash_regime_pairs_byte_misses() {
        // pHash regime: d_bytes=8, k=8. Byte-aligned path can miss pairs;
        // bit-level path is correct.
        let n = 200;
        let d_bytes = 8;
        let mut data = deterministic_random(n, d_bytes, 0xCAFEFACE);
        // Plant 10 pairs with exactly Hamming distance 8: one bit flip per byte.
        for r in 0..10 {
            let src = r * 2;
            let dst = r * 2 + 1;
            let src_row = data[src * d_bytes..(src + 1) * d_bytes].to_vec();
            data[dst * d_bytes..(dst + 1) * d_bytes].copy_from_slice(&src_row);
            // Flip one bit in each of the 8 bytes.
            for b in 0..d_bytes {
                data[dst * d_bytes + b] ^= 1u8 << (b % 8);
            }
        }
        // Brute force ground truth at k=8.
        let mut expected: Vec<(u32, u32, u32)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let ra = &data[i * d_bytes..(i + 1) * d_bytes];
                let rb = &data[j * d_bytes..(j + 1) * d_bytes];
                let d = hamming(ra, rb);
                if d <= 8 {
                    expected.push((i as u32, j as u32, d));
                }
            }
        }
        let expected_set: std::collections::BTreeSet<_> = expected.iter().copied().collect();
        let bit_pairs = pair_set(&find_pairs_self_bit(&data, n, d_bytes, 8));
        assert_eq!(bit_pairs, expected_set, "bit-level missed pairs in pHash regime");
        // The byte-aligned version (no_dedup_par_prefetch) at k=8, d=8 only has
        // d_bytes=8 chunks but pigeonhole needs k+1=9 — so chunk_boundaries returns
        // 8 single-byte chunks and a pair with one bit flipped in every byte
        // CAN be missed by the chunk filter.
        let byte_pairs = pair_set(&find_pairs_self_no_dedup_par_prefetch(&data, n, d_bytes, 8));
        // Bit version must be a superset of byte version.
        assert!(byte_pairs.is_subset(&bit_pairs));
        // And here it should strictly differ (we planted such pairs).
        assert_ne!(bit_pairs, byte_pairs, "expected byte-level to miss at least one planted pair");
    }

    #[test]
    fn index_basic_near_dup_recall() {
        // d=32, k=2. Add 100 random rows, then 50 near-duplicates (1 bit flip).
        // For each near-duplicate query, expect to find the originating row.
        let n = 100usize;
        let d_bytes = 32usize;
        let data = deterministic_random(n, d_bytes, 0xDEADBEEF);
        let mut idx = Index::new(d_bytes, 2);
        for i in 0..n {
            let id = idx.add(&data[i * d_bytes..(i + 1) * d_bytes]);
            assert_eq!(id as usize, i);
        }
        assert_eq!(idx.len(), n);
        // 50 near-duplicates of rows 0..50.
        let mut near = vec![0u8; 50 * d_bytes];
        for i in 0..50 {
            let src = &data[i * d_bytes..(i + 1) * d_bytes];
            near[i * d_bytes..(i + 1) * d_bytes].copy_from_slice(src);
            // Flip one bit deterministically.
            near[i * d_bytes + (i % d_bytes)] ^= 1 << (i % 8);
        }
        // Query each near-dup; expect at least the corresponding original.
        for i in 0..50 {
            let q = &near[i * d_bytes..(i + 1) * d_bytes];
            let hits = idx.query(q);
            let found = hits.iter().any(|&(id, _)| id as usize == i);
            assert!(found, "near-dup {} did not find original", i);
        }
    }

    #[test]
    fn index_matches_find_pairs_self_bit() {
        // Build the index over all rows, then query with each row. The union
        // of query results (i < j only) should equal the pair set from
        // find_pairs_self_bit on the same data (modulo self-reflexivity).
        let n = 150usize;
        let d_bytes = 32usize;
        let mut data = deterministic_random(n, d_bytes, 0x1234_5678_9ABC_DEF0);
        // Inject near-duplicates.
        for r in 1..6 {
            let src = (r * 13) % n;
            let dst = (r * 41) % n;
            if src != dst {
                let (lo_r, hi_r) = if src < dst { (src, dst) } else { (dst, src) };
                let (left, right) = data.split_at_mut(hi_r * d_bytes);
                let src_row = left[lo_r * d_bytes..(lo_r + 1) * d_bytes].to_vec();
                right[..d_bytes].copy_from_slice(&src_row);
                right[r] ^= 1 << (r % 8);
            }
        }
        let mut idx = Index::new(d_bytes, 2);
        idx.add_batch(&data);
        // Build pair set via query.
        let mut from_idx: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
        for i in 0..n {
            let row = &data[i * d_bytes..(i + 1) * d_bytes];
            for (j, d) in idx.query(row) {
                if (j as usize) == i { continue; }
                let (a, b) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
                from_idx.insert((a, b, d));
            }
        }
        let from_bit: std::collections::BTreeSet<_> = find_pairs_self_bit(&data, n, d_bytes, 2).into_iter().collect();
        assert_eq!(from_idx, from_bit);
    }

    /// Empty batch must return None; non-empty must return Some(first_id).
    /// Previously add_batch returned u32 even on empty input, indistinguishable
    /// from "added one row whose id happens to be 0".
    #[test]
    fn index_add_batch_empty_returns_none() {
        let d_bytes = 8usize;
        let mut idx = Index::new(d_bytes, 2);
        assert_eq!(idx.add_batch(&[]), None);
        assert_eq!(idx.len(), 0);
        // After adding one row, first_id is the previous n.
        let one_row = vec![0xAAu8; d_bytes];
        assert_eq!(idx.add_batch(&one_row), Some(0));
        assert_eq!(idx.len(), 1);
        // Another empty batch still returns None.
        assert_eq!(idx.add_batch(&[]), None);
        assert_eq!(idx.len(), 1);
    }

    /// query_batch result must equal looping query for every row.
    #[test]
    fn index_query_batch_matches_per_row_query() {
        let n = 60usize;
        let d_bytes = 16usize;
        let data = deterministic_random(n, d_bytes, 0xB070);
        let mut idx = Index::new(d_bytes, 3);
        idx.add_batch(&data);
        // Build a query batch: 20 rows from the corpus + 20 noise rows.
        let mut queries: Vec<u8> = Vec::new();
        queries.extend_from_slice(&data[..20 * d_bytes]);
        queries.extend_from_slice(&deterministic_random(20, d_bytes, 0xB071));
        let per_row: Vec<Vec<(u32, u32)>> = (0..40)
            .map(|i| idx.query(&queries[i * d_bytes..(i + 1) * d_bytes]))
            .collect();
        let batched = idx.query_batch(&queries);
        assert_eq!(per_row, batched);
        // Also empty batch should give empty Vec.
        assert!(idx.query_batch(&[]).is_empty());
    }

    #[test]
    fn index_add_batch_consistent() {
        let n = 80usize;
        let d_bytes = 16usize;
        let data = deterministic_random(n, d_bytes, 0xBA77E);
        let mut a = Index::new(d_bytes, 4);
        let mut b = Index::new(d_bytes, 4);
        for i in 0..n {
            a.add(&data[i * d_bytes..(i + 1) * d_bytes]);
        }
        b.add_batch(&data);
        assert_eq!(a.len(), b.len());
        // Query a random row in both; should match.
        let q = &data[3 * d_bytes..4 * d_bytes];
        assert_eq!(a.query(q), b.query(q));
    }

    #[test]
    fn adaptive_drops_zero_entropy_bits() {
        // Synthetic skew: bits 0..127 uniform, bits 128..255 constant (all zero).
        let n = 500;
        let d_bytes = 32;
        let mut data = vec![0u8; n * d_bytes];
        // Fill first 16 bytes (bits 0..128) with deterministic random data.
        let rand = deterministic_random(n, 16, 0xADAFADA0);
        for i in 0..n {
            data[i * d_bytes..i * d_bytes + 16].copy_from_slice(&rand[i * 16..(i + 1) * 16]);
            // Last 16 bytes stay zero.
        }
        let entropies = estimate_bit_entropies(&data, n, d_bytes, n);
        // Bits 0..128 should mostly have entropy > 0; bits 128..256 should be 0.
        let high: usize = entropies[..128].iter().filter(|&&h| h > 0.5).count();
        let low: usize = entropies[128..].iter().filter(|&&h| h > 0.0).count();
        assert!(high > 100, "expected most of bits 0..128 to be high entropy, got {}", high);
        assert_eq!(low, 0, "zero-entropy bits 128..256 must have entropy 0");
        // Plan k+1=3 chunks for k=2. All positions must come from bits 0..128.
        let chunks = plan_adaptive_chunks(&entropies, 3, 128);
        for chunk in &chunks {
            for &bit in chunk {
                assert!(bit < 128, "adaptive chunk used a zero-entropy bit {}", bit);
            }
        }
        // Correctness vs find_pairs_self_st on this data.
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 2));
        let b = pair_set(&find_pairs_self_adaptive(&data, n, d_bytes, 2));
        assert_eq!(a, b);
    }

    #[test]
    fn adaptive_matches_st_random() {
        // Uniform random data: adaptive should be set-equal to find_pairs_self_st.
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xA01);
        for &kk in &[2u32, 4, 8] {
            let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, kk));
            let b = pair_set(&find_pairs_self_adaptive(&data, n, d_bytes, kk));
            assert_eq!(a, b, "adaptive mismatch at k={}", kk);
        }
    }

    #[test]
    fn qgram_matches_chunk_k16_q12() {
        let n = 200;
        let d_bytes = 32;
        let data = deterministic_random(n, d_bytes, 0xF00D);
        let a = pair_set(&find_pairs_self_st(&data, n, d_bytes, 16));
        let b = pair_set(&find_pairs_qgram_self(&data, n, d_bytes, 16, 12));
        assert_eq!(a, b, "qgram differs from chunk at k=16 q=12");
    }
}
