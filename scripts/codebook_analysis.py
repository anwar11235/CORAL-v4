"""CORAL v4 — Phase 2 codebook / representation analysis script.

Loads .npz produced by collect_states.py and performs:
  a) Per-head k-means clustering (dim split into H=8 groups of 64-d)
  b) Whole-vector k-means on full 512-d states
  c) Bypass accuracy: replace state with nearest centroid → majority-vote digit → accuracy
  d) Codebook-to-accuracy curve: bypass accuracy vs k
  e) t-SNE visualisation coloured by ground-truth digit

Dependencies: sklearn, matplotlib

Usage:
    python scripts/codebook_analysis.py \\
        --states states_phase1.npz \\
        --output-dir analysis/phase2/ \\
        --segment-idx -1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# K-means helpers (sklearn)
# ---------------------------------------------------------------------------

def _run_kmeans(X: np.ndarray, k: int, n_init: int = 3, max_iter: int = 200):
    """Run k-means; returns (labels [N], centroids [k, d], inertia float)."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=42)
    km.fit(X)
    return km.labels_, km.cluster_centers_, km.inertia_


def _cluster_purity(cluster_labels: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Fraction of states whose cluster's majority digit matches their true digit."""
    correct = 0
    total = 0
    for c in range(k):
        mask = cluster_labels == c
        if not mask.any():
            continue
        digits = true_labels[mask]
        majority = np.bincount(digits, minlength=10).argmax()
        correct += (digits == majority).sum()
        total += mask.sum()
    return correct / total if total > 0 else 0.0


def _bypass_accuracy(cluster_labels: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Bypass accuracy: each cluster's majority digit is its 'prediction'.

    Same as cluster purity when majority-voting.
    """
    return _cluster_purity(cluster_labels, true_labels, k)


def _codebook_perplexity(cluster_labels: np.ndarray, k: int) -> float:
    """Effective codebook usage: exp(entropy of cluster usage distribution)."""
    counts = np.bincount(cluster_labels, minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def per_head_analysis(
    empty_states: np.ndarray,
    empty_labels: np.ndarray,
    n_heads: int = 8,
    k_values: tuple = (16, 32, 64, 128),
) -> dict:
    """Run k-means on each of H equal-dimension head slices.

    Args:
        empty_states: [E, 512] float32 states for empty cells only.
        empty_labels: [E]      int digit labels (1-9).
        n_heads:      Number of head splits.
        k_values:     Codebook sizes to evaluate.

    Returns:
        Dict keyed by k → list (len n_heads) of {inertia, purity, perplexity}.
    """
    E, d = empty_states.shape
    d_head = d // n_heads
    results = {}

    for k in k_values:
        head_results = []
        for h in range(n_heads):
            X_h = empty_states[:, h * d_head:(h + 1) * d_head]
            labels_h, _, inertia = _run_kmeans(X_h, k)
            purity = _cluster_purity(labels_h, empty_labels, k)
            perplexity = _codebook_perplexity(labels_h, k)
            head_results.append({
                "inertia": float(inertia),
                "purity": float(purity),
                "perplexity": float(perplexity),
            })
        results[k] = head_results

    return results


def whole_vector_analysis(
    empty_states: np.ndarray,
    empty_labels: np.ndarray,
    k_values: tuple = (64, 128, 256, 512),
) -> dict:
    """Run k-means on the full-dimensional state vectors.

    Returns:
        Dict keyed by k → {inertia, purity, bypass_accuracy, perplexity}.
    """
    results = {}
    for k in k_values:
        # Clip k if not enough samples
        actual_k = min(k, empty_states.shape[0])
        labels_c, _, inertia = _run_kmeans(empty_states, actual_k)
        purity = _cluster_purity(labels_c, empty_labels, actual_k)
        bypass_acc = _bypass_accuracy(labels_c, empty_labels, actual_k)
        perplexity = _codebook_perplexity(labels_c, actual_k)
        results[k] = {
            "k_actual": actual_k,
            "inertia": float(inertia),
            "purity": float(purity),
            "bypass_accuracy": float(bypass_acc),
            "perplexity": float(perplexity),
        }
    return results


def tsne_visualisation(
    empty_states: np.ndarray,
    empty_labels: np.ndarray,
    output_path: str,
    sample_size: int = 5000,
    perplexity: int = 30,
) -> None:
    """Run t-SNE and save a scatter plot coloured by digit.

    Args:
        empty_states: [E, d] states.
        empty_labels: [E]    digit labels.
        output_path:  Where to save the PNG.
        sample_size:  Max states to include in t-SNE.
        perplexity:   t-SNE perplexity.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  [SKIP] t-SNE requires sklearn and matplotlib. Install with: pip install scikit-learn matplotlib")
        return

    E = empty_states.shape[0]
    n = min(sample_size, E)
    idx = np.random.default_rng(42).choice(E, size=n, replace=False)
    X = empty_states[idx].astype(np.float32)
    y = empty_labels[idx]

    print(f"  Running t-SNE on {n} states (perplexity={perplexity}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.get_cmap("tab10", 9)
    for digit in range(1, 10):
        mask = y == digit
        if not mask.any():
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors(digit - 1)],
            s=2, alpha=0.5, label=str(digit),
        )
    ax.legend(title="Digit", markerscale=4, loc="best", fontsize=8)
    ax.set_title("t-SNE of CORAL reasoning states (empty cells, coloured by digit)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  t-SNE saved to {output_path}")


def bypass_accuracy_curve(
    whole_results: dict,
    output_path: str,
) -> None:
    """Plot bypass accuracy vs k and save as PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] Plot requires matplotlib.")
        return

    ks = sorted(whole_results.keys())
    accs = [whole_results[k]["bypass_accuracy"] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, accs, marker="o", linewidth=2)
    ax.set_xlabel("k (codebook size)")
    ax.set_ylabel("Bypass accuracy")
    ax.set_title("Bypass accuracy vs codebook size (whole-vector k-means)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Bypass accuracy curve saved to {output_path}")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(per_head_results: dict, whole_results: dict) -> None:
    """Print a formatted summary of k-means analysis results."""
    print("\n" + "=" * 70)
    print("PER-HEAD K-MEANS ANALYSIS")
    print("=" * 70)
    for k in sorted(per_head_results.keys()):
        head_data = per_head_results[k]
        mean_purity = np.mean([h["purity"] for h in head_data])
        mean_perp = np.mean([h["perplexity"] for h in head_data])
        print(f"  k={k:4d}  mean_purity={mean_purity:.3f}  mean_perplexity={mean_perp:.1f}")
        for h, hd in enumerate(head_data):
            print(f"         head {h}: purity={hd['purity']:.3f}  perplexity={hd['perplexity']:.1f}  inertia={hd['inertia']:.1f}")

    print("\n" + "=" * 70)
    print("WHOLE-VECTOR K-MEANS ANALYSIS")
    print("=" * 70)
    for k in sorted(whole_results.keys()):
        r = whole_results[k]
        print(
            f"  k={k:4d}  bypass_acc={r['bypass_accuracy']:.3f}  "
            f"purity={r['purity']:.3f}  perplexity={r['perplexity']:.1f}  "
            f"inertia={r['inertia']:.1f}"
        )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 codebook and representation analysis")
    parser.add_argument("--states", required=True, help="Path to .npz from collect_states.py")
    parser.add_argument("--output-dir", required=True, help="Directory to save figures and report")
    parser.add_argument(
        "--segment-idx", type=int, default=-1,
        help="Which collected segment to analyse (default: -1 = last)"
    )
    parser.add_argument("--n-heads", type=int, default=8, help="Number of per-head splits")
    parser.add_argument("--tsne-sample", type=int, default=5000, help="States to sample for t-SNE")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading states from {args.states} ...")
    data = np.load(args.states)
    states_all = data["states"]      # [N, 81, n_segs, d]
    labels_all = data["labels"]      # [N, 81]
    given_mask = data["given_mask"]  # [N, 81]
    seg_indices = data["segment_indices"].tolist()

    N, L, n_segs, d = states_all.shape
    seg_col = args.segment_idx % n_segs
    seg_name = seg_indices[seg_col]
    print(f"Analysing segment index {seg_name} (column {seg_col} of {n_segs})")

    # Select the chosen segment's states
    states = states_all[:, :, seg_col, :]  # [N, 81, d]

    # Filter to empty cells only
    empty_mask = ~given_mask.astype(bool)  # [N, 81] True = empty
    states_flat = states.reshape(N * L, d)
    labels_flat = labels_all.reshape(N * L)
    empty_flat = empty_mask.reshape(N * L)

    empty_states = states_flat[empty_flat].astype(np.float32)
    empty_labels = labels_flat[empty_flat]

    print(f"Empty cells: {empty_states.shape[0]} / {N * L} total")

    # Clip digit labels to 1-9 (remove -100 or 0 if any)
    valid = (empty_labels >= 1) & (empty_labels <= 9)
    empty_states = empty_states[valid]
    empty_labels = empty_labels[valid].astype(np.int32)

    print(f"Valid-label empty cells: {empty_states.shape[0]}")

    # a + c) Per-head k-means
    print("\nRunning per-head k-means ...")
    per_head_results = per_head_analysis(
        empty_states, empty_labels,
        n_heads=args.n_heads,
        k_values=(16, 32, 64, 128),
    )

    # b + c) Whole-vector k-means
    print("Running whole-vector k-means ...")
    whole_results = whole_vector_analysis(
        empty_states, empty_labels,
        k_values=(64, 128, 256, 512),
    )

    # Print report
    print_report(per_head_results, whole_results)

    # d) Bypass accuracy curve
    bypass_accuracy_curve(
        whole_results,
        output_path=os.path.join(args.output_dir, f"bypass_accuracy_seg{seg_name}.png"),
    )

    # e) t-SNE
    tsne_visualisation(
        empty_states, empty_labels,
        output_path=os.path.join(args.output_dir, f"tsne_seg{seg_name}.png"),
        sample_size=args.tsne_sample,
    )

    # Save summary as .npz for later reference
    summary_path = os.path.join(args.output_dir, f"analysis_seg{seg_name}.npz")
    np.savez(
        summary_path,
        whole_k=np.array(sorted(whole_results.keys())),
        whole_bypass_acc=np.array([whole_results[k]["bypass_accuracy"] for k in sorted(whole_results.keys())]),
        whole_purity=np.array([whole_results[k]["purity"] for k in sorted(whole_results.keys())]),
        whole_perplexity=np.array([whole_results[k]["perplexity"] for k in sorted(whole_results.keys())]),
    )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
