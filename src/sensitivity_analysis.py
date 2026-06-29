"""
sensitivity_analysis.py
========================
Weight sensitivity analysis for semi-supervised BERTopic.

Expects data to already be loaded and passed in — all loading and
transformation happens in the calling script.

Usage (from another script):
    from sensitivity_analysis import run_sensitivity_analysis

    results = run_sensitivity_analysis(
        merged_all = merged_all,   # pd.DataFrame
        configs    = configs,      # dict: weight_label → config dict
        output_dir = "results/",   # str or Path
    )

merged_all columns required:
    document_index, topic_01, topic_015, topic_02, topic_025, topic_03

configs dict structure (one entry per weight):
    configs["0.1"] = {
        "words": { "0": [{"word": ..., "score": ...}, ...], ... },
        "info":  pd.DataFrame,   # topic_info.csv content (or None)
        "doc":   pd.DataFrame,   # document_topics.csv content
    }

Returns a dict with four DataFrames:
    { "aggregate", "flow", "entropy", "jaccard" }
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS = ["0.1", "0.15", "0.2", "0.25", "0.3"]

COL_MAP = {
    "0.1":  "topic_01",
    "0.15": "topic_015",
    "0.2":  "topic_02",
    "0.25": "topic_025",
    "0.3":  "topic_03",
}

ADJACENT = [(WEIGHTS[i], WEIGHTS[i + 1]) for i in range(len(WEIGHTS) - 1)]

TOP_N_WORDS    = 15
JACCARD_THRESH = 0.10

# ── Internal helpers ──────────────────────────────────────────────────────────

def _col(w):
    return COL_MAP[w]

def _n_topics(configs, w):
    """Number of real topics (excluding -1) for a given weight."""
    return len([k for k in configs[w]["words"] if k != "-1"])

def _ordered_pair(wa, wb, configs):
    """
    Return (source_weight, dest_weight) so that dest always has
    more topics than source. If equal, keep original order.
    """
    na = _n_topics(configs, wa)
    nb = _n_topics(configs, wb)
    return (wa, wb) if na <= nb else (wb, wa)

def _n_topics(configs, w):
    """Number of real topics (excluding -1) for a given weight."""
    return len([k for k in configs[w]["words"] if k != "-1"])

def _ordered_pair(wa, wb, configs):
    """
    Return (source, dest) so dest always has >= topics than source.
    If equal, keep original order.
    """
    return (wa, wb) if _n_topics(configs, wa) <= _n_topics(configs, wb) else (wb, wa)

def _pair_label(wa, wb):
    return f"{wa}_to_{wb}"

def _get_wordset(words_dict, topic_id, top_n=TOP_N_WORDS):
    entries = words_dict.get(str(topic_id), [])
    words = []
    for entry in entries[:top_n]:
        if isinstance(entry, dict):
            word = entry.get("word")
        elif isinstance(entry, (list, tuple)) and entry:
            word = entry[0]
        else:
            word = None
        if word:
            words.append(word)
    return set(words)

def _jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def _norm_entropy(probs):
    n = len(probs)
    if n <= 1:
        return 0.0
    h = stats.entropy(probs)
    h_max = np.log(n)
    return h / h_max if h_max > 0 else 0.0

def _savefig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {Path(path).name}")

def _validate_inputs(merged_all, configs):
    missing_cols = [c for c in COL_MAP.values() if c not in merged_all.columns]
    if missing_cols:
        raise ValueError(f"merged_all is missing columns: {missing_cols}")
    missing_weights = [w for w in WEIGHTS if w not in configs]
    if missing_weights:
        raise ValueError(f"configs is missing weights: {missing_weights}")
    for w in WEIGHTS:
        for key in ["words", "info", "doc"]:
            if key not in configs[w]:
                raise ValueError(f"configs['{w}'] is missing key '{key}'")

# ── Step 1: Aggregate metrics ─────────────────────────────────────────────────

def _step1_aggregate(merged_all, configs, out_dir):
    print("\n── Step 1: Aggregate metrics")
    d = out_dir / "step1_aggregate"
    d.mkdir(exist_ok=True)

    rows = []
    for w in WEIGHTS:
        c         = _col(w)
        total     = len(merged_all)
        n_out     = (merged_all[c] == -1).sum()
        noise_pct = 100 * n_out / total
        n_topics  = len([k for k in configs[w]["words"] if k != "-1"])

        info = configs[w]["info"]
        dbcv = tc = None
        if info is not None and "DBCV" in info.columns and len(info) > 0:
            dbcv = info["DBCV"].iloc[0]
        if info is not None and "Coherence" in info.columns:
            tc = info.loc[info["Topic"] != -1, "Coherence"].mean()

        rows.append({
            "weight":    w,
            "n_topics":  n_topics,
            "noise_pct": round(noise_pct, 2),
            "dbcv":      round(dbcv, 4) if dbcv is not None else None,
            "tc_mean":   round(tc,   4) if tc   is not None else None,
        })
        print(f"    w={w}: {n_topics} topics, {noise_pct:.1f}% noise")

    df = pd.DataFrame(rows)
    df.to_csv(d / "aggregate_metrics.csv", index=False)

    # plot: topic count + noise (dual axis)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ws  = df["weight"].tolist()
    ax1.plot(ws, df["n_topics"],  "o-",  color="#1A6B8A", linewidth=2, label="n topics")
    ax2.plot(ws, df["noise_pct"], "s--", color="#E76F51", linewidth=1.8, label="noise %")
    ax1.set_xlabel("Target weight")
    ax1.set_ylabel("Number of topics", color="#1A6B8A")
    ax2.set_ylabel("Noise %",          color="#E76F51")
    ax1.tick_params(axis="y", labelcolor="#1A6B8A")
    ax2.tick_params(axis="y", labelcolor="#E76F51")
    ax1.set_ylim(0, 50)
    ax2.set_ylim(0, 100)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs  = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper left", fontsize=9)
    ax1.set_title("Topic count and noise % across weight configurations")
    fig.tight_layout()
    _savefig(fig, d / "plot_topic_count_noise.png")

    if df["dbcv"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["weight"], df["dbcv"], "o-", color="#2A9D8F", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Target weight")
        ax.set_ylabel("DBCV")
        ax.set_title("DBCV across weight configurations")
        fig.tight_layout()
        _savefig(fig, d / "plot_dbcv.png")

    return df


# ── Step 2: Document flow ─────────────────────────────────────────────────────

def _step2_document_flow(merged_all, configs, out_dir):
    print("\n── Step 2: Document flow")
    d = out_dir / "step2_document_flow"
    d.mkdir(exist_ok=True)

    summary_rows = []

    for wa_raw, wb_raw in ADJACENT:
        wa, wb = _ordered_pair(wa_raw, wb_raw, configs)
        na, nb = _n_topics(configs, wa), _n_topics(configs, wb)
        ca, cb = _col(wa), _col(wb)
        label  = _pair_label(wa, wb)
        flag   = "  [reordered]" if wa != wa_raw else ""
        print(f"    {wa} → {wb}  ({na} → {nb} topics){flag}")

        topics_b = [t for t in merged_all[cb].unique() if t != -1]
        rows = []
        for t in sorted(topics_b):
            dest       = merged_all[merged_all[cb] == t]
            total      = len(dest)
            src_counts = dest[ca].value_counts()
            dom_topic  = src_counts.index[0]
            dom_pct    = 100 * src_counts.iloc[0] / total

            # if dominant source is outlier-1, use next real topic
            if dom_topic == -1 and len(src_counts) > 1:
                dom_topic = src_counts.index[1]
                dom_pct   = 100 * src_counts.iloc[1] / total

            outlier_pct = 100 * (dest[ca] == -1).sum() / total

            rows.append({
                "topic_b":         t,
                "n_docs_b":        total,
                "dominant_source": dom_topic,
                "dominant_pct":    round(dom_pct,    1),
                "outlier_pct":     round(outlier_pct, 1),
                "category": (
                    "outlier_rescue" if outlier_pct >= 80
                    else "high_outlier"   if outlier_pct >= 25
                    else "clean_renaming" if dom_pct >= 85 and outlier_pct < 15
                    else "mixed"
                ),
            })

        df = pd.DataFrame(rows).sort_values("n_docs_b", ascending=False)
        df.to_csv(d / f"dominant_source_{label}.csv", index=False)

        total   = len(merged_all)
        same    = (merged_all[ca] == merged_all[cb]).sum()
        changed = ((merged_all[ca] != merged_all[cb]) &
                   (merged_all[ca] != -1) & (merged_all[cb] != -1)).sum()
        to_out  = ((merged_all[ca] != -1) & (merged_all[cb] == -1)).sum()
        fr_out  = ((merged_all[ca] == -1) & (merged_all[cb] != -1)).sum()
        cats    = df["category"].value_counts().to_dict()

        summary_rows.append({
            "transition":       f"{wa} → {wb}",
            "stayed_pct":       round(100 * same    / total, 1),
            "changed_pct":      round(100 * changed / total, 1),
            "to_outlier_pct":   round(100 * to_out  / total, 1),
            "rescued_pct":      round(100 * fr_out  / total, 1),
            "n_clean_renaming": cats.get("clean_renaming", 0),
            "n_outlier_rescue": cats.get("outlier_rescue", 0),
            "n_high_outlier":   cats.get("high_outlier",   0),
            "n_mixed":          cats.get("mixed",          0),
        })
        print(f"      stayed {100*same/total:.1f}%  changed {100*changed/total:.1f}%  "
              f"to_outlier {100*to_out/total:.1f}%  rescued {100*fr_out/total:.1f}%")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(d / "flow_summary.csv", index=False)

    pos = np.arange(len(summary_df))
    w   = 0.55

    # plot: stacked bar — document-level movement
    fig, ax = plt.subplots(figsize=(9, 4))
    stayed  = summary_df["stayed_pct"].values
    changed = summary_df["changed_pct"].values
    to_out  = summary_df["to_outlier_pct"].values
    rescued = summary_df["rescued_pct"].values
    ax.bar(pos, stayed,  w, label="Stayed same topic",   color="#2A9D8F")
    ax.bar(pos, changed, w, bottom=stayed,               label="Changed topic",        color="#E9C46A")
    ax.bar(pos, to_out,  w, bottom=stayed+changed,       label="Became outlier",       color="#E76F51")
    ax.bar(pos, rescued, w, bottom=stayed+changed+to_out,label="Rescued from outlier", color="#8ECAE6")
    ax.set_xticks(pos)
    ax.set_xticklabels(summary_df["transition"].tolist(), fontsize=9)
    ax.set_ylabel("% of documents")
    ax.set_ylim(0, 105)
    ax.set_title("Document-level flow between adjacent weight configurations")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _savefig(fig, d / "plot_flow_stacked.png")

    # plot: topic classification breakdown
    cat_cols   = ["n_clean_renaming", "n_outlier_rescue", "n_high_outlier", "n_mixed"]
    cat_labels = ["Clean renaming",   "Outlier rescue",   "High outlier >25%", "Mixed"]
    cat_colors = ["#2A9D8F",          "#9B5DE5",          "#F4A261",           "#888888"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bottoms = np.zeros(len(summary_df))
    for col_name, lbl, color in zip(cat_cols, cat_labels, cat_colors):
        vals = summary_df[col_name].values.astype(float)
        ax.bar(pos, vals, w, bottom=bottoms, label=lbl, color=color)
        bottoms += vals
    ax.set_xticks(pos)
    ax.set_xticklabels(summary_df["transition"].tolist(), fontsize=9)
    ax.set_ylabel("Number of destination topics")
    ax.set_title("Destination topic classification per transition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, d / "plot_topic_classification.png")

    return summary_df


# ── Step 3: Flow concentration (normalised Shannon entropy) ───────────────────

def _step3_entropy(merged_all, configs, out_dir):
    print("\n── Step 3: Flow concentration (normalised Shannon entropy)")
    d = out_dir / "step3_entropy"
    d.mkdir(exist_ok=True)

    all_entropies = {}
    summary_rows  = []

    for wa_raw, wb_raw in ADJACENT:
        wa, wb = _ordered_pair(wa_raw, wb_raw, configs)
        na, nb = _n_topics(configs, wa), _n_topics(configs, wb)
        ca, cb = _col(wa), _col(wb)
        label  = _pair_label(wa, wb)
        flag   = "  [reordered]" if wa != wa_raw else ""
        print(f"    {wa} → {wb}  ({na} → {nb} topics){flag}")

        topics_a = [t for t in merged_all[ca].unique() if t != -1]
        rows = []
        for t in sorted(topics_a):
            src_docs  = merged_all[merged_all[ca] == t]
            changers  = src_docs[src_docs[cb] != t]
            n_changed = len(changers)
            if n_changed == 0:
                continue

            dest_probs = changers[cb].value_counts().values / n_changed
            n_dest     = len(dest_probs)
            raw_h      = stats.entropy(dest_probs)
            norm_h     = _norm_entropy(dest_probs)

            rows.append({
                "topic_a":        t,
                "n_total":        len(src_docs),
                "n_changed":      n_changed,
                "pct_changed":    round(100 * n_changed / len(src_docs), 1),
                "n_destinations": n_dest,
                "raw_entropy":    round(raw_h,  3),
                "norm_entropy":   round(norm_h, 3),
                "band": (
                    "highly_concentrated" if norm_h < 0.2
                    else "moderate"       if norm_h < 0.5
                    else "scattered"      if norm_h < 0.8
                    else "near_max"
                ),
            })

        df = pd.DataFrame(rows).sort_values("norm_entropy")
        df.to_csv(d / f"entropy_{label}.csv", index=False)
        all_entropies[f"{wa}→{wb}"] = df["norm_entropy"].values

        bands = df["band"].value_counts().to_dict()
        summary_rows.append({
            "transition":            f"{wa} → {wb}",
            "mean_norm_entropy":     round(df["norm_entropy"].mean(),   3),
            "median_norm_entropy":   round(df["norm_entropy"].median(), 3),
            "std_norm_entropy":      round(df["norm_entropy"].std(),    3),
            "min":                   round(df["norm_entropy"].min(),    3),
            "max":                   round(df["norm_entropy"].max(),    3),
            "n_highly_concentrated": bands.get("highly_concentrated", 0),
            "n_moderate":            bands.get("moderate",            0),
            "n_scattered":           bands.get("scattered",           0),
            "n_near_max":            bands.get("near_max",            0),
        })
        print(f"      median={df['norm_entropy'].median():.3f}  "
              f"std={df['norm_entropy'].std():.3f}  "
              f"<0.2: {bands.get('highly_concentrated', 0)}  "
              f">0.5: {bands.get('scattered', 0) + bands.get('near_max', 0)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(d / "entropy_summary.csv", index=False)

    transitions = list(all_entropies.keys())
    pos         = np.arange(len(transitions))

    # plot: violin distributions
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vp = ax.violinplot(
        [all_entropies[t] for t in transitions],
        positions=pos, widths=0.6,
        showmedians=True, showextrema=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#8ECAE6")
        body.set_alpha(0.6)
    vp["cmedians"].set_color("#E76F51")
    vp["cmedians"].set_linewidth(2)
    for y, lbl in [(0.2, "0.2 — concentrated"), (0.5, "0.5 — scattered")]:
        ax.axhline(y, color="gray", linewidth=0.7, linestyle="--")
        ax.text(len(transitions) - 0.05, y + 0.01, lbl,
                ha="right", fontsize=8, color="gray")
    ax.set_xticks(pos)
    ax.set_xticklabels(transitions, fontsize=9)
    ax.set_ylabel("Normalised entropy  H_norm ∈ [0, 1]")
    ax.set_title("Flow concentration: normalised Shannon entropy per transition\n"
                 "(orange = median; lower = more concentrated)")
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    _savefig(fig, d / "plot_entropy_distributions.png")

    # plot: median + std summary bars
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pos, summary_df["median_norm_entropy"], 0.55,
           color="#1A6B8A", alpha=0.8, label="median")
    ax.errorbar(pos, summary_df["median_norm_entropy"],
                yerr=summary_df["std_norm_entropy"],
                fmt="none", color="#E76F51", capsize=4,
                linewidth=1.5, label="±1 std")
    ax.axhline(0.2, color="gray", linewidth=0.7, linestyle="--")
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xticks(pos)
    ax.set_xticklabels(summary_df["transition"].tolist(), fontsize=9)
    ax.set_ylabel("Median normalised entropy")
    ax.set_title("Median flow concentration per transition (lower = more stable)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, d / "plot_entropy_summary.png")

    return summary_df


# ── Step 4: Jaccard keyword similarity ────────────────────────────────────────

def _step4_jaccard(configs, out_dir):
    print("\n── Step 4: Jaccard keyword similarity")
    d = out_dir / "step4_jaccard"
    d.mkdir(exist_ok=True)

    all_scores   = {}
    summary_rows = []

    for wa_raw, wb_raw in ADJACENT:
        wa, wb   = _ordered_pair(wa_raw, wb_raw, configs)
        label    = _pair_label(wa, wb)
        words_a  = configs[wa]["words"]
        words_b  = configs[wb]["words"]
        topics_a = [int(k) for k in words_a if k != "-1"]
        topics_b = [int(k) for k in words_b if k != "-1"]
        print(f"    {wa} → {wb}")

        # best-match table
        bm_rows = []
        for tid_a in sorted(topics_a):
            ws_a = _get_wordset(words_a, tid_a)
            best_j, best_tid, best_shared = -1, None, set()
            for tid_b in topics_b:
                ws_b = _get_wordset(words_b, tid_b)
                j = _jaccard(ws_a, ws_b)
                if j > best_j:
                    best_j, best_tid = j, tid_b
                    best_shared = ws_a & ws_b
            bm_rows.append({
                "topic_a":      tid_a,
                "best_match_b": best_tid,
                "jaccard":      round(best_j, 3),
                "n_shared":     len(best_shared),
                "shared_words": ", ".join(sorted(best_shared)),
            })

        bm_df = pd.DataFrame(bm_rows).sort_values("jaccard", ascending=False)
        bm_df.to_csv(d / f"jaccard_{label}.csv", index=False)
        all_scores[f"{wa}→{wb}"] = bm_df["jaccard"].values

        # splitting candidates
        sp_rows = []
        for tid_a in sorted(topics_a):
            ws_a    = _get_wordset(words_a, tid_a)
            matches = []
            for tid_b in topics_b:
                ws_b = _get_wordset(words_b, tid_b)
                j    = _jaccard(ws_a, ws_b)
                if j >= JACCARD_THRESH:
                    matches.append((tid_b, round(j, 3)))
            matches.sort(key=lambda x: -x[1])
            sp_rows.append({
                "topic_a":   tid_a,
                "n_matches": len(matches),
                "matches":   str(matches),
            })

        sp_df = pd.DataFrame(sp_rows).sort_values("n_matches", ascending=False)
        sp_df.to_csv(d / f"splitting_{label}.csv", index=False)

        n_perfect = int((bm_df["jaccard"] == 1.0).sum())
        n_split   = int((sp_df["n_matches"] >= 2).sum())

        summary_rows.append({
            "transition":             f"{wa} → {wb}",
            "mean_jaccard":           round(bm_df["jaccard"].mean(),   3),
            "median_jaccard":         round(bm_df["jaccard"].median(), 3),
            "std_jaccard":            round(bm_df["jaccard"].std(),    3),
            "n_perfect_1.0":          n_perfect,
            "n_topics_a":             len(topics_a),
            "n_splitting_candidates": n_split,
        })
        print(f"      mean={bm_df['jaccard'].mean():.3f}  "
              f"median={bm_df['jaccard'].median():.3f}  "
              f"perfect={n_perfect}  splits={n_split}")

    # skip-one: 0.1 → 0.3
    words_a  = configs["0.1"]["words"]
    words_b  = configs["0.3"]["words"]
    topics_a = [int(k) for k in words_a if k != "-1"]
    topics_b = [int(k) for k in words_b if k != "-1"]
    bm_rows  = []
    for tid_a in sorted(topics_a):
        ws_a = _get_wordset(words_a, tid_a)
        best_j, best_tid, best_shared = -1, None, set()
        for tid_b in topics_b:
            ws_b = _get_wordset(words_b, tid_b)
            j    = _jaccard(ws_a, ws_b)
            if j > best_j:
                best_j, best_tid = j, tid_b
                best_shared = ws_a & ws_b
        bm_rows.append({"topic_a": tid_a, "best_match_b": best_tid,
                         "jaccard": round(best_j, 3), "n_shared": len(best_shared),
                         "shared_words": ", ".join(sorted(best_shared))})
    bm_df = pd.DataFrame(bm_rows).sort_values("jaccard", ascending=False)
    bm_df.to_csv(d / "jaccard_01_to_03_skip.csv", index=False)
    all_scores["0.1→0.3 (skip)"] = bm_df["jaccard"].values
    print(f"    0.1 → 0.3 (skip): mean={bm_df['jaccard'].mean():.3f}  "
          f"median={bm_df['jaccard'].median():.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(d / "jaccard_summary.csv", index=False)

    transitions = list(all_scores.keys())

    # plot: boxplot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bp = ax.boxplot(
        [all_scores[t] for t in transitions],
        labels=transitions, patch_artist=True,
        medianprops={"color": "#E76F51", "linewidth": 2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#8ECAE6")
        patch.set_alpha(0.7)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
    ax.text(len(transitions) - 0.05, 0.51, "0.5 — moderate overlap",
            ha="right", fontsize=8, color="gray")
    ax.set_ylabel("Jaccard similarity (top-15 keywords)")
    ax.set_title("Keyword similarity across transitions\n"
                 "(orange = median; higher = more semantically similar)")
    ax.set_ylim(-0.02, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    _savefig(fig, d / "plot_jaccard_distributions.png")

    # plot: mean + median line
    pos = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos, summary_df["mean_jaccard"],   "o-",  color="#1A6B8A",
            linewidth=2, label="mean")
    ax.plot(pos, summary_df["median_jaccard"], "s--", color="#2A9D8F",
            linewidth=2, label="median")
    ax.fill_between(
        pos,
        summary_df["mean_jaccard"] - summary_df["std_jaccard"],
        summary_df["mean_jaccard"] + summary_df["std_jaccard"],
        alpha=0.15, color="#1A6B8A", label="±1 std",
    )
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(summary_df["transition"].tolist(), fontsize=9)
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Keyword similarity summary across transitions")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, d / "plot_jaccard_summary.png")

    return summary_df


# ── Public entry point ────────────────────────────────────────────────────────

def run_sensitivity_analysis(merged_all, configs, output_dir):
    """
    Run all four sensitivity analysis steps and save results.

    Parameters
    ----------
    merged_all : pd.DataFrame
        One row per document. Must contain columns:
        document_index, topic_01, topic_015, topic_02, topic_025, topic_03.

    configs : dict
        Keys are weight labels: "0.1", "0.15", "0.2", "0.25", "0.3".
        Each value is a dict with:
            "words" : dict           — topic_id (str) → list of {word, score}
            "info"  : pd.DataFrame   — topic_info.csv content (or None)
            "doc"   : pd.DataFrame   — document_topics.csv content

    output_dir : str or Path
        Where to write all results. Created if it does not exist.

    Returns
    -------
    dict with keys "aggregate", "flow", "entropy", "jaccard" — each a DataFrame
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_dir.resolve()}")
    print(f"Documents        : {len(merged_all):,}")
    print(f"Weights          : {WEIGHTS}")

    _validate_inputs(merged_all, configs)

    agg_df     = _step1_aggregate(merged_all, configs, out_dir)
    flow_df    = _step2_document_flow(merged_all, configs, out_dir)
    entropy_df = _step3_entropy(merged_all, configs, out_dir)
    jaccard_df = _step4_jaccard(configs, out_dir)

    print("\n── Done ──────────────────────────────────────────────────────────")
    print(f"Results saved to: {out_dir.resolve()}/")
    print("  step1_aggregate/      — aggregate_metrics.csv + plots")
    print("  step2_document_flow/  — dominant_source_*.csv, flow_summary.csv + plots")
    print("  step3_entropy/        — entropy_*.csv, entropy_summary.csv + plots")
    print("  step4_jaccard/        — jaccard_*.csv, splitting_*.csv, summary + plots")

    return {
        "aggregate": agg_df,
        "flow":      flow_df,
        "entropy":   entropy_df,
        "jaccard":   jaccard_df,
    }