"""
rear_end_analysis.py
====================
Added value analysis for rear-end topics 9, 18, 23, 27.

All four topics share "Påkørsel bagfra" as their dominant VD situation string.
This analysis shows what the narrative clusters reveal that the single label cannot.

Sections:
  1. Profile comparison — road type, alcohol, severity, junction, light
  2. Statistical tests — are the differences between clusters significant?
  3. Geographic distribution — kommune_group normalised rates
  4. Temporal trends — normalised per 10,000 accidents per year
  5. Changepoint detection — PELT on log-rate with Poisson adjustment
  6. Label distribution — where do all "Påkørsel bagfra" accidents go?

Input:
    data     — full accident dataframe (one row per accident)
               must contain: assigned_topic, year, VEJKATEGORI, KRYDS_UHELD,
               LYS, VEJR, sprit_flag, ANTAL_DRAEBTE, UH_UHID_UHANTALVTILS,
               report_category, accident_situation, kommune_group,
               police_narrative
    output_dir — folder to write results and plots into
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ruptures as rpt
from statsmodels.stats.proportion import proportions_ztest

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

REAR_END_TOPICS = {
    9:  "Topic 9 — general rear-end",
    18: "Topic 18 — harmonika/motorway queue",
    23: "Topic 23 — motorway queue kødannelse",
    27: "Topic 27 — emergency braking rear-end",
}

TOPIC_COLORS = {
    9:  "#3266ad",
    18: "#c0392b",
    23: "#27ae60",
    27: "#e67e22",
}

LABEL_STR   = "Påkørsel bagfra"
TOPIC_COL   = "assigned_topic"
YEAR_COL    = "year"
KOMMUNE_COL = "kommune_group"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_penalty(criterion, n):
    c = criterion.lower()
    if c == "aic":
        return 2
    elif c == "bic":
        return np.log(n)
    elif c == "mbic":
        return 3 * np.log(n)
    raise ValueError(f"Unknown criterion: {criterion}")


def _year_rate(data, mask, year_total, scale=10000):
    """Return normalised rate per `scale` accidents per year."""
    counts = data[mask].groupby(YEAR_COL).size().rename("n")
    combined = pd.concat([year_total, counts], axis=1).fillna(0)
    combined["rate"] = combined["n"] / combined["n_total"] * scale
    return combined.reset_index()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_rear_end_analysis(data: pd.DataFrame, output_dir: str, criterion: str = "bic"):
    os.makedirs(output_dir, exist_ok=True)

    year_total = data.groupby(YEAR_COL).size().rename("n_total")

    # -----------------------------------------------------------------------
    # 1. Profile comparison
    # -----------------------------------------------------------------------
    rows = []
    for t, label in REAR_END_TOPICS.items():
        grp = data[data[TOPIC_COL] == t]
        n   = len(grp)
        rows.append({
            "topic":            t,
            "label":            label,
            "n":                n,
            "pct_motorvej":     round((grp["VEJKATEGORI"] == "Hldv").mean() * 100, 1),
            "pct_junction":     round((grp["KRYDS_UHELD"] == "Ja").mean()  * 100, 1),
            "pct_dark":         round((grp["LYS"] == "Mørke").mean()       * 100, 1),
            "pct_sprit":        round(grp["sprit_flag"].mean()              * 100, 1),
            "pct_fatal":        round(grp["ANTAL_DRAEBTE"].gt(0).mean()    * 100, 2),
            "killed":           int(grp["ANTAL_DRAEBTE"].sum()),
            "alv_mean":         round(grp["UH_UHID_UHANTALVTILS"].mean(),  4),
            "pct_personskade":  round((grp["report_category"] == "Pskduh").mean() * 100, 1),
        })

    # add full label baseline
    lbl = data[data["accident_situation"] == LABEL_STR]
    rows.append({
        "topic":           "label_all",
        "label":           f"VD label — {LABEL_STR} (all)",
        "n":               len(lbl),
        "pct_motorvej":    round((lbl["VEJKATEGORI"] == "Hldv").mean() * 100, 1),
        "pct_junction":    round((lbl["KRYDS_UHELD"] == "Ja").mean()   * 100, 1),
        "pct_dark":        round((lbl["LYS"] == "Mørke").mean()        * 100, 1),
        "pct_sprit":       round(lbl["sprit_flag"].mean()               * 100, 1),
        "pct_fatal":       round(lbl["ANTAL_DRAEBTE"].gt(0).mean()     * 100, 2),
        "killed":          int(lbl["ANTAL_DRAEBTE"].sum()),
        "alv_mean":        round(lbl["UH_UHID_UHANTALVTILS"].mean(),   4),
        "pct_personskade": round((lbl["report_category"] == "Pskduh").mean() * 100, 1),
    })

    profile_df = pd.DataFrame(rows)
    profile_df.to_csv(f"{output_dir}/01_profile_comparison.csv", index=False)
    print("[1] Profile comparison saved")

    # -----------------------------------------------------------------------
    # 2. Statistical tests between clusters
    # -----------------------------------------------------------------------
    topics = list(REAR_END_TOPICS.keys())
    test_rows = []

    for field, col, transform in [
        ("Motorway",  "VEJKATEGORI", lambda g: (g["VEJKATEGORI"] == "Hldv").astype(int)),
        ("Sprit",     "sprit_flag",  lambda g: g["sprit_flag"]),
        ("Fatal",     "ANTAL_DRAEBTE", lambda g: g["ANTAL_DRAEBTE"].gt(0).astype(int)),
    ]:
        for i, ta in enumerate(topics):
            for tb in topics[i+1:]:
                ga = data[data[TOPIC_COL] == ta]
                gb = data[data[TOPIC_COL] == tb]
                ca = int(transform(ga).sum())
                cb = int(transform(gb).sum())
                stat, p = proportions_ztest([ca, cb], [len(ga), len(gb)])
                test_rows.append({
                    "field":    field,
                    "topic_a":  ta,
                    "topic_b":  tb,
                    "rate_a":   round(ca / len(ga) * 100, 2),
                    "rate_b":   round(cb / len(gb) * 100, 2),
                    "z":        round(stat, 3),
                    "p":        round(p, 4),
                    "sig":      "*" if p < 0.05 else "",
                })

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(f"{output_dir}/02_statistical_tests.csv", index=False)
    print("[2] Statistical tests saved")

    # -----------------------------------------------------------------------
    # 3. Geographic distribution — kommune_group
    # -----------------------------------------------------------------------
    gruppe_total = data.groupby(KOMMUNE_COL).size().rename("n_total")
    geo_rows = []

    for t in topics:
        gt = (data[data[TOPIC_COL] == t]
              .groupby(KOMMUNE_COL).size().rename("n_topic"))
        combined = pd.concat([gruppe_total, gt], axis=1).fillna(0)
        combined["rate_per_1000"] = (combined["n_topic"] / combined["n_total"] * 1000).round(2)
        combined["topic"] = t
        geo_rows.append(combined.reset_index()[
            [KOMMUNE_COL, "topic", "n_total", "n_topic", "rate_per_1000"]
        ])

    geo_df = pd.concat(geo_rows).sort_values(
        ["topic", "rate_per_1000"], ascending=[True, False]
    )
    geo_df.to_csv(f"{output_dir}/03_geographic_distribution.csv", index=False)
    print("[3] Geographic distribution saved")

    # -----------------------------------------------------------------------
    # 4 + 5. Temporal trends + changepoint detection
    # -----------------------------------------------------------------------
    cp_rows = []

    # per-topic rates
    topic_rates = {}
    for t in topics:
        df_t = _year_rate(data, data[TOPIC_COL] == t, year_total, scale=10000)
        topic_rates[t] = df_t

    # label rate
    label_rate = _year_rate(data, data["accident_situation"] == LABEL_STR,
                            year_total, scale=10000)

    # changepoint detection on log-rate
    for t, df_t in topic_rates.items():
        series    = df_t.set_index(YEAR_COL)["n"]
        totals    = df_t.set_index(YEAR_COL)["n_total"]
        log_rate  = np.log((series + 0.5) / totals)
        n         = len(log_rate)
        penalty   = _get_penalty(criterion, n)

        algo       = rpt.Pelt(model="rbf").fit(log_rate.values.reshape(-1, 1))
        breakpoints = algo.predict(pen=penalty)[:-1]

        for bp in breakpoints:
            bp_year = log_rate.index[bp]
            before  = df_t[df_t[YEAR_COL] < bp_year]["rate"].mean()
            after   = df_t[df_t[YEAR_COL] >= bp_year]["rate"].mean()
            cp_rows.append({
                "topic":           t,
                "breakpoint_year": bp_year,
                "criterion":       criterion,
                "mean_rate_before": round(before, 3),
                "mean_rate_after":  round(after, 3),
                "direction":       "↑" if after > before else "↓",
            })

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(f"{output_dir}/04_changepoints.csv", index=False)
    print(f"[4] Changepoints saved ({len(cp_df)} detected)")
    if len(cp_df) > 0:
        print(cp_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Plot 1 — two-panel: VD label (top) + four clusters (bottom)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    ax1.plot(label_rate[YEAR_COL], label_rate["rate"],
             color="black", linewidth=2, marker="o", markersize=3)
    ax1.set_title(f"VD label — {LABEL_STR} (all accidents)", fontsize=11)
    ax1.set_ylabel("Rate per 10,000 accidents", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="lightgrey", linewidth=0.5)

    for t, df_t in topic_rates.items():
        ax2.plot(df_t[YEAR_COL], df_t["rate"],
                 color=TOPIC_COLORS[t], linewidth=2, marker="o", markersize=3,
                 label=REAR_END_TOPICS[t])

        # mark changepoints
        for _, row in cp_df[cp_df["topic"] == t].iterrows():
            ax2.axvline(x=row["breakpoint_year"], color=TOPIC_COLORS[t],
                        linestyle="--", linewidth=1, alpha=0.6)
            ax2.text(row["breakpoint_year"] + 0.3,
                     df_t[df_t[YEAR_COL] == row["breakpoint_year"]]["rate"].values[0],
                     f"{row['breakpoint_year']}{row['direction']}",
                     fontsize=7, color=TOPIC_COLORS[t])

    ax2.set_title("BERTopic clusters — rear-end sub-types from narrative", fontsize=11)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Rate per 10,000 accidents", fontsize=10)
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Temporal plot saved")

    # -----------------------------------------------------------------------
    # Plot 2 — geographic bar chart
    # -----------------------------------------------------------------------
    kommune_order = ["Hovedstad", "Storby", "Provinsby", "Opland", "Land"]
    geo_pivot = (geo_df[geo_df[KOMMUNE_COL].isin(kommune_order)]
                 .pivot(index=KOMMUNE_COL, columns="topic", values="rate_per_1000")
                 .reindex(kommune_order))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(kommune_order))
    width = 0.2

    for i, t in enumerate(topics):
        ax.bar(x + i * width, geo_pivot[t], width,
               color=TOPIC_COLORS[t], label=REAR_END_TOPICS[t])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(kommune_order, fontsize=10)
    ax.set_ylabel("Rate per 1,000 accidents", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_geographic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Geographic plot saved")

    # -----------------------------------------------------------------------
    # 5. Label distribution — where does Påkørsel bagfra go?
    # -----------------------------------------------------------------------
    label_mask = data["accident_situation"] == LABEL_STR
    label_dist = (
        data[label_mask][TOPIC_COL]
        .value_counts()
        .reset_index()
        .rename(columns={TOPIC_COL: "topic", "count": "n"})
        .assign(pct=lambda x: (x["n"] / label_mask.sum() * 100).round(1))
    )
    label_dist.to_csv(f"{output_dir}/05_label_distribution.csv", index=False)
    print(f"[5] Label distribution saved — {LABEL_STR} goes to {len(label_dist)} topics")

    print(f"\nAll outputs written to: {output_dir}")
    return profile_df, tests_df, geo_df, cp_df


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile_df, tests_df, geo_df, cp_df = run_rear_end_analysis(
        data=data,
        output_dir="results/rear_end_analysis",
        criterion="bic",
    )
