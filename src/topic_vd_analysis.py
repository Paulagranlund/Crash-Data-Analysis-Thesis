"""
topic_vd_analysis.py
====================
For each BERTopic cluster: what does the VD label tell us,
and what does the cluster capture additionally?

Input:  df — merged dataframe with one row per accident, containing
            - topic column (assigned_topic or similar)
            - VD accident-level fields (same names as vejman.dk export)
        topic_col — name of the topic assignment column
        output_dir — folder to write CSV results to

All outputs are CSVs. No interpretation, no plots.
"""

import os
import pandas as pd
import numpy as np

SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _crosstab_pct(df, topic_col, label_col, normalize="index"):
    """Cross-tabulate topic vs label, return row-normalised percentages."""
    ct = pd.crosstab(df[topic_col], df[label_col], normalize=normalize) * 100
    ct.columns = [f"{label_col}={c}_pct" for c in ct.columns]
    return ct


def _topic_profile(df, topic_col, group_cols):
    """For each topic compute count + mean/value_counts of group_cols."""
    rows = []
    for topic, grp in df.groupby(topic_col):
        row = {"topic": topic, "n": len(grp)}
        for col in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_median"] = grp[col].median()
            else:
                top = grp[col].value_counts(normalize=True).head(3)
                for val, pct in top.items():
                    row[f"{col}_top_{val}_pct"] = round(pct * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("topic")


# ---------------------------------------------------------------------------
# main analysis function
# ---------------------------------------------------------------------------

def run_topic_vd_analysis(
    df: pd.DataFrame,
    topic_col: str,
    output_dir: str,
    field_map: dict = None,
):
    """
    field_map — optional dict mapping your actual column names to the
                canonical VD field names expected by this script.

                Keys   = what the column is called in your dataframe
                Values = what this script calls it internally

    Example:
        field_map = {
            "accident_year"       : "AAR",
            "uheld_art"           : "UHELDSART",
            "uheld_situation"     : "UHELDSSITUATION",
            "kryds"               : "KRYDS_UHELD",
            "sprit_niveau"        : "SPRIT",
            "lys_forhold"         : "LYS",
            "vejr"                : "VEJRFORHOLD",
            "vej_kategori"        : "VEJKATEGORI",
            "kommune"             : "UHELDKOMMUNE",
            "antal_draebte"       : "ANTAL_DRAEBTE",
            "antal_alv_tilskade"  : "ANTAL_ALV_TILSKADEKOMNE",
            "antal_let_tilskade"  : "ANTAL_LET_TILSKADEKOMNE",
            "antal_tilskadekomne" : "ANTAL_TILSKADEKOMNE",
        }

    Only include fields whose names differ from the canonical ones.
    Fields not listed are left unchanged.
    """

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Apply field_map — rename to canonical names before any analysis
    # -----------------------------------------------------------------------
    df = df.copy()

    if field_map:
        applicable = {k: v for k, v in field_map.items() if k in df.columns}
        not_found  = {k: v for k, v in field_map.items() if k not in df.columns}
        df = df.rename(columns=applicable)
        if applicable:
            print("Field mapping applied:")
            for src, dst in applicable.items():
                print(f"  {src!r:35s} -> {dst!r}")
        if not_found:
            print("Field mapping skipped (not in df):")
            for src, dst in not_found.items():
                print(f"  {src!r:35s} -> {dst!r}  [NOT FOUND]")

    # -----------------------------------------------------------------------
    # 0. Sanity check — confirm columns present after mapping
    # -----------------------------------------------------------------------
    optional = [
        "AAR", "UHELDSART", "UHELDSSITUATION", "KRYDS_UHELD",
        "SPRIT", "LYS", "VEJRFORHOLD", "VEJKATEGORI", "UHELDKOMMUNE",
        "ANTAL_DRAEBTE", "ANTAL_ALV_TILSKADEKOMNE",
        "ANTAL_LET_TILSKADEKOMNE", "ANTAL_TILSKADEKOMNE",
    ]
    present = [c for c in optional if c in df.columns]
    missing = [c for c in optional if c not in df.columns]
    print(f"\nTopic column     : {topic_col}")
    print(f"VD fields found  : {present}")
    if missing:
        print(f"VD fields missing: {missing}  (skipped in relevant steps)")

    # -----------------------------------------------------------------------
    # 1. Base distribution — how many accidents per topic
    # -----------------------------------------------------------------------
    topic_counts = (
        df[topic_col]
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="n")
        .assign(pct=lambda x: 100 * x["n"] / x["n"].sum())
        .sort_values("topic")
    )
    topic_counts.to_csv(f"{output_dir}/01_topic_counts.csv", index=False)
    print(f"\n[1] Topic counts saved → {output_dir}/01_topic_counts.csv")

    # -----------------------------------------------------------------------
    # 2. UHELDSART distribution per topic
    #    What share of each topic is personskade vs materielskade?
    # -----------------------------------------------------------------------
    if "UHELDSART" in df.columns:
        ct = _crosstab_pct(df, topic_col, "UHELDSART")
        ct_n = pd.crosstab(df[topic_col], df["UHELDSART"])
        ct_n.columns = [f"UHELDSART={c}_n" for c in ct_n.columns]
        out = ct.join(ct_n)
        out.to_csv(f"{output_dir}/02_uheldsart_per_topic.csv")
        print(f"[2] Uheldsart per topic → {output_dir}/02_uheldsart_per_topic.csv")

    # -----------------------------------------------------------------------
    # 3. UHELDSSITUATION distribution per topic
    #    Core question: do multiple topics share the same situation codes,
    #    or does each topic map to distinct codes?
    # -----------------------------------------------------------------------
    if "UHELDSSITUATION" in df.columns:
        # top 10 situation codes per topic
        sit_rows = []
        for topic, grp in df.groupby(topic_col):
            vc = grp["UHELDSSITUATION"].value_counts(normalize=True).head(10)
            for code, pct in vc.items():
                sit_rows.append({
                    "topic": topic,
                    "UHELDSSITUATION": code,
                    "pct_within_topic": round(pct * 100, 2),
                    "n": int(grp["UHELDSSITUATION"].eq(code).sum()),
                })
        sit_df = pd.DataFrame(sit_rows)
        sit_df.to_csv(f"{output_dir}/03_situation_per_topic.csv", index=False)

        # inverse: for each situation code, which topics claim it?
        sit_inv_rows = []
        for code, grp in df.groupby("UHELDSSITUATION"):
            vc = grp[topic_col].value_counts(normalize=True).head(5)
            for topic, pct in vc.items():
                sit_inv_rows.append({
                    "UHELDSSITUATION": code,
                    "topic": topic,
                    "pct_of_code_in_topic": round(pct * 100, 2),
                    "n": int(grp[topic_col].eq(topic).sum()),
                })
        pd.DataFrame(sit_inv_rows).to_csv(
            f"{output_dir}/03b_topic_per_situation.csv", index=False
        )
        print(f"[3] Situation per topic → {output_dir}/03_situation_per_topic.csv")
        print(f"    Inverse (topic per situation) → {output_dir}/03b_topic_per_situation.csv")

    # -----------------------------------------------------------------------
    # 4. SPRIT cross-tabulation — alcohol topics (6, 15)
    #    Core question: how many cluster members lack the Sprit label?
    #
    #    Expects a pre-computed binary column "sprit_flag" (0/1) in df,
    #    where 1 = spirituspåvirket at accident level.
    #    The caller is responsible for computing this flag before calling
    #    this function, e.g.:
    #
    #       df_acc["sprit_flag"] = df_acc["SPRIT"].astype(int)
    #
    # -----------------------------------------------------------------------
    if "sprit_flag" in df.columns:
        n_flagged = int(df["sprit_flag"].sum())
        print(f"    SPRIT: {n_flagged} accidents flagged ({100*n_flagged/len(df):.1f}%)")

        sprit_rows = []
        for topic, grp in df.groupby(topic_col):
            n = len(grp)
            n_labelled   = int(grp["sprit_flag"].sum())
            n_unlabelled = n - n_labelled
            sprit_rows.append({
                "topic": topic,
                "n": n,
                "n_sprit_labelled": n_labelled,
                "n_sprit_unlabelled": n_unlabelled,
                "pct_labelled": round(100 * n_labelled / n, 2),
                "pct_unlabelled": round(100 * n_unlabelled / n, 2),
            })

        pd.DataFrame(sprit_rows).to_csv(
            f"{output_dir}/04_sprit_crosstab.csv", index=False
        )
        print(f"[4] Sprit cross-tab -> {output_dir}/04_sprit_crosstab.csv")

    # -----------------------------------------------------------------------
    # 5. KRYDS_UHELD distribution per topic
    #    Junction topics (0, 2, 19, 21, 24, 36) should be Ja-heavy
    # -----------------------------------------------------------------------
    if "KRYDS_UHELD" in df.columns:
        ct = _crosstab_pct(df, topic_col, "KRYDS_UHELD")
        ct_n = pd.crosstab(df[topic_col], df["KRYDS_UHELD"])
        ct_n.columns = [f"KRYDS_UHELD={c}_n" for c in ct_n.columns]
        ct.join(ct_n).to_csv(f"{output_dir}/05_kryds_per_topic.csv")
        print(f"[5] Kryds_uheld per topic → {output_dir}/05_kryds_per_topic.csv")

    # -----------------------------------------------------------------------
    # 6. LYS distribution per topic
    #    Night overrepresentation in hit-and-run (5,11), drunk (6,15), deer (32)
    # -----------------------------------------------------------------------
    if "LYS" in df.columns:
        ct = _crosstab_pct(df, topic_col, "LYS")
        ct_n = pd.crosstab(df[topic_col], df["LYS"])
        ct_n.columns = [f"LYS={c}_n" for c in ct_n.columns]
        ct.join(ct_n).to_csv(f"{output_dir}/06_lys_per_topic.csv")
        print(f"[6] Lys per topic → {output_dir}/06_lys_per_topic.csv")

    # -----------------------------------------------------------------------
    # 7. VEJKATEGORI distribution per topic
    #    Motorway topics (8, 23, 35) should be motorvej-heavy
    #    Animal topics: deer (32) rural, dog (12) urban
    # -----------------------------------------------------------------------
    if "VEJKATEGORI" in df.columns:
        ct = _crosstab_pct(df, topic_col, "VEJKATEGORI")
        ct_n = pd.crosstab(df[topic_col], df["VEJKATEGORI"])
        ct_n.columns = [f"VEJKATEGORI={c}_n" for c in ct_n.columns]
        ct.join(ct_n).to_csv(f"{output_dir}/07_vejkategori_per_topic.csv")
        print(f"[7] Vejkategori per topic → {output_dir}/07_vejkategori_per_topic.csv")

    # -----------------------------------------------------------------------
    # 8. Severity per topic
    #    Combines dræbte + alv + let tilskadekomne
    # -----------------------------------------------------------------------
    sev_cols = [c for c in [
        "ANTAL_DRAEBTE", "ANTAL_ALV_TILSKADEKOMNE",
        "ANTAL_LET_TILSKADEKOMNE", "ANTAL_TILSKADEKOMNE"
    ] if c in df.columns]

    if sev_cols:
        sev_rows = []
        for topic, grp in df.groupby(topic_col):
            row = {"topic": topic, "n": len(grp)}
            for col in sev_cols:
                row[f"{col}_mean"] = round(grp[col].mean(), 4)
                row[f"{col}_sum"] = int(grp[col].sum())
            if "ANTAL_DRAEBTE" in sev_cols:
                row["pct_fatal"] = round(
                    100 * grp["ANTAL_DRAEBTE"].gt(0).mean(), 2
                )
            sev_rows.append(row)
        pd.DataFrame(sev_rows).sort_values("topic").to_csv(
            f"{output_dir}/08_severity_per_topic.csv", index=False
        )
        print(f"[8] Severity per topic → {output_dir}/08_severity_per_topic.csv")

    # -----------------------------------------------------------------------
    # 9. Temporal distribution per topic (year)
    # -----------------------------------------------------------------------
    if "AAR" in df.columns:
        year_topic = (
            df.groupby(["AAR", topic_col])
            .size()
            .reset_index(name="n")
        )
        # also as share of total accidents per year
        year_totals = df.groupby("AAR").size().rename("n_year")
        year_topic = year_topic.join(year_totals, on="AAR")
        year_topic["pct_of_year"] = round(
            100 * year_topic["n"] / year_topic["n_year"], 4
        )
        year_topic.to_csv(f"{output_dir}/09_topic_per_year.csv", index=False)
        print(f"[9] Topic per year → {output_dir}/09_topic_per_year.csv")

    # -----------------------------------------------------------------------
    # 10. VEJRFORHOLD distribution per topic
    #     Ice/snow topics (22, 37) should show glat/is overrepresentation
    # -----------------------------------------------------------------------
    if "VEJRFORHOLD" in df.columns:
        ct = _crosstab_pct(df, topic_col, "VEJRFORHOLD")
        ct_n = pd.crosstab(df[topic_col], df["VEJRFORHOLD"])
        ct_n.columns = [f"VEJRFORHOLD={c}_n" for c in ct_n.columns]
        ct.join(ct_n).to_csv(f"{output_dir}/10_vejrforhold_per_topic.csv")
        print(f"[10] Vejrforhold per topic → {output_dir}/10_vejrforhold_per_topic.csv")

    # -----------------------------------------------------------------------
    # 11. No-label topics — pure descriptive profile
    #     Topics 5, 11, 17, 20, 33, 39, 40 have no VD equivalent.
    #     Profile them on all available fields to characterise what VD misses.
    # -----------------------------------------------------------------------
    no_label_topics = [5, 11, 17, 20, 33, 39, 40]
    no_label_df = df[df[topic_col].isin(no_label_topics)].copy()

    if len(no_label_df) > 0:
        profile_cols = [c for c in present if c not in [topic_col, "UHELDSSITUATION"]]
        profile = _topic_profile(no_label_df, topic_col, profile_cols)
        profile.to_csv(f"{output_dir}/11_no_label_topic_profile.csv", index=False)
        print(f"[11] No-label topic profile → {output_dir}/11_no_label_topic_profile.csv")

    # -----------------------------------------------------------------------
    # 12. Situation code overlap matrix — which topics share situation codes?
    #     Answers: are partial-coverage topics actually distinguishable by
    #     Uheldssituation alone, or does the text carry the distinction?
    # -----------------------------------------------------------------------
    if "UHELDSSITUATION" in df.columns:
        # For each pair of topics, compute Jaccard similarity of their
        # top-20 most common situation codes
        topic_sit_sets = {}
        for topic, grp in df.groupby(topic_col):
            top20 = set(grp["UHELDSSITUATION"].value_counts().head(20).index)
            topic_sit_sets[topic] = top20

        topics_sorted = sorted(topic_sit_sets.keys())
        jaccard_rows = []
        for t1 in topics_sorted:
            for t2 in topics_sorted:
                if t1 >= t2:
                    continue
                s1, s2 = topic_sit_sets[t1], topic_sit_sets[t2]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                j = inter / union if union > 0 else 0
                jaccard_rows.append({
                    "topic_a": t1, "topic_b": t2,
                    "jaccard": round(j, 3),
                    "shared_codes": inter,
                    "shared_code_list": str(sorted(s1 & s2)),
                })
        jac_df = pd.DataFrame(jaccard_rows).sort_values("jaccard", ascending=False)
        jac_df.to_csv(f"{output_dir}/12_situation_jaccard.csv", index=False)
        print(f"[12] Situation code Jaccard overlap → {output_dir}/12_situation_jaccard.csv")

    # -----------------------------------------------------------------------
    # 13. UHELDSSITUATION uniqueness — for each topic, what fraction of
    #     its accidents have a situation code that also appears in OTHER topics?
    #     High overlap = label cannot distinguish; low overlap = text adds real info
    # -----------------------------------------------------------------------
    if "UHELDSSITUATION" in df.columns:
        # all situation codes present in each topic
        topic_all_codes = {
            t: set(g["UHELDSSITUATION"].dropna().unique())
            for t, g in df.groupby(topic_col)
        }
        uniqueness_rows = []
        for topic, grp in df.groupby(topic_col):
            own_codes = topic_all_codes[topic]
            other_codes = set().union(*[
                v for k, v in topic_all_codes.items() if k != topic
            ])
            shared = own_codes & other_codes
            unique = own_codes - other_codes
            # fraction of documents in this topic with a shared code
            n_shared_docs = grp["UHELDSSITUATION"].isin(shared).sum()
            n_unique_docs = grp["UHELDSSITUATION"].isin(unique).sum()
            uniqueness_rows.append({
                "topic": topic,
                "n": len(grp),
                "n_unique_situation_codes": len(unique),
                "n_shared_situation_codes": len(shared),
                "n_docs_with_unique_code": int(n_unique_docs),
                "n_docs_with_shared_code": int(n_shared_docs),
                "pct_docs_unique_code": round(100 * n_unique_docs / len(grp), 2),
                "pct_docs_shared_code": round(100 * n_shared_docs / len(grp), 2),
            })
        pd.DataFrame(uniqueness_rows).sort_values("topic").to_csv(
            f"{output_dir}/13_situation_uniqueness.csv", index=False
        )
        print(f"[13] Situation code uniqueness → {output_dir}/13_situation_uniqueness.csv")

    # -----------------------------------------------------------------------
    # 14. Animal topics (12, 29, 32) — confirm same VD code, different cluster
    # -----------------------------------------------------------------------
    animal_topics = [12, 29, 32]
    animal_df = df[df[topic_col].isin(animal_topics)].copy()

    if len(animal_df) > 0 and "UHELDSSITUATION" in df.columns:
        animal_sit = pd.crosstab(
            animal_df[topic_col],
            animal_df["UHELDSSITUATION"],
            normalize="index"
        ) * 100
        animal_sit.to_csv(f"{output_dir}/14_animal_situation_crosstab.csv")

        # severity comparison across three species topics
        if sev_cols:
            animal_sev = (
                animal_df.groupby(topic_col)[sev_cols]
                .agg(["mean", "sum"])
            )
            animal_sev.to_csv(f"{output_dir}/14b_animal_severity.csv")
        print(f"[14] Animal topic analysis → {output_dir}/14_animal_*.csv")

    # -----------------------------------------------------------------------
    # 15. Summary table — one row per topic with key label coverage metrics
    # -----------------------------------------------------------------------
    summary_rows = []
    for topic, grp in df.groupby(topic_col):
        row = {"topic": topic, "n": len(grp)}

        if "UHELDSART" in df.columns:
            row["pct_personskade"] = round(
                100 * grp["UHELDSART"].eq("Pskduh").mean(), 2
            )
        if "KRYDS_UHELD" in df.columns:
            row["pct_kryds"] = round(
                100 * grp["KRYDS_UHELD"].eq("Ja").mean(), 2
            )
        if "sprit_flag" in df.columns:
            row["pct_sprit_labelled"] = round(
                100 * grp["sprit_flag"].mean(), 2
            )
        if "LYS" in df.columns:
            row["pct_morke"] = round(
                100 * grp["LYS"].isin(["Mørke", "Mørkt", "morke", "2"]).mean(), 2
            )
        if "ANTAL_DRAEBTE" in df.columns:
            row["pct_fatal"] = round(
                100 * grp["ANTAL_DRAEBTE"].gt(0).mean(), 2
            )
        if "UHELDSSITUATION" in df.columns:
            row["top1_situation"] = (
                grp["UHELDSSITUATION"].value_counts().index[0]
                if grp["UHELDSSITUATION"].notna().any() else None
            )
            row["top1_situation_pct"] = round(
                100 * grp["UHELDSSITUATION"].value_counts(normalize=True).iloc[0], 2
            ) if grp["UHELDSSITUATION"].notna().any() else None
            row["n_distinct_situations"] = grp["UHELDSSITUATION"].nunique()
        if "AAR" in df.columns:
            row["year_min"] = grp["AAR"].min()
            row["year_max"] = grp["AAR"].max()

        summary_rows.append(row)

    pd.DataFrame(summary_rows).sort_values("topic").to_csv(
        f"{output_dir}/00_summary.csv", index=False
    )
    print(f"\n[00] Summary table → {output_dir}/00_summary.csv")
    print("\nDone.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
# Edit field_map so the keys match your actual column names.
# Remove any entry where your column name already matches the canonical name.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    field_map = {
        # "your_column_name"    : "CANONICAL_NAME",
        # Examples — replace left side with your actual names:
        # "accident_year"       : "AAR",
        # "uheld_art"           : "UHELDSART",
        # "uheld_situation"     : "UHELDSSITUATION",
        # "kryds_uheld"         : "KRYDS_UHELD",
        # "sprit"               : "SPRIT",
        # "lys"                 : "LYS",
        # "vejrforhold"         : "VEJRFORHOLD",
        # "vejkategori"         : "VEJKATEGORI",
        # "uheldkommune"        : "UHELDKOMMUNE",
        # "antal_draebte"       : "ANTAL_DRAEBTE",
        # "antal_alv_tilskade"  : "ANTAL_ALV_TILSKADEKOMNE",
        # "antal_let_tilskade"  : "ANTAL_LET_TILSKADEKOMNE",
        # "antal_tilskadekomne" : "ANTAL_TILSKADEKOMNE",
    }

    run_topic_vd_analysis(
        df=df,
        topic_col="assigned_topic",
        output_dir="results/topic_vd_analysis",
        field_map=field_map,
    )