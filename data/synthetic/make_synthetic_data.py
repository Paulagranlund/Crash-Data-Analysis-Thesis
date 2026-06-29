"""make_synthetic_data.py
=======================
Generate small, schema-correct stand-in files so the Track 2 pipeline runs end
to end without the restricted VD data. The values are random, so anything the
notebooks compute on them is meaningless by construction; this exists only to
let the code execute, not to produce findings.

Run from the repository root:

    python data/synthetic/make_synthetic_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SYNTH = os.path.join(REPO_ROOT, "data", "synthetic")
SEED = 42
N_ACC = 1200  # number of synthetic accidents

rng = np.random.default_rng(SEED)


def _write_header2(df, path):
    """Write an .xlsx with two blank rows above the header, matching the real
    exports that the loaders read with header=2."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, startrow=2, index=False)


def main():
    os.makedirs(os.path.join(SYNTH, "base"), exist_ok=True)
    os.makedirs(os.path.join(SYNTH, "element"), exist_ok=True)

    acc_ids = np.arange(1, N_ACC + 1)
    years = rng.integers(2000, 2026, size=N_ACC)
    dates = [f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,28):02d}" for y in years]

    # situation strings/codes — include the ones the analyses key on
    sit_strings = rng.choice(
        ["Påkørsel bagfra", "Eneuheld", "Frontalkollision", "Krydsningsuheld",
         "Fodgænger", "Parkeret køretøj"], size=N_ACC)
    sit_codes = rng.choice([140, 201, 312, 410, 510, 660], size=N_ACC)
    report_cat = rng.choice(["Anmsuh", "Exuh", "Pskduh", "Mskduh"], size=N_ACC)
    narratives = [
        " ".join(rng.choice(
            ["bil", "kørte", "ind", "i", "vognbane", "bremsede", "holdt", "stille",
             "cyklist", "væltede", "spiritus", "påkørte", "flugtede", "ukendt", "fører"],
            size=rng.integers(4, 12))) for _ in range(N_ACC)
    ]

    # ── base accident data ───────────────────────────────────────────────────
    base = pd.DataFrame({
        "UHELDSDATO": dates,
        "UHELDSART": report_cat,
        "KODE_UHELDSSITUATION": sit_codes,
        "UHELDSSITUATION": sit_strings,
        "UHELDSTEKST": narratives,
        "AAR": years,
        "UHELDS_ID": acc_ids,
    })
    _write_header2(base, os.path.join(SYNTH, "base", "base_2000_2025.xlsx"))

    # cleaned length (data_load drops nothing here since all rows are valid)
    n_clean = N_ACC

    # ── element / person rows (data 5) ───────────────────────────────────────
    el_rows = []
    for aid in acc_ids:
        for elem in range(1, rng.integers(1, 4) + 1):
            el_rows.append({
                "UHELDS_ID": aid,
                "ALDER": int(rng.integers(15, 85)),
                "PERSONNR": 1,
                "ELEMENTNR": elem,
                "ELEMENTART": rng.choice(["Cykl", "Pkør", "Lastb", "Fodg", "Knal"]),
            })
    _write_header2(pd.DataFrame(el_rows), os.path.join(SYNTH, "element", "element_rows.xlsx"))

    # ── VD fields (lable_analysis_1..4), person-level ────────────────────────
    vd_rows = []
    for aid in acc_ids:
        for _ in range(rng.integers(1, 3)):
            vd_rows.append({
                "UHELDS_ID": aid,
                "KODE_UHELDSSITUATION": int(rng.choice([140, 201, 312, 410])),
                "KODE_UHELDSART": rng.choice(["Anmsuh", "Exuh", "Pskduh", "Mskduh"]),
                "VEJR": rng.choice(["Tør", "Regn", "Sne", "Glat"]),
                "VEJKATEGORI": rng.choice(["Hldv", "Kvej", "Bvej", "Mvej"]),
                "KRYDS_UHELD": rng.choice(["Ja", "Nej"]),
                "LYS": rng.choice(["Dagslys", "Mørke", "Tusmørke"]),
                "SPRIT": int(rng.choice([0, 0, 0, 55, 80, 999])),
                "ANTAL_DRAEBTE": int(rng.choice([0, 0, 0, 1])),
                "UH_UHID_UHANTALVTILS": int(rng.integers(0, 3)),
                "UH_UHID_UHANTLETTILS": int(rng.integers(0, 4)),
            })
    vd = pd.DataFrame(vd_rows)
    for i in range(4):
        part = vd.iloc[i::4].reset_index(drop=True)
        _write_header2(part, os.path.join(SYNTH, f"lable_analysis_{i+1}.xlsx"))

    # ── severity (bad_uheld) ─────────────────────────────────────────────────
    bad_rows = []
    for aid in acc_ids:
        for elem in range(1, rng.integers(1, 4) + 1):
            bad_rows.append({
                "UHELDS_ID": aid,
                "ELEMENTNR": elem,
                "PERSONSKADE": rng.choice(["Ingen", "Let", "Alv", "Dr"]),
            })
    _write_header2(pd.DataFrame(bad_rows), os.path.join(SYNTH, "bad_uheld.xlsx"))

    # ── coordinates + kommune ────────────────────────────────────────────────
    from analysis_data import KOMMUNE_GROUPS
    kommune_codes = list(KOMMUNE_GROUPS.keys())
    coords = pd.DataFrame({
        "UHELDS_ID": acc_ids,
        "KODE_UHELDKOMMUNE": rng.choice(kommune_codes, size=N_ACC),
        "x": rng.uniform(440000, 900000, size=N_ACC),
        "y": rng.uniform(6050000, 6400000, size=N_ACC),
    })
    coords.to_parquet(os.path.join(SYNTH, "df_coords.parquet"), index=False)

    # ── BERTopic outputs per configuration (written to results/, not data/) ───
    from config import RESULTS_SEMI_DIR

    def make_topics(size):
        topics = rng.integers(-1, 41, size=size)
        # Ensure Topic 11 exists with enough rows for case-study plots and tests.
        topic11_idx = rng.choice(size, size=max(80, size // 12), replace=False)
        topics[topic11_idx] = 11
        return topics

    topic_configs = [
        "unsupervised",
        "main_0.3",
        "report_accident_0.25",
        "all_0.1",
        "all_0.15",
        "all_0.2",
        "all_0.25",
        "all_0.3",
        "main_0.1",
        "main_0.15",
        "main_0.2",
        "main_0.25",
    ]
    main_topic11_idx = None
    for cfg in topic_configs:
        d = os.path.join(RESULTS_SEMI_DIR, cfg)
        os.makedirs(d, exist_ok=True)
        topics = make_topics(n_clean)
        if cfg == "main_0.3":
            main_topic11_idx = np.flatnonzero(topics == 11)
        doc = pd.DataFrame({
            "document_index": np.arange(n_clean),
            "assigned_topic": topics,
            "assigned_topic_probability": rng.uniform(0, 1, size=n_clean).round(3),
        })
        doc.to_csv(os.path.join(d, "document_topics.csv"), index=False)

        info = (pd.Series(topics).value_counts().rename_axis("Topic")
                .reset_index(name="Count").sort_values("Topic"))
        info["Name"] = info["Topic"].apply(lambda t: f"{t}_synthetic_topic")
        info.to_csv(os.path.join(d, "topic_info.csv"), index=False)

        topic_words = {
            str(t): [[f"synthetic_{int(t)}_{i}", round(float(rng.uniform(0.05, 0.9)), 4)] for i in range(10)]
            for t in sorted(set(topics)) if t != -1
        }
        with open(os.path.join(d, "topic_words.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(topic_words, f, ensure_ascii=False, indent=2)

    if main_topic11_idx is not None and len(main_topic11_idx):
        hyphen_idx = main_topic11_idx[::3]
        space_idx = main_topic11_idx[1::3]
        base.loc[hyphen_idx, "UHELDSTEKST"] = [
            "personbil påkørte parkeret bil og fortsatte som fuh-flugt ukendt fører"
            for _ in hyphen_idx
        ]
        base.loc[space_idx, "UHELDSTEKST"] = [
            "fører ramte cyklist og forlod stedet som fuh flugt uden oplysninger"
            for _ in space_idx
        ]
        _write_header2(base, os.path.join(SYNTH, "base", "base_2000_2025.xlsx"))


    # ── one-off: drunk-driving flag file (read with default header, then [2:]) ─
    flag_vals = rng.choice([np.nan, 0, 999, 60, 75, 120], size=N_ACC)
    rows = [["junk", "junk"], ["junk", "junk"]] + [[flag_vals[i], int(acc_ids[i])] for i in range(N_ACC)]
    pd.DataFrame(rows, columns=["c0", "c1"]).to_excel(
        os.path.join(SYNTH, "drunk_driving_data.xlsx"), index=False)

    # ── one-off: criminal rates (read with header=None; quarters on row 2) ─────
    from analysis_data import KOMMUNE_GROUPS
    region_codes = sorted(KOMMUNE_GROUPS.keys())
    quarters = [f"{y}Q1" for y in range(2015, 2026)]
    grid = []
    grid.append([0, "", *([None] * len(quarters))])             # row 0 (col0 int keeps dtype clean)
    grid.append([0, "", *([None] * len(quarters))])             # row 1
    grid.append([0, "", *quarters])                             # row 2: quarter labels in cols 2+
    for code in region_codes:                                   # rows 3+
        grid.append([int(code), f"region_{code}", *list(rng.integers(100, 5000, size=len(quarters)))])
    maxlen = max(len(r) for r in grid)
    grid = [r + [None] * (maxlen - len(r)) for r in grid]
    pd.DataFrame(grid).to_excel(
        os.path.join(SYNTH, "criminal_rates_with_codes.xlsx"), index=False, header=False)

    print(f"Synthetic data written under {SYNTH} ({N_ACC} accidents).")


if __name__ == "__main__":
    main()
