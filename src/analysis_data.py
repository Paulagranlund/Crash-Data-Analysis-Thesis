"""analysis_data.py
================
One configurable entry point that builds the dataframe every Track 2 analysis
needs, so the data paths live in config.py instead of being hardcoded in each
notebook.

    from analysis_data import build_analysis_dataframe

    df_acc   = build_analysis_dataframe(config="main_0.3")            # accident level
    df_merged = build_analysis_dataframe(level="person")             # person level

``config`` selects which semi-supervised run under results_semi/ is merged on
(``main_0.3``, ``report_accident_0.25`` or ``all_0.2``). ``level="accident"``
returns one row per accident with the full VD, severity, element and kommune
fields; ``level="person"`` keeps person rows with age and all three topic
configurations, matching the case-study notebooks.
"""
import os

import pandas as pd

from config import (
    BASE_DATA_FOLDER,
    ELEMENT_DATA_FOLDER,
    VD_FIELDS_FILES,
    SEVERITY_FILE,
    COORDS_FILE,
    RESULTS_SEMI_DIR,
    CONFIG_DIRS,
)
from data_load import load_data_for_modeling

AGE_BINS = [-2, 0, 20, 25, 35, 45, 60, 101]
AGE_LABELS = ["Unknown", "0-20", "21-25", "26-35", "36-45", "46-60", "61+"]

KOMMUNE_GROUPS = {
    101: "Hovedstad", 147: "Hovedstad", 151: "Hovedstad", 153: "Hovedstad",
    155: "Hovedstad", 157: "Hovedstad", 159: "Hovedstad", 161: "Hovedstad",
    163: "Hovedstad", 165: "Hovedstad", 167: "Hovedstad", 169: "Hovedstad",
    173: "Hovedstad", 175: "Hovedstad", 183: "Hovedstad", 185: "Hovedstad",
    187: "Hovedstad", 190: "Hovedstad", 201: "Hovedstad", 223: "Hovedstad",
    230: "Hovedstad", 240: "Hovedstad", 253: "Hovedstad", 269: "Hovedstad",
    461: "Storby", 751: "Storby", 851: "Storby",
    217: "Provinsby", 219: "Provinsby", 259: "Provinsby", 265: "Provinsby",
    330: "Provinsby", 370: "Provinsby", 561: "Provinsby", 607: "Provinsby",
    615: "Provinsby", 621: "Provinsby", 630: "Provinsby", 657: "Provinsby",
    661: "Provinsby", 730: "Provinsby", 740: "Provinsby", 791: "Provinsby",
    210: "Opland", 250: "Opland", 260: "Opland", 270: "Opland", 316: "Opland",
    320: "Opland", 329: "Opland", 336: "Opland", 340: "Opland", 350: "Opland",
    410: "Opland", 420: "Opland", 430: "Opland", 440: "Opland", 450: "Opland",
    480: "Opland", 575: "Opland", 706: "Opland", 710: "Opland", 727: "Opland",
    746: "Opland", 756: "Opland", 766: "Opland", 840: "Opland",
    306: "Land", 326: "Land", 360: "Land", 376: "Land", 390: "Land", 400: "Land",
    479: "Land", 482: "Land", 492: "Land", 510: "Land", 530: "Land", 540: "Land",
    550: "Land", 563: "Land", 573: "Land", 580: "Land", 665: "Land", 671: "Land",
    707: "Land", 741: "Land", 760: "Land", 773: "Land", 779: "Land", 787: "Land",
    810: "Land", 813: "Land", 820: "Land", 825: "Land", 846: "Land", 849: "Land",
    860: "Land",
}


def _load_base_with_topics(configs):
    """Load the cleaned base data and merge the given topic configuration(s).

    Topic assignments are joined positionally, matching the notebooks, so the
    document_topics file for each configuration must have the same length as the
    cleaned base data.
    """
    data = load_data_for_modeling(
        data_folder=BASE_DATA_FOLDER, encode_labels=False, verbose=False
    )
    df = data["df"].reset_index(drop=True)

    for cfg in configs:
        folder = CONFIG_DIRS[cfg]
        doc = pd.read_csv(os.path.join(RESULTS_SEMI_DIR, folder, "document_topics.csv"))
        if len(doc) != len(df):
            raise ValueError(
                f"document_topics for '{cfg}' has {len(doc)} rows but base data "
                f"has {len(df)}; they must match for the positional join."
            )
        col = {"main_0.3": "topic_main_0_3",
               "report_accident_0.25": "topic_acc_cat_0_25",
               "all_0.2": "topic_all_0_2"}[cfg]
        df[col] = doc["assigned_topic"].values
        df[f"prob_{col}"] = doc["assigned_topic_probability"].values

    return df


def _load_element_rows():
    """Load element/person rows (accident_id, person_age, person_number,
    element_number, element_type) from the element data folder."""
    rename = {"UHELDS_ID": "accident_id", "ALDER": "person_age",
              "PERSONNR": "person_number", "ELEMENTNR": "element_number",
              "ELEMENTART": "element_type"}
    frames = []
    for file in sorted(os.listdir(ELEMENT_DATA_FOLDER)):
        if not file.endswith(".xlsx"):
            continue
        t = pd.read_excel(os.path.join(ELEMENT_DATA_FOLDER, file), header=2).rename(columns=rename)
        t = t[t["person_age"].notna()]
        keep = [c for c in rename.values() if c in t.columns]
        frames.append(t[keep])
    return pd.concat(frames, ignore_index=True)


def _build_person(configs):
    """Person-level frame for the case studies (drunk driving, solo bicycle)."""
    df = _load_base_with_topics(configs)

    el = _load_element_rows()
    df_age = el[["accident_id", "person_age", "person_number", "element_number"]]

    df_merged = df.merge(df_age, on="accident_id", how="left")
    df_merged["person_age"] = df_merged["person_age"].fillna(-1)
    df_merged["age_group"] = pd.cut(df_merged["person_age"], bins=AGE_BINS, labels=AGE_LABELS)
    df_merged["year"] = pd.to_datetime(df_merged["accident_date"]).dt.year
    return df_merged


def _build_accident(config):
    """Accident-level frame (df_acc): one row per accident with VD, severity,
    element age/type, cyclist, coordinate and kommune fields."""
    df = _load_base_with_topics([config])

    # element age and type pivots (one row per accident)
    el = _load_element_rows()
    el_p1 = el[(el["person_number"] == 1) & (el["element_number"].isin([1, 2, 3]))].copy()

    df_age = el_p1.pivot_table(index="accident_id", columns="element_number",
                               values="person_age", aggfunc="first")
    df_age.columns = [f"element_{int(c)}_age" for c in df_age.columns]
    df_age = df_age.reindex(columns=["element_1_age", "element_2_age", "element_3_age"]).reset_index()

    df_type = el_p1.pivot_table(index="accident_id", columns="element_number",
                                values="element_type", aggfunc="first")
    df_type.columns = [f"element_{int(c)}_type" for c in df_type.columns]
    df_type = df_type.reindex(columns=["element_1_type", "element_2_type", "element_3_type"]).reset_index()

    df = df.merge(df_age, on="accident_id", how="left").merge(df_type, on="accident_id", how="left")

    # severity flags from bad_uheld (one row per accident)
    bad = pd.read_excel(SEVERITY_FILE, header=2).rename(columns={"UHELDS_ID": "accident_id"})
    df_bad = pd.DataFrame(bad["accident_id"].unique(), columns=["accident_id"]).set_index("accident_id")
    for elem in [1, 2]:
        ed = bad[bad["ELEMENTNR"] == elem]
        df_bad[f"element_{elem}_dr"] = ed.groupby("accident_id")["PERSONSKADE"].apply(lambda x: int("Dr" in x.values))
        df_bad[f"element_{elem}_alv"] = ed.groupby("accident_id")["PERSONSKADE"].apply(lambda x: int("Alv" in x.values))
    e3 = bad[bad["ELEMENTNR"] >= 3]
    df_bad["element_3plus_dr"] = e3.groupby("accident_id")["PERSONSKADE"].apply(lambda x: int("Dr" in x.values))
    df_bad["element_3plus_alv"] = e3.groupby("accident_id")["PERSONSKADE"].apply(lambda x: int("Alv" in x.values))
    df_bad = df_bad.fillna(0).astype(int).reset_index()
    df = df.merge(df_bad, on="accident_id", how="left")

    # VD accident-level fields (lable_analysis), person-level so SPRIT is per person
    vd = pd.concat(
        [pd.read_excel(f, header=2).rename(columns={"UHELDS_ID": "accident_id"}) for f in VD_FIELDS_FILES],
        ignore_index=True,
    )
    df = df.merge(vd, on="accident_id", how="left")
    df = df.rename(columns={
        "KODE_UHELDSSITUATION": "UHELDSSITUATION",
        "KODE_UHELDSART": "UHELDSART",
        "VEJR": "VEJRFORHOLD",
        "UH_UHID_UHANTALVTILS": "ANTAL_ALV_TILSKADEKOMNE",
        "UH_UHID_UHANTLETTILS": "ANTAL_LET_TILSKADEKOMNE",
    })

    # accident-level sprit flag (max across persons), then deduplicate
    df["sprit_person_flag"] = (
        df["SPRIT"].notna() & (df["SPRIT"] != 0) & (df["SPRIT"] != 999) & (df["SPRIT"] >= 51)
    ).astype(int)
    df_sprit = df.groupby("accident_id")["sprit_person_flag"].max().reset_index(name="sprit_flag")
    df_acc = (
        df.drop_duplicates("accident_id")
        .drop(columns=["SPRIT", "sprit_person_flag"], errors="ignore")
        .merge(df_sprit, on="accident_id", how="left")
    )
    df_acc["sprit_flag"] = df_acc["sprit_flag"].fillna(0).astype(int)

    # cyclist flag and max element number from element rows
    el_c = el.copy()
    el_c["has_cyclist"] = (el_c["element_type"] == "Cykl").astype(int)
    cyclist = el_c.groupby("accident_id")["has_cyclist"].max().reset_index()
    df_acc = df_acc.merge(cyclist, on="accident_id", how="left")
    df_acc["has_cyclist"] = df_acc["has_cyclist"].fillna(0).astype(int)
    maxel = el_c.groupby("accident_id")["element_number"].max().reset_index(name="max_element_number")
    df_acc = df_acc.merge(maxel, on="accident_id", how="left")

    # coordinates and kommune
    coords = pd.read_parquet(COORDS_FILE).rename(columns={"UHELDS_ID": "accident_id"})
    coords = coords[["accident_id", "KODE_UHELDKOMMUNE", "x", "y"]]
    df_acc = df_acc.merge(coords, on="accident_id", how="left")
    df_acc["kommune_group"] = df_acc["KODE_UHELDKOMMUNE"].map(KOMMUNE_GROUPS)

    df_acc["assigned_topic"] = df_acc["topic_main_0_3"]
    return df_acc


def build_analysis_dataframe(config="main_0.3", level="accident"):
    """Build the analysis dataframe.

    Parameters
    ----------
    config : which semi-supervised run to merge for the accident-level frame
             (one of CONFIG_DIRS). Ignored for level="person", which loads all
             three configurations as the case studies do.
    level : "accident" for the deduplicated df_acc, or "person" for the
            person-level df_merged.
    """
    if level == "accident":
        return _build_accident(config)
    elif level == "person":
        return _build_person(list(CONFIG_DIRS.keys()))
    raise ValueError("level must be 'accident' or 'person'")
