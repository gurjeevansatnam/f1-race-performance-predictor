# builds the table the models train on - one row per driver per race
#
# needs these first (from my data scripts):
#   build_sessions_datasets.py -> sessions.csv + session_results.csv
#   build_datasets.py          -> historical_race_results.csv
# saves to data/processed/features/model_dataset.csv
#
# pre quali model uses:  race_elo_rel, quali_pos_last5
# post quali model uses: log_quali_pos, quali_gap_capped, quali_elo_rel

import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))   # so elo.py can be imported
from elo import elo_before_each_race

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = PROJECT_ROOT / "data" / "processed" / "session_datasets"
RACES_FILE = PROJECT_ROOT / "data" / "processed" / "datasets" / "historical_race_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features" / "model_dataset.csv"

MISSING_QUALI_POS = 20    # no quali time = start at the back
GAP_CAP_PCT = 2.0         # anything more than 2% off pole just counts as 2%
DRIVER = "name_acronym"   # use VER, HAM etc - car numbers change (bearman had 3 different ones)
FORM_WINDOW = 5           # "recent" = last 5 races
NO_HISTORY_QUALI = 10     # new driver with no races yet = treat as midfield


def load_race_table(sessions, races):
    race_sessions = sessions[sessions["session_name"] == "Race"]
    race_sessions = race_sessions[["session_key", "meeting_key", "date_start", "circuit_short_name", "year"]]
    race_sessions = race_sessions.rename(columns={"date_start": "race_date"})

    columns = ["session_key", "driver_number", "full_name", "name_acronym", "team_name",
               "position", "points", "winner", "dnf"]

    df = races[columns].merge(race_sessions, on="session_key", how="inner")

    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    df["points"] = df["points"].fillna(0)
    df["race_label"] = df["year"].astype(str) + " " + df["circuit_short_name"]
    return df


def qualifying_positions(results):
    quali = results[results["session_name"] == "Qualifying"]
    quali = quali[["meeting_key", "driver_number", "position"]]
    return quali.rename(columns={"position": "quali_pos"})


# quali times are a list [Q1, Q2, Q3] but the csv saves it as text like "[75.582, 75.455, 75.372]"
# this turns it back into a proper list
def as_list(value):
    if isinstance(value, (list, tuple)):            # already a list (live api gives lists)
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [None, None, None]                    # no times at all
    parsed = literal_eval(str(value))
    return parsed if isinstance(parsed, list) else [parsed]


# % slower than pole, but compared inside the last session each driver got to (Q1/Q2/Q3)
# can't just compare best laps because the track gets quicker through quali -
# hungary 2025 norris did 74.890 in Q2, faster than leclerc's pole lap in Q3 (75.372)
def qualifying_gap_pct(quali):
    q = quali[["meeting_key", "driver_number", "duration", "gap_to_leader"]].copy()
    q["laps"] = q["duration"].apply(as_list)
    q["gaps"] = q["gap_to_leader"].apply(as_list)

    def last_segment(row):
        done = [i for i, lap in enumerate(row["laps"]) if lap is not None]
        if not done:                                  # didn't set a time
            return pd.Series({"segment": np.nan, "gap_pct": np.nan})
        s = done[-1]                                  # 0 = Q1, 1 = Q2, 2 = Q3
        lap, gap = row["laps"][s], row["gaps"][s]
        gap_pct = 100 * gap / (lap - gap) if gap is not None else np.nan
        return pd.Series({"segment": s, "gap_pct": gap_pct})

    q = pd.concat([q, q.apply(last_segment, axis=1)], axis=1)

    # someone knocked out in Q1/Q2 should never look closer to pole than a Q3 driver
    top_segment = q.groupby("meeting_key")["segment"].transform("max")
    in_top = q["segment"] == top_segment
    slowest_top = q["gap_pct"].where(in_top).groupby(q["meeting_key"]).transform("max")
    q["quali_gap_pct"] = np.where(in_top, q["gap_pct"], np.maximum(q["gap_pct"], slowest_top) + 0.01)
    return q[["meeting_key", "driver_number", "quali_gap_pct"]]


def qualifying_features(results):
    # grid position + gap to pole for every driver
    quali = results[results["session_name"] == "Qualifying"]
    return qualifying_positions(results).merge(qualifying_gap_pct(quali),
                                               on=["meeting_key", "driver_number"])


def add_model_inputs(df):
    df = df.copy()
    df["quali_pos"] = pd.to_numeric(df["quali_pos"], errors="coerce")   # openf1 sometimes has text in here e.g. "RT" (singapore 2026)
    df["quali_missing"] = df["quali_pos"].isna().astype(int)
    df["quali_pos"] = df["quali_pos"].fillna(MISSING_QUALI_POS)
    df["log_quali_pos"] = np.log(df["quali_pos"])
    # no quali time -> give them the slowest gap in that race
    worst = df.groupby("session_key")["quali_gap_pct"].transform("max")
    df["quali_gap_pct"] = df["quali_gap_pct"].fillna(worst).fillna(GAP_CAP_PCT)
    df["quali_gap_capped"] = df["quali_gap_pct"].clip(upper=GAP_CAP_PCT)
    return df


def add_elo_features(df):
    # both elo ratings as they were before each race

    # quali elo - from past quali positions (post quali model)
    before, _ = elo_before_each_race(df, DRIVER, "quali_pos")
    df = df.merge(before.rename(columns={"elo": "quali_elo"}), on=["session_key", DRIVER], how="left")

    # race elo - from past finishing positions, dnf = last (pre quali model)
    df["finish_pos"] = df["position"].fillna(25)
    before, _ = elo_before_each_race(df, DRIVER, "finish_pos")
    df = df.merge(before.rename(columns={"elo": "race_elo"}), on=["session_key", DRIVER], how="left")
    df = df.drop(columns="finish_pos")

    # the model compares drivers within a race, so make it relative to the best rated
    # driver in that race (best = 0, everyone else negative)
    for col in ["quali_elo", "race_elo"]:
        df[f"{col}_rel"] = df[col] - df.groupby("session_key")[col].transform("max")
    return df


def rolling_mean_before(df, column, window):
    # average over each driver's last few races
    # shift(1) leaves out the current race so it only uses the past
    return df.groupby(DRIVER)[column].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def add_form_features(df):
    df = df.sort_values(["race_date", DRIVER]).copy()

    # avg quali position over the last 5 races (pre quali model)
    df["quali_pos_last5"] = rolling_mean_before(df, "quali_pos", FORM_WINDOW).fillna(NO_HISTORY_QUALI)

    # points form - tested this in analysis.py and it doesn't help, kept it to show that
    df["driver_form_points"] = rolling_mean_before(df, "points", FORM_WINDOW).fillna(0)
    team = (df.groupby(["session_key", "race_date", "team_name"], as_index=False)["points"].sum()
              .sort_values("race_date"))
    team["team_form_points"] = team.groupby("team_name")["points"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    df = df.merge(team[["session_key", "team_name", "team_form_points"]],
                  on=["session_key", "team_name"], how="left")
    df["team_form_points"] = df["team_form_points"].fillna(0)
    return df


def validate(df):
    assert not df.duplicated(["session_key", "driver_number"]).any(), "a driver appears twice in one race"
    assert (df.groupby("session_key")["winner"].sum() == 1).all(), "every race needs exactly one winner"
    assert df["race_date"].notna().all(), "some races have no date"
    assert df["quali_elo"].notna().all() and df["race_elo"].notna().all(), "some drivers have no Elo"


def build_model_dataset():
    #load data
    sessions = pd.read_csv(SESSION_DIR / "sessions.csv")
    results = pd.read_csv(SESSION_DIR / "session_results.csv")
    races = pd.read_csv(RACES_FILE)

    #build df
    df = load_race_table(sessions, races)
    df = df.merge(qualifying_features(results), on=["meeting_key", "driver_number"], how="left")
    df = add_model_inputs(df)
    df = add_elo_features(df)
    df = add_form_features(df)
    df = df.sort_values(["race_date", "quali_pos"]).reset_index(drop=True)

    validate(df)
    return df


if __name__ == "__main__":
    dataset = build_model_dataset()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(dataset)} rows / {dataset['session_key'].nunique()} races -> {OUTPUT_FILE}\n")

    pre = ["race_elo_rel", "quali_pos_last5"]
    post = ["log_quali_pos", "quali_gap_capped", "quali_elo_rel"]
    race = dataset[dataset["race_label"] == "2025 Yas Marina Circuit"]
    print("Abu Dhabi 2025 - what each model will see (top 5 on the grid):")
    print(race[["name_acronym"] + pre + post + ["winner"]].head(5).round(2).to_string(index=False), "\n")

    first = dataset[dataset["race_date"] == dataset["race_date"].min()]
    print("First race in the data (no history yet): Elo =", first["quali_elo"].unique(),
          "| quali_pos_last5 =", first["quali_pos_last5"].unique())
    print("Missing values in model features:", int(dataset[pre + post].isna().sum().sum()))
