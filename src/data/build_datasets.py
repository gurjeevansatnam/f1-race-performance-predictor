from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RACE_RESULTS_DIR = (PROJECT_ROOT/ "data"/ "processed"/ "race_results")
DATASETS_DIR = (PROJECT_ROOT/ "data"/ "processed"/ "datasets")

DATASETS_DIR.mkdir(parents=True,exist_ok=True)

def load_race_results():

    race_files = sorted(RACE_RESULTS_DIR.glob("*.csv"))
    all_races = []

    for race_file in race_files:
        df_race = pd.read_csv(race_file)
        all_races.append(df_race)

    df_races = pd.concat(all_races,ignore_index=True)

    return df_races

def load_race_dates():
    session_files = sorted((PROJECT_ROOT / "data" / "raw" / "sessions").glob("*.csv"))
    all_sessions = []
    for session_file in session_files:
        df_sessions = pd.read_csv(session_file)
        df_races = df_sessions[df_sessions["session_name"] == "Race"].copy()
        all_sessions.append(df_races)

    df_race_dates = pd.concat(all_sessions, ignore_index=True)
    return df_race_dates[["session_key","date_start"]]


def build_historical_race_dataset():
    df_races = load_race_results()
    df_race_dates = load_race_dates()
    df_races = pd.merge(df_races, df_race_dates, on="session_key", how="left")
    df_races["race_date"] = pd.to_datetime(df_races["date_start"], utc=True)
    df_races = df_races.sort_values(["race_date","position"]).reset_index(drop=True)
    validate_race_dataset(df_races)
    return df_races

def validate_race_dataset(df_races):
    #check for missing race dates
    if df_races["race_date"].isna().any():
        raise ValueError("Missing race dates found.")
    #check duplicate drivers
    if df_races.duplicated(subset=["session_key", "driver_number"]).any():
        raise ValueError("Duplicate driver entries found")
    #race should only have one winner
    winners_per_race = (df_races.groupby("session_key")["winner"].sum())
    if (winners_per_race != 1).any():
        raise ValueError("Invalid winner count found")


def save_historical_race_dataset(df_races):
    output_file = (DATASETS_DIR/ "historical_race_results.csv")
    df_races.to_csv(output_file, index=False)
    print(f"Saved historical race dataset to {output_file}")


########################################################################################
if __name__ == "__main__":

    df_historical_races = (build_historical_race_dataset())

    save_historical_race_dataset(df_historical_races)

    print(
        f"Dataset contains "
        f"{len(df_historical_races)} driver-race rows "
        f"across "
        f"{df_historical_races['session_key'].nunique()} races."
    )

