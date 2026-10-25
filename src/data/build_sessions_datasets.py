from pathlib import Path

import pandas as pd


#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SESSIONS_DIR = RAW_DATA_DIR / "sessions"
SESSION_RESULTS_DIR = RAW_DATA_DIR / "session_results"
LAPS_DIR = RAW_DATA_DIR / "laps"
STINTS_DIR = RAW_DATA_DIR / "stints"
WEATHER_DIR = RAW_DATA_DIR / "weather"

SESSION_DATASETS_DIR = PROJECT_ROOT / "data" / "processed" / "session_datasets"
SESSION_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


#Sessions we care about
RELEVANT_SESSION_NAMES = ["Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]


# --------------------------------------------------
# Load session metadata
# --------------------------------------------------

def load_sessions(start_year, end_year):

    all_sessions = []

    for year in range(start_year, end_year + 1):

        session_file = SESSIONS_DIR / f"{year}.csv"

        if not session_file.exists():
            print(f"Session file missing for {year}. Skipping...")
            continue

        df_sessions = pd.read_csv(session_file)

        df_sessions = df_sessions[df_sessions["session_name"].isin(RELEVANT_SESSION_NAMES)].copy()

        df_sessions["year"] = year

        all_sessions.append(df_sessions)

    if not all_sessions:
        return pd.DataFrame()

    df_sessions = pd.concat(all_sessions, ignore_index=True)

    df_sessions["date_start"] = pd.to_datetime(df_sessions["date_start"], utc=True)

    df_sessions = df_sessions.sort_values(["date_start", "session_key"]).reset_index(drop=True)

    return df_sessions


# --------------------------------------------------
# Load files belonging to known sessions
# --------------------------------------------------

def load_session_files(df_sessions, data_directory):

    all_data = []

    for session_key in df_sessions["session_key"].unique():

        data_file = data_directory / f"{session_key}.csv"

        if not data_file.exists():
            continue

        df_data = pd.read_csv(data_file)

        if df_data.empty:
            continue

        all_data.append(df_data)

    if not all_data:
        return pd.DataFrame()

    df_data = pd.concat(all_data, ignore_index=True)

    return df_data


# --------------------------------------------------
# Add session information
# --------------------------------------------------

def add_session_metadata(df_data, df_sessions):

    if df_data.empty:
        return df_data

    metadata_columns = ["session_key", "meeting_key", "year", "country_name", "location", "session_name", "session_type", "date_start"]

    available_columns = [column for column in metadata_columns if column in df_sessions.columns]

    df_metadata = df_sessions[available_columns].drop_duplicates(subset=["session_key"]).copy()

    # Some OpenF1 datasets already contain meeting_key.
    # We only need session_key for this metadata merge.
    duplicate_columns = [column for column in df_metadata.columns if (column != "session_key" and column in df_data.columns)]

    df_metadata = df_metadata.drop(columns=duplicate_columns)

    df_data = pd.merge(df_data, df_metadata, on="session_key", how="left")

    return df_data


# --------------------------------------------------
# Build processed session datasets
# --------------------------------------------------

def build_session_datasets(start_year, end_year):

    print("Loading session metadata...")

    df_sessions = load_sessions(start_year, end_year)

    if df_sessions.empty:
        raise ValueError("No session metadata found.")

    print(f"Loaded {len(df_sessions)} sessions.")


    print("\nLoading session results...")
    df_session_results = load_session_files(df_sessions, SESSION_RESULTS_DIR)
    df_session_results = add_session_metadata(df_session_results, df_sessions)


    print("Loading laps...")
    df_laps = load_session_files(df_sessions, LAPS_DIR)
    df_laps = add_session_metadata(df_laps, df_sessions)


    print("Loading stints...")
    df_stints = load_session_files(df_sessions, STINTS_DIR)
    df_stints = add_session_metadata(df_stints, df_sessions)


    print("Loading weather...")
    df_weather = load_session_files(df_sessions, WEATHER_DIR)
    df_weather = add_session_metadata(df_weather, df_sessions)


    return (df_sessions, df_session_results, df_laps, df_stints, df_weather)


# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

def save_session_datasets(df_sessions, df_session_results, df_laps, df_stints, df_weather):

    datasets = {
        "sessions.csv": df_sessions,
        "session_results.csv": df_session_results,
        "laps.csv": df_laps,
        "stints.csv": df_stints,
        "weather.csv": df_weather
    }

    for filename, dataframe in datasets.items():

        output_file = SESSION_DATASETS_DIR / filename

        dataframe.to_csv(output_file, index=False)

        print(f"Saved {filename}: {len(dataframe)} rows")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    START_YEAR = 2023
    END_YEAR = 2025

    (df_sessions, df_session_results, df_laps, df_stints, df_weather) = build_session_datasets(START_YEAR, END_YEAR)

    save_session_datasets(df_sessions, df_session_results, df_laps, df_stints, df_weather)

    print("\nSession dataset processing complete.")