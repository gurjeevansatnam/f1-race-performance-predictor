from pathlib import Path
import pandas as pd

from openf1_client import (
    get_sessions,
    get_drivers,
    get_session_results
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RACE_RESULTS_DIR = PROCESSED_DATA_DIR / "race_results"
RACE_RESULTS_DIR.mkdir(parents=True,exist_ok=True)


#build race results dataframe
def build_race_results(df_session_results, df_drivers):

    df_session_results = df_session_results.copy()
    df_session_results["winner"] = (df_session_results["position"] == 1).astype(int)
    df_race_results = pd.merge(df_session_results, df_drivers,
        on=[
            "session_key",
            "meeting_key",
            "driver_number"
        ],
        how="left"
    )

    return df_race_results


def get_historical_results(start_year, end_year):
    all_results = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching race results for {year}...")
        df_sessions = get_sessions(year)
        df_races = df_sessions[df_sessions["session_name"] == "Race"].copy()

        for _, race in df_races.iterrows():
            session_key = race["session_key"]
            result_file = RACE_RESULTS_DIR / f"{session_key}.csv"
            print(f'Fetching {year} {race["country_name"]} GP')
            if result_file.exists():
                print("loading saved race data")
                df_race_results = pd.read_csv(result_file)
            else:
                print("downloading race data")
                df_drivers = get_drivers(session_key)
                if df_drivers.empty:
                    print(
                        f'Skipping {year} {race["country_name"]} GP '
                        f'- no driver data available'
                    )
                    continue
                df_session_results = get_session_results(session_key)
                if df_session_results.empty:
                    print(f'skipping {year} {race["country_name"]} GP' f'- no session data available')
                    continue

                df_race_results = build_race_results(df_session_results, df_drivers)

                df_race_results["race_name"] = race["country_name"]
                df_race_results["year"] = year
                df_race_results.to_csv(result_file, index=False)

            all_results.append(df_race_results)

    df_historical_results = pd.concat(all_results, ignore_index=True)

    return df_historical_results



if __name__ == "__main__":

    START_YEAR = 2023
    END_YEAR = 2025

    df_historical_results = get_historical_results( START_YEAR, END_YEAR)
    print("completed")

