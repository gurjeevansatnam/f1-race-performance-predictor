# gets all the data up to date (including this season) and rebuilds model_dataset.csv
# the github job runs this before every live prediction
#
# run: python src/data/update_data.py
#
# 1. download any quali/race results i don't have yet - anything already downloaded
#    is saved in data/raw so it only fetches new races
# 2. rebuild sessions.csv + session_results.csv
# 3. rebuild historical_race_results.csv
# 4. rebuild model_dataset.csv
#
# only gets quali and race data, laps/tyres/weather aren't used by the models

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent / "features"))

from openf1_client import (SESSIONS_DIR, get_drivers, get_openf1_data,  # noqa: E402
                           get_session_results, get_sessions)
from fetch_race_results import RACE_RESULTS_DIR, build_race_results  # noqa: E402
from build_sessions_datasets import (SESSION_DATASETS_DIR, SESSION_RESULTS_DIR,  # noqa: E402
                                     add_session_metadata, load_session_files, load_sessions)
from build_datasets import build_historical_race_dataset, save_historical_race_dataset  # noqa: E402
from build_model_dataset import OUTPUT_FILE, build_model_dataset  # noqa: E402

FIRST_YEAR = 2023
# results can still change for a few hours after a session (penalties, dsq's etc)
# so only save them once the session finished at least this long ago
WAIT_HOURS = 6


def sessions_for(year, current_year):
    # old seasons don't change so the saved file is fine, current season always gets downloaded fresh
    if year < current_year:
        return get_sessions(year)
    sessions = get_openf1_data("sessions", {"year": year})
    if not sessions.empty:
        sessions.to_csv(SESSIONS_DIR / f"{year}.csv", index=False)
    return sessions


def finished(sessions, now):
    # quali and race sessions that ended at least WAIT_HOURS ago
    s = sessions[sessions["session_name"].isin(["Qualifying", "Race"])]
    if "is_cancelled" in s:
        s = s[s["is_cancelled"].astype(str).str.lower() != "true"]
    ended = pd.to_datetime(s["date_end"], utc=True)
    return s[ended < now - pd.Timedelta(hours=WAIT_HOURS)]


def download_new_results(year, sessions, now):
    new = 0
    for _, s in finished(sessions, now).iterrows():
        key = s["session_key"]
        if s["session_name"] == "Qualifying":
            if not (SESSION_RESULTS_DIR / f"{key}.csv").exists():
                new += not get_session_results(key).empty
            continue

        race_file = RACE_RESULTS_DIR / f"{key}.csv"           # same file fetch_race_results.py makes
        if race_file.exists():
            continue
        drivers, results = get_drivers(key), get_session_results(key)
        if drivers.empty or results.empty:
            print(f"  {year} {s['circuit_short_name']}: no results on OpenF1 yet")
            continue
        race = build_race_results(results, drivers)
        race["race_name"], race["year"] = s["country_name"], year
        race.to_csv(race_file, index=False)
        new += 1
        print(f"  new race: {year} {s['circuit_short_name']}")
    return new


def update_data(now):
    current_year = now.year
    new = 0
    for year in range(FIRST_YEAR, current_year + 1):
        sessions = sessions_for(year, current_year)
        if sessions.empty:
            print(f"{year}: no sessions on OpenF1 yet")
            continue
        new += download_new_results(year, sessions, now)
    print(f"Downloaded {new} new sessions")

    # rebuild the processed files from everything in data/raw
    sessions = load_sessions(FIRST_YEAR, current_year)
    results = add_session_metadata(load_session_files(sessions, SESSION_RESULTS_DIR), sessions)
    sessions.to_csv(SESSION_DATASETS_DIR / "sessions.csv", index=False)
    results.to_csv(SESSION_DATASETS_DIR / "session_results.csv", index=False)
    save_historical_race_dataset(build_historical_race_dataset())

    dataset = build_model_dataset()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)
    last = dataset.sort_values("race_date").iloc[-1]
    print(f"model_dataset.csv: {dataset['session_key'].nunique()} races, latest = {last['race_label']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help="pretend it's this time (for testing) e.g. 2025-06-01")
    args = parser.parse_args()
    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")
    update_data(now)
