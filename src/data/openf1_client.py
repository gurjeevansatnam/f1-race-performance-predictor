from pathlib import Path
import time

import pandas as pd
import requests

#API endpoint 
BASE_URL = "https://api.openf1.org/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#DIR
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SESSIONS_DIR = RAW_DATA_DIR / "sessions"
DRIVERS_DIR = RAW_DATA_DIR / "drivers"
SESSION_RESULTS_DIR = RAW_DATA_DIR / "session_results"
LAPS_DIR = RAW_DATA_DIR / "laps"
STINTS_DIR = RAW_DATA_DIR / "stints"
WEATHER_DIR = RAW_DATA_DIR / "weather"

SESSIONS_DIR.mkdir(parents=True,exist_ok=True)
DRIVERS_DIR.mkdir(parents=True,exist_ok=True)
SESSION_RESULTS_DIR.mkdir(parents=True,exist_ok=True)
LAPS_DIR.mkdir(parents=True, exist_ok=True)
STINTS_DIR.mkdir(parents=True, exist_ok=True)
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

#Get Raw data
def get_openf1_data(endpoint, params, max_retries=5):
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(max_retries):
        time.sleep(0.5)
        response = requests.get(url,params=params,timeout=30)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait_time = float(retry_after)
            else:
                wait_time = 2 ** attempt
            print(f"Rate limit reached. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            continue

        if response.status_code == 404:
            return pd.DataFrame()

        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected OpenF1 response: {data}")
        return pd.DataFrame(data)
    raise RuntimeError(
        f"Failed to get {endpoint} after {max_retries} attempts.")

#Get all sessions for a given year
def get_sessions(year):
    session_file = SESSIONS_DIR / f"{year}.csv"

    if session_file.exists():
        print(f"Loading saved sessions for {year}")
        return pd.read_csv(session_file)

    params = {"year": year}
    df_sessions = get_openf1_data("sessions",params)

    if not df_sessions.empty:
        df_sessions.to_csv(session_file, index=False)

    return df_sessions

#Get all drivers for a season
def get_drivers(session_key):
    driver_file = DRIVERS_DIR / f"{session_key}.csv"
    if driver_file.exists():
        print(f"Loading saved drivers for session {session_key}")
        return pd.read_csv(driver_file)

    drivers_params = {"session_key": session_key}
    df_drivers = get_openf1_data("drivers", drivers_params)
    if not df_drivers.empty:
        df_drivers.to_csv(driver_file, index=False)

    return df_drivers

#Get all sessions for a grand prix weekend
def get_weekend_sessions(df_sessions, meeting_key):
    df_weekend = df_sessions[df_sessions["meeting_key"] == meeting_key].copy()
    return df_weekend

#get session results for a given session key
def get_session_results(session_key):
    session_result_file = (SESSION_RESULTS_DIR / f"{session_key}.csv")
    if session_result_file.exists():
        print(f"Loading saved session result for {session_key}")
        return pd.read_csv(session_result_file)

    params = {"session_key": session_key}

    df_session_results = get_openf1_data("session_result",params)

    if not df_session_results.empty:
        df_session_results.to_csv(session_result_file,index=False)

    return df_session_results

#get lap data
def get_laps(session_key):
    lap_file = LAPS_DIR / f"{session_key}.csv"

    if lap_file.exists():
        print(f"Loading saved laps for session {session_key}")
        return pd.read_csv(lap_file)

    params = {"session_key": session_key}
    df_laps = get_openf1_data("laps",params)

    if not df_laps.empty:
        df_laps.to_csv(lap_file,index=False)

    return df_laps

def get_stints(session_key):
    stint_file = STINTS_DIR / f"{session_key}.csv"

    if stint_file.exists():
        print(f"Loading saved stints for session {session_key}")
        return pd.read_csv(stint_file)

    params = {"session_key": session_key}

    df_stints = get_openf1_data("stints",params)

    if not df_stints.empty:
        df_stints.to_csv(stint_file,index=False)

    return df_stints


def get_weather(session_key):
    weather_file = WEATHER_DIR / f"{session_key}.csv"

    if weather_file.exists():
        print(f"Loading saved weather for session {session_key}")
        return pd.read_csv(weather_file)

    params = {"session_key": session_key}
    df_weather = get_openf1_data("weather", params)

    if not df_weather.empty:
        df_weather.to_csv(weather_file, index=False)
    return df_weather


