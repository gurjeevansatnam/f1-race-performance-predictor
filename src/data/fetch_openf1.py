import requests
import pandas as pd
from datetime import datetime

url = "https://api.openf1.org/v1/sessions"


def get_laps(session_key):
    laps_url = "https://api.openf1.org/v1/laps"
    laps_params = {"session_key": session_key}
    laps_response = requests.get(laps_url, params=laps_params)
    laps_data = laps_response.json()
    df_laps = pd.DataFrame(laps_data)
    return df_laps

def get_stints(session_key):
    stints_url = "https://api.openf1.org/v1/stints"
    stints_params = {"session_key": session_key}
    stints_response = requests.get(stints_url, params=stints_params)
    stints_data = stints_response.json()
    df_stints = pd.DataFrame(stints_data)
    return df_stints

def get_weather(session_key):
    weather_url = "https://api.openf1.org/v1/weather"
    weather_params = {"session_key": session_key}
    weather_response = requests.get(weather_url, params=weather_params, timeout=30)
    weather_data = weather_response.json()
    df_weather = pd.DataFrame(weather_data)
    return df_weather

def build_race_dataset(df_laps, df_stints, df_weather):
    # Merge laps and stints
    df_merged = pd.merge(df_laps, df_stints, on=["session_key", "driver_number"], how="left", suffixes=('_laps', '_stints'))
    df_merged = df_merged[(df_merged["lap_number"] >= df_merged["lap_start"]) & (df_merged["lap_number"] <= df_merged["lap_end"])]
    df_merged["tyre_age"] = (df_merged["tyre_age_at_start"] + (df_merged["lap_number"] - df_merged["lap_start"]))

    df_merged["date_start"] = pd.to_datetime(df_merged["date_start"], format="ISO8601")
    df_weather["date"] = pd.to_datetime(df_weather["date"], format="ISO8601")

    df_merged = df_merged.sort_values("date_start")
    df_weather = df_weather.sort_values("date")

    # Merge with weather data
    df_final = pd.merge_asof(
        df_merged,
        df_weather,
        left_on="date_start",
        right_on="date",
        direction="backward"
    )

    return df_final

df_laps = get_laps(session_key)
df_stints = get_stints(session_key)
df_weather = get_weather(session_key)

df_race = build_race_dataset(
    df_laps,
    df_stints,
    df_weather
)

print(df_race.shape)
print(df_race.head())