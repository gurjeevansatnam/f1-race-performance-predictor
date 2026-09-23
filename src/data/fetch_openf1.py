import requests
import pandas as pd

url = "https://api.openf1.org/v1/sessions"

params = {"year": 2023, "session_name": "Race"}
response = requests.get(url, params=params)
data = response.json()
print(response.status_code)
df_sessions = pd.DataFrame(data)
british_gp_sessions = df_sessions[df_sessions["country_name"] == 'United Kingdom']
session_key = british_gp_sessions.iloc[0]["session_key"]

laps_url = "https://api.openf1.org/v1/laps"
laps_params = {"session_key": session_key}
laps_response = requests.get(laps_url, params=laps_params)
print(laps_response.status_code)
data_laps = laps_response.json()
df_laps = pd.DataFrame(data_laps)

stints_url = "https://api.openf1.org/v1/stints"
stints_params = {"session_key": session_key}
stints_response = requests.get(stints_url, params=stints_params)
print(stints_response.status_code)
data_stints = stints_response.json()
df_stints = pd.DataFrame(data_stints)

#merge laps and stints
df_merged = pd.merge(df_laps, df_stints, on=["session_key", "driver_number"], how="left",suffixes=('_laps', '_stints'))
df_merged = df_merged[(df_merged["lap_number"] >= df_merged["lap_start"]) & (df_merged["lap_number"] <= df_merged["lap_end"])]
df_merged["tyre_age"] = (df_merged["tyre_age_at_start"] + (df_merged["lap_number"] - df_merged["lap_start"]))


