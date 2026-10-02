from openf1_client import (
    get_sessions,
    get_session_results,
    get_laps,
    get_stints,
    get_weather
)

RELEVANT_SESSION_NAMES = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Sprint Qualifying",
    "Sprint Shootout",
    "Sprint",
    "Qualifying",
    "Race"
]

def fetch_data_for_session(session_key):
    get_session_results(session_key)
    get_laps(session_key)
    get_stints(session_key)
    get_weather(session_key)

def fetch_session_data(start_year, end_year):
    for year in range(start_year, end_year + 1):
        print(f"fetching session data for {year}...")
        df_sessions = get_sessions(year)
        if df_sessions.empty:
            print(f'no session data available for {year}. Skipping...')
            continue

        df_relevant_sessions = df_sessions[df_sessions["session_name"].isin(RELEVANT_SESSION_NAMES)].copy()

        for _, session in df_relevant_sessions.iterrows():
            session_key = session["session_key"]
            session_name = session["session_name"]
            country_name = session["country_name"]
            fetch_data_for_session(session_key)


def load_expected_sessions(start_year, end_year):
    all_sessions = []

    for year in range(start_year, end_year + 1):
        session_file = (SESSIONS_DIR / f"{year}.csv")
        df_sessions = pd.read_csv(session_file)
        df_sessions = df_sessions[df_sessions["session_name"].isin(RELEVANT_SESSION_NAMES)].copy()
        all_sessions.append(df_sessions)

    df_sessions = pd.concat(all_sessions,ignore_index=True)

    return df_sessions







if __name__ == "__main__":

    START_YEAR = 2023
    END_YEAR = 2025

    fetch_session_data(
        START_YEAR,
        END_YEAR
    )


    load_expected_sessions(
    START_YEAR,
    END_YEAR)   

    df_sessions = load_expected_sessions()

    print(
        df_sessions[
            [
                "session_key",
                "country_name",
                "session_name"
            ]
        ]
    )

    print(
        f"\nTotal relevant sessions: "
        f"{len(df_sessions)}"
    )

    print(
        "\nSessions by type:"
    )

    print(
        df_sessions["session_name"]
        .value_counts()
    )

    


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       