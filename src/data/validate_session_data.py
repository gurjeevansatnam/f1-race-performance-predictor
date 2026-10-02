from pathlib import Path
import pandas as pd

#DIR
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SESSIONS_DIR = RAW_DATA_DIR / "sessions"



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
    
    
    df_sessions = load_expected_sessions(START_YEAR,
    END_YEAR)

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


