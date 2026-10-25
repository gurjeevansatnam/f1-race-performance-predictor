import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACES_FILE = PROJECT_ROOT / "data" / "processed" / "datasets" / "historical_race_results.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "session_datasets" / "session_results.csv"

def load_my_data():
    #load files, races and qualifying sessions
    races = pd.read_csv(RACES_FILE)
    results = pd.read_csv(RESULTS_FILE)
    quali = results[results["session_name"] == "Qualifying"]
    return races, quali

def winners_with_grid(races, quali):
    #get winners with qualifying grid positions
    winners = races[races["winner"] == 1]
    grid = quali[["meeting_key", "driver_number", "position"]]
    grid = grid.rename(columns={"position": "quali_pos"})
    return winners.merge(grid, on=["meeting_key", "driver_number"], how="left")

def pole_win_rate_by_year(winners):
    #post qualifying baseline, get average pole win rate by year
    from_pole = winners["quali_pos"] == 1
    return from_pole.groupby(winners["year"]).mean()

def winner_grid_counts(winners, years):
    #get counts of winning grid positions for given years
    recent = winners[winners["year"].isin(years)]
    return recent["quali_pos"].value_counts().sort_index()

def previous_winner_repeat_rate(winners, years):
    #pre qualifying baseline, get share of races won by the same driver who won the race before
    winners = winners.sort_values("race_date")
    previous = winners["name_acronym"].shift(1)
    repeat = winners["name_acronym"] == previous
    return repeat[winners["year"].isin(years)].mean()


if __name__ == "__main__":
    races, quali = load_my_data()
    winners = winners_with_grid(races, quali)
    print(winners[["year", "race_name", "full_name", "quali_pos"]].head(10), "\n")
    print("Pole win rate by year:\n", pole_win_rate_by_year(winners), "\n")
    print("Winning grid positions 2024-25:\n", winner_grid_counts(winners, [2024, 2025]), "\n")
    print("Previous winner wins again (2024-25):", previous_winner_repeat_rate(winners, [2024, 2025]))