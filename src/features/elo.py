# elo ratings (like chess) but for f1 drivers
#
# everyone starts on 1500. after each session every pair of drivers counts as a
# head to head - whoever finished ahead "wins" it
#   expected = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
#   change   = K * (actual - expected), winner gets +change, loser gets -change
# so beating someone highly rated gets you a lot more than beating a backmarker
#
# quali elo (from quali positions) goes in the post quali model
# race elo (from finishing positions) goes in the pre quali model
#
# ratings are saved BEFORE each race gets applied, otherwise the feature would
# already contain the result of the race we're trying to predict (leakage)
#
# demo: python src/features/elo.py

from pathlib import Path

import pandas as pd

START_RATING = 1500.0
# how fast ratings move after each session
K_FACTOR = 8.0


def expected_score(rating_a, rating_b):
    # chance A beats B (0 to 1) based on the rating gap
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_ratings(ratings, results, k=K_FACTOR):
    # results is like {"VER": 1, "NOR": 2, ...} lower = better

    # ratings before this session
    old = {d: ratings.get(d, START_RATING) for d in results}
    new = dict(old)
    drivers = list(results)
    for i, a in enumerate(drivers):
        for b in drivers[i + 1:]:                 # every pair, once
            if results[a] == results[b]:
                continue                          # a tie - nobody wins
            actual = 1.0 if results[a] < results[b] else 0.0
            change = k * (actual - expected_score(old[a], old[b]))
            new[a] += change
            new[b] -= change
    ratings.update(new)
    return ratings


# goes through the races in date order - saves everyone's rating before the race,
# then updates with that race's results
# returns the before table (the feature) + the final ratings (for predicting the next race)
def elo_before_each_race(df, driver_col, result_col, k=K_FACTOR):
    ratings, rows = {}, []
    for session_key, race in df.sort_values("race_date").groupby("session_key", sort=False):
        results = dict(zip(race[driver_col], race[result_col]))
        for driver in results:
            rows.append((session_key, driver, ratings.get(driver, START_RATING)))   # record before
        update_ratings(ratings, results, k)                                         # then update
    before = pd.DataFrame(rows, columns=["session_key", driver_col, "elo"])
    return before, ratings


if __name__ == "__main__":
    print("--- Worked example: two drivers, both 1500 ---")
    r = update_ratings({"A": 1500.0, "B": 1500.0}, {"A": 1, "B": 2})
    print(f"A beats B: A = {r['A']:.0f}, B = {r['B']:.0f}")

    print("\n--- Upset: a 1500 driver beats a 1700 driver ---")
    r = update_ratings({"A": 1500.0, "B": 1700.0}, {"A": 1, "B": 2})
    print(f"A beats B: A = {r['A']:.1f} (+{r['A'] - 1500:.1f}), B = {r['B']:.1f}")

    dataset_file = Path(__file__).resolve().parents[2] / "data" / "processed" / "features" / "model_dataset.csv"
    df = pd.read_csv(dataset_file)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)

    _, quali_ratings = elo_before_each_race(df, "name_acronym", "quali_pos")
    df["finish_pos"] = df["position"].fillna(25)          # dnf = last
    _, race_ratings = elo_before_each_race(df, "name_acronym", "finish_pos")

    top = pd.DataFrame({"quali_elo": quali_ratings, "race_elo": race_ratings}).round(0)
    print("\n--- Ratings after the last race in the data, top 8 by qualifying Elo ---")
    print(top.sort_values("quali_elo", ascending=False).head(8).to_string())

    before, _ = elo_before_each_race(df, "name_acronym", "quali_pos")
    first = before["session_key"].iloc[0]
    print("\nFirst race in the data - everyone starts at 1500:",
          before[before["session_key"] == first]["elo"].unique())
