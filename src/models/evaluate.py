# tests the models fairly using walk forward testing
#
# for every race in 2024 and 2025 (48 races):
#   train only on races before it -> predict it -> score it
# that's how it would actually work in real life, you never get to train on the future
#
# scores:
#   top1     - did the favourite win
#   top3     - was the winner in the top 3 picks
#   log_loss - -log(chance it gave the actual winner), lower = better
#              punishes being confidently wrong
#
# run: python src/models/evaluate.py

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from race_models import (POST_FEATURES, PRE_FEATURES, GridPositionBaseline,  # noqa: E402
                         PreviousWinnerBaseline, post_qualifying_model, pre_qualifying_model)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "processed" / "features" / "model_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_YEARS = [2024, 2025]

# name: (function that makes a new model, features it uses)
MODELS = {
    "PRE baseline (last winner)":  (PreviousWinnerBaseline, None),
    "PRE model":                   (pre_qualifying_model, PRE_FEATURES),
    "POST baseline (grid slot)":   (GridPositionBaseline, None),
    "POST model":                  (post_qualifying_model, POST_FEATURES),
}


def load_dataset():
    df = pd.read_csv(DATASET)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    return df.sort_values(["race_date", "quali_pos"]).reset_index(drop=True)


def race_metrics(race, prob):
    # scores one race. prob = the model's win chance for each driver
    race = race.assign(prob=prob)
    # ties broken alphabetically - NOT by grid position, the pre quali model
    # isn't allowed to know that (would be leakage)
    ranked = race.sort_values(["prob", "name_acronym"], ascending=[False, True])
    winner = race[race["winner"] == 1].iloc[0]
    return {
        "race": winner["race_label"],
        "year": winner["year"],
        "predicted": ranked["name_acronym"].iloc[0],
        "predicted_prob": ranked["prob"].iloc[0],
        "actual": winner["name_acronym"],
        "winner_prob": winner["prob"],
        "top1": int(ranked["winner"].iloc[0] == 1),
        "top3": int(ranked["winner"].iloc[:3].sum() == 1),
        "log_loss": -np.log(max(winner["prob"], 1e-6)),
    }


def walk_forward(df, make_model, features, test_years=TEST_YEARS):
    # predict every race in test_years, training only on the races before it each time
    rows = []
    races = df.drop_duplicates("session_key")
    for _, r in races[races["year"].isin(test_years)].iterrows():
        train = df[df["race_date"] < r["race_date"]]            # only the past
        test = df[df["session_key"] == r["session_key"]]
        model = make_model().fit(train, features)
        rows.append(race_metrics(test, model.predict_proba(test)))
    return pd.DataFrame(rows)


def summarise(per_race):
    # average the per race scores into one row
    return pd.Series({
        "winner_right": per_race["top1"].mean(),
        "winner_in_top3": per_race["top3"].mean(),
        "log_loss": per_race["log_loss"].mean(),
        "right_2024": per_race.loc[per_race["year"] == 2024, "top1"].mean(),
        "right_2025": per_race.loc[per_race["year"] == 2025, "top1"].mean(),
    })


if __name__ == "__main__":
    df = load_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)

    per_race, summary = {}, {}
    for name, (make_model, features) in MODELS.items():
        print(f"  testing {name} ...")
        per_race[name] = walk_forward(df, make_model, features)
        summary[name] = summarise(per_race[name])
    summary = pd.DataFrame(summary).T

    pd.set_option("display.width", 150)
    print("\nWALK-FORWARD RESULTS - 48 races, 2024-2025 (each predicted using only earlier races)\n")
    print(summary.round(3).to_string())
    summary.round(4).to_csv(RESULTS_DIR / "model_results.csv", index_label="model")

    # save the race by race predictions for both final models
    for stage in ["PRE", "POST"]:
        per_race[f"{stage} model"].to_csv(RESULTS_DIR / f"{stage.lower()}_model_per_race.csv", index=False)

    both = per_race["PRE model"][["race", "predicted", "winner_prob"]].rename(
        columns={"predicted": "pre_pick", "winner_prob": "pre_winner_prob"})
    both[["post_pick", "post_winner_prob", "actual"]] = per_race["POST model"][["predicted", "winner_prob", "actual"]]
    print("\nLast 6 races of 2025 - each model's pick vs the actual winner:")
    print(both.tail(6).round(2).to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")
