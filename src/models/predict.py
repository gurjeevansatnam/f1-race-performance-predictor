"""
predict the winner of a race - before quali (pre) or after quali (post)

live (needs internet, uses openf1):
    python src/models/predict.py --stage pre  --year 2026 --race "Austin"
    python src/models/predict.py --stage post --year 2026 --race "Austin"

backtest a past race (that race and everything after it gets hidden):
    python src/models/predict.py --stage post --backtest "2025 Yas Marina Circuit"

--race can be the circuit, city or country e.g. "Monza", "Las Vegas", "Japan"
saves results/prediction_<stage>_<race>.csv and .png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for folder in [HERE, PROJECT_ROOT / "src" / "data", PROJECT_ROOT / "src" / "features"]:
    sys.path.append(str(folder))

from build_model_dataset import (DRIVER, FORM_WINDOW, NO_HISTORY_QUALI,  # noqa: E402
                                 add_model_inputs, qualifying_gap_pct)
from elo import START_RATING, elo_before_each_race  # noqa: E402
from evaluate import RESULTS_DIR, load_dataset  # noqa: E402
from race_models import (POST_FEATURES, PRE_FEATURES,  # noqa: E402
                         post_qualifying_model, pre_qualifying_model)

SESSION_DIR = PROJECT_ROOT / "data" / "processed" / "session_datasets"
DRIVER_COLS = ["driver_number", "full_name", "name_acronym", "team_name"]


# features for the one race we're predicting
# these HAVE to be worked out the same way as in build_model_dataset.py
# otherwise the model gets numbers that mean something different to what it trained on


def pre_qualifying_features(drivers, history):
    # race elo + avg quali position over the last 5 races, using everything in history
    race = drivers[DRIVER_COLS].drop_duplicates(DRIVER).copy()

    finished = history.assign(finish_pos=history["position"].fillna(25))   # dnf = last
    _, ratings = elo_before_each_race(finished, DRIVER, "finish_pos")
    race["race_elo"] = race[DRIVER].map(ratings).fillna(START_RATING)      # new driver = 1500
    race["race_elo_rel"] = race["race_elo"] - race["race_elo"].max()

    last5 = (history.sort_values("race_date").groupby(DRIVER)["quali_pos"]
                    .apply(lambda s: s.tail(FORM_WINDOW).mean()))
    race["quali_pos_last5"] = race[DRIVER].map(last5).fillna(NO_HISTORY_QUALI)
    race["session_key"] = 0                     # all rows are the same race
    return race


def post_qualifying_features(quali, drivers, history):
    # grid position, gap to pole, quali elo
    race = quali[["meeting_key", "driver_number", "position", "duration", "gap_to_leader"]].copy()
    race = race.merge(qualifying_gap_pct(race), on=["meeting_key", "driver_number"], how="left")
    race = race.rename(columns={"position": "quali_pos"})
    race = race.merge(drivers[DRIVER_COLS].drop_duplicates("driver_number"), on="driver_number", how="left")
    race["session_key"] = 0
    race = add_model_inputs(race)

    _, ratings = elo_before_each_race(history, DRIVER, "quali_pos")
    race["quali_elo"] = race[DRIVER].map(ratings).fillna(START_RATING)
    race["quali_elo_rel"] = race["quali_elo"] - race["quali_elo"].max()
    return race


def predict(stage, history, drivers, quali=None):
    # train on all of history then give every driver a win chance
    if stage == "pre":
        model = pre_qualifying_model().fit(history, PRE_FEATURES)
        race = pre_qualifying_features(drivers, history)
    else:
        model = post_qualifying_model().fit(history, POST_FEATURES)
        race = post_qualifying_features(quali, drivers, history)
    race["win_probability"] = model.predict_proba(race)
    race = race.sort_values(["win_probability", DRIVER], ascending=[False, True]).reset_index(drop=True)
    race.index += 1                                   # rank from 1
    return race


def load_backtest(label, history):
    # a past race from my own data files
    sessions = pd.read_csv(SESSION_DIR / "sessions.csv")
    results = pd.read_csv(SESSION_DIR / "session_results.csv")
    sessions["label"] = sessions["year"].astype(str) + " " + sessions["circuit_short_name"]
    q = sessions[(sessions["label"].str.lower() == label.lower()) & (sessions["session_name"] == "Qualifying")]
    if q.empty:
        raise SystemExit(f"No race called '{label}' in my data. Example: '2025 Monza'")
    q = q.iloc[0]
    this_race = history[history["meeting_key"] == q["meeting_key"]]
    quali = results[results["session_key"] == q["session_key"]]
    actual = this_race.loc[this_race["winner"] == 1, "full_name"]
    return dict(label=q["label"], drivers=this_race, quali=quali,
                actual=actual.iloc[0] if len(actual) else None,
                race_date=this_race["race_date"].min())


def find_meeting(sessions, race_name):
    # the race weekend where the circuit, city or country matches race_name
    text = race_name.lower()
    where = sessions[["circuit_short_name", "location", "country_name"]].astype(str)
    hit = sessions[where.apply(lambda c: c.str.lower().str.contains(text, regex=False)).any(axis=1)]
    if hit.empty:
        raise SystemExit(f"No race matching '{race_name}'. Options: "
                         + ", ".join(sessions["circuit_short_name"].unique()))
    return sessions[sessions["meeting_key"] == hit["meeting_key"].iloc[-1]]


def fetch_live(year, race_name, stage):
    # this weekend's drivers (+ the quali result for post) straight from openf1
    from openf1_client import get_openf1_data              # only needed for live

    sessions = get_openf1_data("sessions", {"year": year})  # always fresh, not the saved file
    if sessions.empty:
        raise SystemExit(f"No sessions found for {year}.")
    weekend = find_meeting(sessions, race_name).sort_values("date_start")
    label = f"{year} {weekend['circuit_short_name'].iloc[0]}"

    # drivers from the latest session of the weekend that has any
    # before the weekend starts there won't be any so just use the latest session anywhere
    drivers = pd.DataFrame()
    for key in weekend["session_key"].iloc[::-1]:
        drivers = get_openf1_data("drivers", {"session_key": key})
        if not drivers.empty:
            break
    if drivers.empty:
        drivers = get_openf1_data("drivers", {"session_key": "latest"})

    quali = None
    if stage == "post":
        q = weekend[weekend["session_name"] == "Qualifying"]
        quali = get_openf1_data("session_result", {"session_key": q["session_key"].iloc[0]}) if len(q) else pd.DataFrame()
        if quali.empty:
            raise SystemExit("Qualifying results aren't out yet - use --stage pre for now.")
        quali["meeting_key"] = q["meeting_key"].iloc[0]
    return dict(label=label, drivers=drivers, quali=quali, actual=None)


def plot_prediction(pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = pred.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top["full_name"], top["win_probability"] * 100, color="#e10600")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Predicted win probability (%)")
    ax.set_title(title)
    ax.set_xlim(0, top["win_probability"].max() * 120)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["pre", "post"], required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--race")
    parser.add_argument("--backtest", help='a past race, e.g. "2025 Monza"')
    args = parser.parse_args()

    history = load_dataset()
    if args.backtest:
        race = load_backtest(args.backtest, history)
        history = history[history["race_date"] < race["race_date"]]   # hide this race and everything after
    elif args.year and args.race:
        race = fetch_live(args.year, args.race, args.stage)
    else:
        parser.error("use --year and --race, or --backtest")

    pred = predict(args.stage, history, race["drivers"], race["quali"])
    features = PRE_FEATURES if args.stage == "pre" else POST_FEATURES
    when = "before qualifying" if args.stage == "pre" else "after qualifying"

    print(f"\n{race['label']} - win probabilities {when} "
          f"(trained on {history['session_key'].nunique()} races)\n")
    print(pred[["full_name", "team_name"] + features + ["win_probability"]].head(10)
          .to_string(float_format="{:.2f}".format, formatters={"win_probability": "{:.1%}".format}))
    print(f"\nPredicted winner: {pred.iloc[0]['full_name']} ({pred.iloc[0]['win_probability']:.0%})")
    if race["actual"]:
        print(f"Actual winner:    {race['actual']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"prediction_{args.stage}_" + race["label"].lower().replace(" ", "_")
    pred[["full_name", "name_acronym", "team_name"] + features + ["win_probability"]].to_csv(
        RESULTS_DIR / f"{stem}.csv", index_label="rank")
    plot_prediction(pred, f"{race['label']} - predicted winner ({when})", RESULTS_DIR / f"{stem}.png")
    print(f"Saved {RESULTS_DIR / stem}.csv and .png")


if __name__ == "__main__":
    main()
