# the live bit - works out which race is next and if quali has happened yet,
# then puts the right prediction in README.md
#   before quali -> pre quali model
#   after quali  -> post quali model
#
# run update_data.py first so it has the latest results:
#   python src/data/update_data.py
#   python src/update_readme.py
# the github job runs both every few hours
# for testing: --now "2025-12-06 18:00" pretends it's a different time

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from predict import fetch_live, plot_prediction, predict  # noqa: E402
from evaluate import RESULTS_DIR, load_dataset  # noqa: E402

README = PROJECT_ROOT / "README.md"
SESSIONS_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"
CHART = RESULTS_DIR / "live_prediction.png"
LOG = RESULTS_DIR / "live_predictions.csv"
START, END = "<!-- LIVE-PREDICTION:START -->", "<!-- LIVE-PREDICTION:END -->"


def next_race(now):
    # first race this season that hasn't finished yet (None if the season's over)
    file = SESSIONS_DIR / f"{now.year}.csv"
    if not file.exists():
        return None, None
    sessions = pd.read_csv(file)
    sessions = sessions[sessions["is_cancelled"].astype(str).str.lower() != "true"]
    sessions["date_end"] = pd.to_datetime(sessions["date_end"], utc=True)
    races = sessions[(sessions["session_name"] == "Race") & (sessions["date_end"] > now)]
    if races.empty:
        return None, None
    race = races.sort_values("date_end").iloc[0]
    quali = sessions[(sessions["meeting_key"] == race["meeting_key"]) & (sessions["session_name"] == "Qualifying")]
    quali_done = len(quali) > 0 and quali["date_end"].iloc[0] < now
    return race, quali_done


def make_prediction(race, quali_done):
    # post quali if the quali results are out, otherwise pre quali
    year, circuit = int(race["year"]), race["circuit_short_name"]
    stage, live = "pre", None
    if quali_done:
        try:
            live, stage = fetch_live(year, circuit, "post"), "post"
        except SystemExit:                       # quali's finished but the results aren't up yet
            pass
    if live is None:
        live = fetch_live(year, circuit, "pre")

    history = load_dataset()
    history = history[history["race_date"] < pd.to_datetime(race["date_start"], utc=True)]
    return stage, live["label"], predict(stage, history, live["drivers"], live["quali"])


def build_section(label, stage, pred):
    when = "after qualifying" if stage == "post" else "before qualifying"
    model = ("post-qualifying model: grid position, gap to pole, qualifying Elo" if stage == "post"
             else "pre-qualifying model: race Elo, recent qualifying form")
    rows = "\n".join(f"| {rank} | {r.full_name} | {r.team_name} | {r.win_probability:.0%} |"
                     for rank, r in pred.head(5).iterrows())
    return (f"## Live prediction: {label}\n\n"
            f"**Predicted winner: {pred.iloc[0]['full_name']}** ({pred.iloc[0]['win_probability']:.0%} chance), "
            f"predicted {when} using the {model}.\n\n"
            f"| # | Driver | Team | Win chance |\n|---|---|---|---|\n{rows}\n\n"
            f"![Live prediction](results/{CHART.name})\n")


def write_readme(section, now):
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    stamp = f"\n_Updated automatically {now:%d %b %Y %H:%M} UTC_\n"
    if START in text and END in text:
        before, rest = text.split(START, 1)
        old, after = rest.split(END, 1)
        if old.split("\n_Updated automatically")[0].strip() == section.strip():
            return False                                   # same as last time, don't touch it
    else:
        before, after = text.rstrip() + "\n\n", "\n"
    README.write_text(f"{before}{START}\n{section}{stamp}{END}{after}", encoding="utf-8")
    return True


def log_prediction(label, stage, pred, now):
    # keep every live prediction so i can score the 2026 season at the end
    row = pd.DataFrame([{"race": label, "stage": stage, "predicted": pred.iloc[0]["full_name"],
                         "win_probability": round(pred.iloc[0]["win_probability"], 4),
                         "made_at": now.strftime("%Y-%m-%d %H:%M")}])
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    if not log.empty:
        log = log[~((log["race"] == label) & (log["stage"] == stage))]     # replace it, don't add duplicates
    pd.concat([log, row], ignore_index=True).to_csv(LOG, index=False)


def main(now):
    race, quali_done = next_race(now)
    if race is None:
        section = "## Live prediction\n\nNo more races this season - predictions start again next season.\n"
        changed = write_readme(section, now)
    else:
        stage, label, pred = make_prediction(race, quali_done)
        section = build_section(label, stage, pred)
        changed = write_readme(section, now)
        if changed:
            RESULTS_DIR.mkdir(exist_ok=True)
            when = "after qualifying" if stage == "post" else "before qualifying"
            plot_prediction(pred, f"{label} - predicted winner ({when})", CHART)
            log_prediction(label, stage, pred, now)
        print(f"{label}: {stage}-qualifying -> {pred.iloc[0]['full_name']} ({pred.iloc[0]['win_probability']:.0%})")
    print("README updated" if changed else "No change - README left as it is")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help='pretend it is this time (for testing) e.g. "2025-12-06 18:00"')
    args = parser.parse_args()
    main(pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC"))
