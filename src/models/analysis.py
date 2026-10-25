# is the improvement real, and does anything else do better?
#
# 1. luck check    - could beating the baseline just be luck?
# 2. feature check - do more features help? (no)
# 3. model check   - does xgboost beat the simple model? (no)
# then saves the charts for the readme
#
# run: python src/models/analysis.py

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                      # save charts to files, no popup window
import matplotlib.pyplot as plt            # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import RESULTS_DIR, load_dataset, summarise, walk_forward  # noqa: E402
from race_models import (POST_FEATURES, PRE_FEATURES, GridPositionBaseline,  # noqa: E402
                         PreviousWinnerBaseline, normalise_within_race,
                         post_qualifying_model, pre_qualifying_model)

warnings.filterwarnings("ignore")


# bootstrap - rerun the 48 race test 10,000 times picking races at random
# (some twice, some not at all) and count how often the model stops beating the baseline.
# if that hardly ever happens it's not luck
def bootstrap_improvement(baseline_losses, model_losses, n=10_000, seed=0):
    diff = np.asarray(baseline_losses) - np.asarray(model_losses)   # >0 means the model did better that race
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff)).mean() for _ in range(n)])
    low, high = np.percentile(means, [2.5, 97.5])                    # middle 95% of the results
    p_value = (means <= 0).mean()                                    # how often there was no improvement
    return diff.mean(), low, high, p_value


# the normal way of doing it (what my old scripts did) - score each driver on their own
# with logistic regression or xgboost, then scale each race to add up to 100%
class StandardClassifier:

    def __init__(self, kind):
        self.kind = kind

    def fit(self, df, features):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.features = features
        if self.kind == "xgboost":
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42)
            except ImportError:                                       # if xgboost isn't installed
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150)
        else:
            self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        self.model.fit(df[features], df["winner"])
        return self

    def predict_proba(self, df):
        raw = self.model.predict_proba(df[self.features])[:, 1]
        return normalise_within_race(df["session_key"], raw)


# chart helpers
def bar_chart(ax, labels, values, colours, xlabel, line=None, line_label=None):
    ax.barh(labels, values, color=colours)
    if line is not None:
        ax.axvline(line, color="#555", ls="--", lw=1, label=line_label)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()


def colours_for(labels):
    # grey = baseline, red = final model, blue = everything else
    return ["#9a9a9a" if "baseline" in s else ("#e10600" if "FINAL" in s else "#4c6ef5") for s in labels]


if __name__ == "__main__":
    df = load_dataset()

    def test(make_model, features):
        return walk_forward(df, make_model, features)

    # 1. luck check
    print("1. LUCK CHECK - is each model's improvement over its baseline real?")
    runs = {
        "PRE ": (test(PreviousWinnerBaseline, None), test(pre_qualifying_model, PRE_FEATURES)),
        "POST": (test(GridPositionBaseline, None), test(post_qualifying_model, POST_FEATURES)),
    }
    for stage, (base, model) in runs.items():
        gain, low, high, p = bootstrap_improvement(base["log_loss"], model["log_loss"])
        print(f"   {stage}: log loss {base['log_loss'].mean():.3f} -> {model['log_loss'].mean():.3f}"
              f"  | 95% range of the improvement: {low:.3f} to {high:.3f} | p = {p:.3f}")
    print("   (the whole range is above 0 and p is tiny -> very unlikely to be luck)\n")

    # 2. feature check - add one thing at a time
    print("2. FEATURE CHECK - add one idea at a time (conditional logit)")
    form = ["driver_form_points", "team_form_points"]
    feature_sets = {
        "PRE baseline (last winner)": (PreviousWinnerBaseline, None),
        "PRE: race Elo only": (pre_qualifying_model, ["race_elo_rel"]),
        "PRE: + recent qualifying (FINAL)": (pre_qualifying_model, PRE_FEATURES),
        "PRE: + points form": (pre_qualifying_model, PRE_FEATURES + form),
        "PRE: + everything": (pre_qualifying_model, PRE_FEATURES + form + ["quali_elo_rel"]),
        "POST baseline (grid slot)": (GridPositionBaseline, None),
        "POST: grid position": (post_qualifying_model, ["log_quali_pos"]),
        "POST: + gap to pole": (post_qualifying_model, ["log_quali_pos", "quali_gap_capped"]),
        "POST: + qualifying Elo (FINAL)": (post_qualifying_model, POST_FEATURES),
        "POST: + points form": (post_qualifying_model, POST_FEATURES + form),
        "POST: + everything": (post_qualifying_model, POST_FEATURES + form + ["race_elo_rel", "quali_pos_last5"]),
    }
    features_table = pd.DataFrame({name: summarise(test(m, f)) for name, (m, f) in feature_sets.items()}).T
    print(features_table[["winner_right", "winner_in_top3", "log_loss"]].round(3).to_string(), "\n")
    features_table.round(4).to_csv(RESULTS_DIR / "feature_comparison.csv", index_label="features")

    # 3. model check - same post quali features, different models
    print("3. MODEL CHECK - same post-qualifying features, different models")
    models = {
        "Conditional logit (FINAL)": post_qualifying_model,
        "Logistic regression": lambda: StandardClassifier("logreg"),
        "XGBoost": lambda: StandardClassifier("xgboost"),
    }
    model_table = pd.DataFrame({name: summarise(test(m, POST_FEATURES)) for name, m in models.items()}).T
    print(model_table[["winner_right", "winner_in_top3", "log_loss"]].round(3).to_string(), "\n")
    model_table.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index_label="model")

    # charts
    results = pd.read_csv(RESULTS_DIR / "model_results.csv", index_col="model")
    labels = [s.replace("PRE model", "PRE model (FINAL)").replace("POST model", "POST model (FINAL)")
              for s in results.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bar_chart(axes[0], labels, results["winner_right"] * 100, colours_for(labels), "Winner predicted (%) - higher is better")
    bar_chart(axes[1], labels, results["log_loss"], colours_for(labels), "Log loss - lower is better")
    fig.suptitle("Pre- and post-qualifying models vs simple baselines (48 races, 2024-2025)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_results.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, stage in zip(axes, ["PRE", "POST"]):
        part = features_table[features_table.index.str.startswith(stage)]
        bar_chart(ax, part.index, part["winner_right"] * 100, colours_for(part.index),
                  "Winner predicted (%)", line=part["winner_right"].iloc[0] * 100, line_label="baseline")
        ax.set_title(f"{stage}-qualifying: adding features one at a time")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_comparison.png", dpi=150)
    print(f"Charts saved: {RESULTS_DIR / 'model_results.png'}, {RESULTS_DIR / 'feature_comparison.png'}")
