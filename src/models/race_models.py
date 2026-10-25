# the models - each one gives every driver in a race a win chance, adding up to 100%
#
# baselines (simple rules to beat):
#   GridPositionBaseline   - post quali, "pole usually wins"
#   PreviousWinnerBaseline - pre quali, "last race's winner wins again"
# actual model:
#   ConditionalLogit - same model for both, just different features
#
# they all have fit(train_df, features) and predict_proba(race_df)
# so i can swap them around easily when testing

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize          # finds the best weights
from scipy.special import logsumexp          # stops the softmax maths overflowing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRE_FEATURES = ["race_elo_rel", "quali_pos_last5"]                         # known before quali
POST_FEATURES = ["log_quali_pos", "quali_gap_capped", "quali_elo_rel"]     # known after quali
ALPHA = 3.0                                                                # penalty to keep the weights small


# turns any scores into probabilities that add up to 1 in each race
# e.g. race 7 scores [3, 1] -> [0.75, 0.25]
def normalise_within_race(race_ids, scores):
    scores = pd.Series(np.asarray(scores, dtype=float))
    race_totals = scores.groupby(np.asarray(race_ids)).transform("sum")
    return (scores / race_totals).to_numpy()


class GridPositionBaseline:
    # post quali baseline - how often has each grid slot won before?
    # P1 won x% of the time, P2 y% etc, and each driver gets the rate for where they start

    def fit(self, df, features=None):
        wins = df.groupby("quali_pos")["winner"].agg(["sum", "count"])   # wins and starts per grid slot
        self.rate = (wins["sum"] + 0.1) / (wins["count"] + 1)           # smoothed so it's never exactly 0
        self.default = 0.1 / 25                                          # grid slot never seen before
        return self

    def predict_proba(self, df):
        p = df["quali_pos"].map(self.rate).fillna(self.default)
        return normalise_within_race(df["session_key"], p)


class PreviousWinnerBaseline:
    # pre quali baseline - last race's winner wins again
    # works out how often that actually happens (about 25%) and gives the
    # last winner that chance, everyone else splits the rest

    def fit(self, df, features=None):
        winners = df[df["winner"] == 1].sort_values("race_date")
        repeat = winners["name_acronym"] == winners["name_acronym"].shift(1)
        self.repeat_rate = repeat.iloc[1:].mean() if len(winners) > 1 else 0.25
        self.last_winner = winners["name_acronym"].iloc[-1]             # most recent winner in the training data
        return self

    def predict_proba(self, df):
        is_last_winner = df["name_acronym"] == self.last_winner
        others = len(df) - is_last_winner.sum()
        p = np.where(is_last_winner, self.repeat_rate, (1 - self.repeat_rate) / max(others, 1))
        return normalise_within_race(df["session_key"], p)


class ConditionalLogit:
    # the main model - compares the drivers within each race
    #  1. each driver gets a score = w1*feature1 + w2*feature2 ...
    #  2. softmax over the drivers in that race turns scores into chances:
    #       P(i wins) = exp(score_i) / sum of exp(score) for everyone in the race
    #     so it always adds to 100% and depends on who you're racing against
    #  3. training picks the weights that make the actual winners most likely,
    #     with ALPHA as a penalty so it doesn't just memorise ~70 races

    def __init__(self, alpha=ALPHA):
        self.alpha = alpha

    def fit(self, df, features):
        self.features = features
        # put all features on the same scale (mean 0, spread 1) so the weights are comparable
        self.prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        X = self.prep.fit_transform(df[features])
        y = df["winner"].to_numpy()
        races = df["session_key"].to_numpy()
        groups = [np.where(races == r)[0] for r in np.unique(races)]   # row numbers for each race

        def loss(w):
            # how badly do these weights explain who actually won (lower = better)
            scores = X @ w
            total, gradient = 0.0, np.zeros_like(w)
            for rows in groups:                                  # one race at a time
                s = scores[rows]
                log_total = logsumexp(s)
                p = np.exp(s - log_total)                        # softmax = win chances
                total += log_total - s[y[rows] == 1].sum()       # -log P(actual winner)
                gradient += X[rows].T @ (p - y[rows])            # which way to move the weights
            return total + self.alpha * w @ w, gradient + 2 * self.alpha * w

        self.w = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict_proba(self, df):
        scores = pd.Series(self.prep.transform(df[self.features]) @ self.w)
        scores = scores - scores.groupby(df["session_key"].to_numpy()).transform("max")   # stops huge numbers
        return normalise_within_race(df["session_key"], np.exp(scores))

    def weights(self):
        # what it learned - one weight per feature
        return pd.Series(self.w, index=self.features).round(3)


def pre_qualifying_model():
    return ConditionalLogit(alpha=ALPHA)


def post_qualifying_model():
    return ConditionalLogit(alpha=ALPHA)


if __name__ == "__main__":
    dataset_file = Path(__file__).resolve().parents[2] / "data" / "processed" / "features" / "model_dataset.csv"
    df = pd.read_csv(dataset_file)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)

    print("--- normalise_within_race ---")
    print(normalise_within_race([7, 7, 9, 9], [3, 1, 2, 2]), "\n")

    # predict abu dhabi 2025 using only the races before it
    race = df[df["race_label"] == "2025 Yas Marina Circuit"].copy()
    train = df[df["race_date"] < race["race_date"].min()]          # only the past

    pre = pre_qualifying_model().fit(train, PRE_FEATURES)
    post = post_qualifying_model().fit(train, POST_FEATURES)
    print("Learned weights - PRE-qualifying model:")
    print(pre.weights().to_string(), "\n")
    print("Learned weights - POST-qualifying model:")
    print(post.weights().to_string(), "\n")

    race["pre_qualifying"] = pre.predict_proba(race)
    race["post_qualifying"] = post.predict_proba(race)
    race["post_baseline"] = GridPositionBaseline().fit(train).predict_proba(race)
    print("--- 2025 Abu Dhabi: win probabilities (trained on the races before it) ---")
    print(race.sort_values("post_qualifying", ascending=False)
          [["name_acronym", "quali_pos", "pre_qualifying", "post_qualifying", "post_baseline", "winner"]]
          .head(6).round(3).to_string(index=False))
    print("\nEach column adds up to:", race[["pre_qualifying", "post_qualifying"]].sum().round(6).tolist())
