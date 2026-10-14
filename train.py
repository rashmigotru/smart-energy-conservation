"""
Rebuilds the notebook pipeline (Sections 2-9) and saves:
  energy_models.joblib  - same bundle as notebook cell 9C (so the notebook's own file works too)
  analytics.json        - numbers for the dashboard (usage patterns, waste, model scores, clusters)

Usage:  python train.py [path/to/energydata_complete.csv]
"""
import sys, json, math
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             f1_score, accuracy_score, roc_auc_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
ECO_MAX_WH = 60
TARIFF_MULTIPLIER = {"Off-peak": 0.75, "Normal": 1.00, "Peak": 1.50}
TEST_FRACTION = 0.20
CLASSES = ["Eco", "Moderate", "Wasteful"]
APPLIANCES = {"Washing machine": (True, 2, 1.0), "Dishwasher": (True, 2, 1.2),
              "Clothes dryer": (True, 1, 2.0), "Refrigerator": (False, 24, None)}

FEATURES = ["hour", "hour_sin", "hour_cos", "dayofweek", "is_weekend", "tariff_Off-peak", "tariff_Peak", "lights",
            "T_gap_in_out", "T_out", "RH_out", "T_out_roll1h", "RH_out_roll1h", "Press_mm_hg", "Windspeed", "Visibility",
            "Tdewpoint", "RH_indoor_mean", "T6", "RH_6"]


def tariff_period(h):
    if 18 <= h < 22: return "Peak"
    if h >= 22 or h < 6: return "Off-peak"
    return "Normal"


def prepare(csv):
    raw = pd.read_csv(csv)
    q1, q3 = raw["Appliances"].quantile([0.25, 0.75])
    fence = float(q3 + 1.5 * (q3 - q1))
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True).drop(columns=["rv1", "rv2"])
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["tariff_period"] = df["hour"].map(tariff_period)
    df["tariff_Off-peak"] = (df["tariff_period"] == "Off-peak").astype(int)
    df["tariff_Peak"] = (df["tariff_period"] == "Peak").astype(int)
    indoor_T = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
    indoor_RH = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
    df["T_indoor_mean"] = df[indoor_T].mean(axis=1)
    df["RH_indoor_mean"] = df[indoor_RH].mean(axis=1)
    df["T_gap_in_out"] = df["T_indoor_mean"] - df["T_out"]
    for c in ["T_out", "RH_out"]:
        df[f"{c}_roll1h"] = df[c].rolling(6, min_periods=1).mean()
    df["energy_class"] = np.where(df["Appliances"] <= ECO_MAX_WH, "Eco",
                                  np.where(df["Appliances"] <= fence, "Moderate", "Wasteful"))
    df["label"] = df["energy_class"].map({c: i for i, c in enumerate(CLASSES)})
    df["day_type"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")
    return df, fence


def main(csv):
    df, fence = prepare(csv)
    split = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:split], df.iloc[split:]
    Xtr, Xte = train[FEATURES], test[FEATURES]
    ytr, yte = train["Appliances"], test["Appliances"]
    ctr, cte = train["label"], test["label"]

    # ---------------- regression (Random Forest was best in the notebook) ----------------
    base = DummyRegressor().fit(Xtr, ytr)
    reg = RandomForestRegressor(300, min_samples_leaf=10, max_features=0.33, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    p = reg.predict(Xte)
    reg_scores = {"R2": r2_score(yte, p), "RMSE": float(np.sqrt(mean_squared_error(yte, p))), "MAE": mean_absolute_error(yte, p)}
    pb = base.predict(Xte)
    base_scores = {"R2": r2_score(yte, pb), "RMSE": float(np.sqrt(mean_squared_error(yte, pb))), "MAE": mean_absolute_error(yte, pb)}

    # ---------------- classification ----------------
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE).fit(Xtr, ctr)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=60,
                                         early_stopping=False, class_weight="balanced", random_state=RANDOM_STATE).fit(Xtr, ctr)
    cls_scores = {}
    for nm, m in [("Gradient Boosting", clf), ("Decision Tree", tree)]:
        pr = m.predict(Xte)
        pp = m.predict_proba(Xte)
        cls_scores[nm] = {"accuracy": accuracy_score(cte, pr), "macro_f1": f1_score(cte, pr, average="macro"),
                          "wasteful_recall": float(((pr == 2) & (cte == 2)).sum() / max((cte == 2).sum(), 1)),
                          "macro_auc": float(np.mean([roc_auc_score((cte == k).astype(int), pp[:, k]) for k in range(3)]))}

    # ---------------- helpers for the tools ----------------
    train_median = Xtr.median()
    lights_by_hour = {dt: df[df["day_type"] == dt].groupby("hour")["lights"].mean().round(0).to_dict() for dt in ["weekday", "weekend"]}
    VALID = {"outdoor_temp": (df["T_out"].min(), df["T_out"].max()), "outdoor_rh": (df["RH_out"].min(), df["RH_out"].max()),
             "indoor_temp": (df["T_indoor_mean"].min(), df["T_indoor_mean"].max()), "lights_wh": (0, df["lights"].max())}

    bundle = {"features": FEATURES, "classes": CLASSES, "eco_max_wh": ECO_MAX_WH, "waste_min_wh": fence,
              "tariff_multiplier": TARIFF_MULTIPLIER, "regressor": reg, "regressor_name": "Random Forest",
              "classifier": clf, "classifier_name": "Gradient Boosting", "tree_classifier": tree,
              "train_median": train_median.to_dict(), "lights_by_hour": lights_by_hour,
              "valid_ranges": {k: tuple(map(float, v)) for k, v in VALID.items()},
              "reg_mae": float(reg_scores["MAE"]), "reg_rmse": reg_scores["RMSE"], "appliances": APPLIANCES}
    joblib.dump(bundle, "energy_models.joblib")

    # ---------------- analytics for the dashboard ----------------
    hour_mult = {h: TARIFF_MULTIPLIER[tariff_period(h)] for h in range(24)}
    hourly = df.groupby("hour")["Appliances"].mean()
    waste_by_hour = df.assign(w=(df["energy_class"] == "Wasteful")).groupby("hour")["w"].mean() * 100
    df["excess"] = (df["Appliances"] - fence).clip(lower=0)
    total_e = df["Appliances"].sum()
    by_period = df.groupby("tariff_period")["excess"].sum() / df["excess"].sum() * 100
    dow = df.groupby("dayofweek")["Appliances"].mean()

    # clustering of full days (K-Means, k chosen by silhouette like the notebook)
    full = df.groupby(df["date"].dt.date).size()
    full = full[full == 144].index
    daily = df.groupby([df["date"].dt.date, "hour"])["Appliances"].mean().unstack().loc[full].dropna()
    Xd = daily.values
    sil = {k: silhouette_score(Xd, KMeans(k, n_init=20, random_state=RANDOM_STATE).fit_predict(Xd)) for k in range(2, 6)}
    bk = max(sil, key=sil.get)
    km = KMeans(bk, n_init=20, random_state=RANDOM_STATE).fit(Xd)
    peak_hours = [h for h in range(24) if tariff_period(h) == "Peak"]
    kwh = Xd.sum(axis=1) * 6 / 1000
    peak_share = Xd[:, peak_hours].sum(axis=1) / Xd.sum(axis=1)
    order = np.argsort([kwh[km.labels_ == c].mean() for c in range(bk)])
    clusters = []
    for rank, c in enumerate(order):
        m = km.labels_ == c
        name = "Low-usage day" if rank == 0 else ("High-usage day" if rank == bk - 1 else "Typical day")
        clusters.append({"name": name, "days": int(m.sum()), "avg_kwh": round(float(kwh[m].mean()), 1),
                         "peak_share": round(float(peak_share[m].mean()) * 100, 1),
                         "busiest_hour": int(Xd[m].mean(axis=0).argmax()),
                         "curve": [round(float(v), 1) for v in Xd[m].mean(axis=0)]})

    ana = {
        "dataset": {"rows": int(len(df)), "from": str(df["date"].min().date()), "to": str(df["date"].max().date())},
        "thresholds": {"eco_max": ECO_MAX_WH, "waste_min": round(fence, 1)},
        "hourly_mean": [round(float(hourly[h]), 1) for h in range(24)],
        "waste_by_hour_pct": [round(float(waste_by_hour[h]), 1) for h in range(24)],
        "dow_mean": [round(float(dow[i]), 1) for i in range(7)],
        "tariff_mult": [hour_mult[h] for h in range(24)],
        "tariff_period": [tariff_period(h) for h in range(24)],
        "class_share": {c: round(float((df["energy_class"] == c).mean()) * 100, 1) for c in CLASSES},
        "waste": {"time_share": round(float((df["energy_class"] == "Wasteful").mean()) * 100, 1),
                  "energy_share": round(float(df.loc[df["energy_class"] == "Wasteful", "Appliances"].sum() / total_e) * 100, 1),
                  "excess_share": round(float(df["excess"].sum() / total_e) * 100, 1),
                  "excess_by_period": {k: round(float(v), 1) for k, v in by_period.items()}},
        "models": {"regression": {"name": "Random Forest", **{k: round(v, 3) for k, v in reg_scores.items()},
                                  "baseline": {k: round(v, 3) for k, v in base_scores.items()}},
                   "classification": {k: {a: round(b, 3) for a, b in v.items()} for k, v in cls_scores.items()}},
        "clusters": {"k": bk, "silhouette": round(sil[bk], 2), "groups": clusters},
    }
    with open("analytics.json", "w") as f:
        json.dump(ana, f)
    print("Saved energy_models.joblib and analytics.json")
    print("Regression:", {k: round(v, 3) for k, v in reg_scores.items()})
    print("Classifier:", cls_scores)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "energydata_complete.csv")
