"""
Smart Resource Conservation - backend.
Serves the frontend (static/index.html) and the JSON API that wraps the notebook's Section 8-9 logic.

Run:  python app.py        ->  http://localhost:5000
Uses energy_models.joblib (saved by notebook cell 9C, or by train.py). If missing, it trains from
energydata_complete.csv automatically.
"""
import os, json, math
import numpy as np, pandas as pd, joblib
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "energy_models.joblib")
ANALYTICS_FILE = os.path.join(HERE, "analytics.json")
CSV_FILE = os.path.join(HERE, "energydata_complete.csv")

if not os.path.exists(MODEL_FILE):
    if not os.path.exists(CSV_FILE):
        raise SystemExit("Missing energy_models.joblib. Either run notebook cell 9C and copy the file here, "
                         "or put energydata_complete.csv here so the app can train.")
    import train
    os.chdir(HERE)
    train.main(CSV_FILE)

B = joblib.load(MODEL_FILE)
FEATURES, CLASSES = B["features"], B["classes"]
TARIFF = B["tariff_multiplier"]
ECO, WASTE = B["eco_max_wh"], B["waste_min_wh"]
REG, CLF, TREE = B["regressor"], B["classifier"], B["tree_classifier"]
MEDIAN = pd.Series(B["train_median"])
LIGHTS = {k: {int(h): v for h, v in d.items()} for k, d in B["lights_by_hour"].items()}
VALID = B["valid_ranges"]
APPLIANCES = B["appliances"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def tariff_period(h):
    if 18 <= h < 22: return "Peak"
    if h >= 22 or h < 6: return "Off-peak"
    return "Normal"


HOUR_MULT = np.array([TARIFF[tariff_period(h)] for h in range(24)])


# ---------------------------------------------------------------- notebook logic
def recommend_start(preferred, duration=2, max_shift=6, earliest=6, latest_finish=23):
    cost = lambda s: float(np.mean([HOUR_MULT[s + i] for i in range(duration)]))
    lo = max(earliest, preferred - max_shift)
    hi = min(latest_finish - duration, preferred + max_shift)
    cands = list(range(lo, hi + 1)) or [preferred]
    best = min(cands, key=lambda s: (round(cost(s), 4), abs(s - preferred)))
    return best, cost(preferred), cost(best), cands


def build_row(day, hour, lights, t_out, rh_out, t_in):
    dow = DAYS.index(day)
    tp = tariff_period(hour)
    a, b = 17.27, 237.7
    g = a * t_out / (b + t_out) + math.log(max(rh_out, 1) / 100)
    row = MEDIAN.copy()
    row.update({"hour": hour, "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
                "dayofweek": dow, "is_weekend": int(dow >= 5), "tariff_Off-peak": int(tp == "Off-peak"), "tariff_Peak": int(tp == "Peak"),
                "lights": lights, "T_gap_in_out": t_in - t_out, "T_out": t_out, "RH_out": rh_out,
                "T_out_roll1h": t_out, "RH_out_roll1h": rh_out, "Tdewpoint": b * g / (a - g), "T6": t_out, "RH_6": rh_out})
    return pd.DataFrame([row])[FEATURES]


def advice_for(klass, period):
    if klass == "Wasteful" and period == "Peak":
        return "Heavy usage is expected during the PEAK tariff. Postpone flexible appliances (washer, dishwasher, dryer) to the lighter hours below."
    if klass == "Wasteful" and period == "Normal":
        return "Heavy usage is expected in normal-tariff hours. Avoid running several large appliances together and check for standby loads."
    if klass == "Wasteful":
        return "Heavy usage at night is unusual - check whether an appliance has been left running."
    if klass == "Moderate" and period == "Peak":
        return "Typical usage, but this is the PEAK tariff: avoid adding flexible loads now."
    if klass == "Moderate":
        return "Typical usage. No action needed; flexible loads are fine if the tariff is not peak."
    return "Low usage expected. A good moment for flexible loads if the tariff is not peak."


# ---------------------------------------------------------------- web app
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.json.sort_keys = False   # keep appliance order as defined in the notebook


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "appliances": {k: {"shiftable": v[0], "duration_h": v[1], "kwh": v[2]} for k, v in APPLIANCES.items()},
        "days": DAYS, "classes": CLASSES, "eco_max_wh": ECO, "waste_min_wh": round(WASTE, 1),
        "tariff": TARIFF, "tariff_by_hour": [tariff_period(h) for h in range(24)], "hour_mult": HOUR_MULT.tolist(),
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in VALID.items()},
        "models": {"regressor": B["regressor_name"], "classifier": B["classifier_name"],
                   "mae": round(B["reg_mae"], 1), "rmse": round(B["reg_rmse"], 1)}})


@app.get("/api/analytics")
def analytics():
    if not os.path.exists(ANALYTICS_FILE):
        return jsonify({"error": "analytics.json not found - run `python train.py` once."}), 404
    with open(ANALYTICS_FILE) as f:
        return jsonify(json.load(f))


def num(d, key, default, lo=None, hi=None):
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return v


@app.post("/api/schedule")
def schedule():
    d = request.get_json(force=True, silent=True) or {}
    try:
        name = d.get("appliance", "Washing machine")
        if name not in APPLIANCES:
            raise ValueError("Unknown appliance")
        shiftable, def_dur, def_kwh = APPLIANCES[name]
        if not shiftable:
            return jsonify({"shiftable": False, "appliance": name,
                            "message": f"{name} must run continuously (food safety), so it is NOT rescheduled.",
                            "advice": "Keep it away from heat, keep door seals clean, and use the Usage Risk Check to see which hours the household load is high."})
        duration = int(num(d, "duration_h", def_dur, 1, 6))
        kwh = num(d, "kwh", def_kwh, 0.05, 20)
        pref = int(num(d, "preferred_hour", 18, 0, 23))
        max_shift = int(num(d, "max_shift_h", 6, 0, 12))
        earliest = int(num(d, "earliest_start", 6, 0, 23))
        latest = int(num(d, "latest_finish", 23, 1, 24))
        if latest - earliest < duration:
            raise ValueError("The allowed window is shorter than the cycle - widen earliest start / latest finish.")
        if pref + duration > 24:
            raise ValueError("The cycle must finish on the same day: choose an earlier preferred start.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    best, c_pref, c_best, cands = recommend_start(pref, duration, max_shift, earliest, latest)
    saving = 100 * (c_pref - c_best) / c_pref
    shift = best - pref
    if shift == 0:
        msg = "Your preferred time is already the cheapest allowed option. No change needed."
    elif saving > 0:
        msg = f"Start {abs(shift)} h {'earlier' if shift < 0 else 'later'}: about {saving:.0f}% cheaper under the assumed tariff."
    else:
        msg = f"Your preferred time lies outside the allowed window, so the nearest allowed start is suggested ({saving:.0f}% vs preferred)."
    return jsonify({"shiftable": True, "appliance": name, "duration_h": duration, "kwh": kwh,
                    "preferred_start": pref, "recommended_start": best, "shift_h": shift,
                    "cost_index_preferred": round(kwh * c_pref, 3), "cost_index_recommended": round(kwh * c_best, 3),
                    "saving_pct": round(saving, 1), "message": msg, "candidates": cands,
                    "allowed_window": [earliest, latest]})


@app.post("/api/risk")
def risk():
    d = request.get_json(force=True, silent=True) or {}
    try:
        day = d.get("day", "Monday")
        if day not in DAYS:
            raise ValueError("Unknown day")
        hour = int(num(d, "hour", 18, 0, 23))
        lights = num(d, "lights_wh", 30, 0)
        t_out = num(d, "outdoor_temp", 8.0)
        rh = num(d, "outdoor_rh", 70.0, 0, 100)
        t_in = num(d, "indoor_temp", 20.0)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    warnings = []
    for nm, val in [("outdoor_temp", t_out), ("outdoor_rh", rh), ("indoor_temp", t_in), ("lights_wh", lights)]:
        lo, hi = VALID[nm]
        if not lo <= val <= hi:
            warnings.append(f"{nm.replace('_', ' ')} = {val:g} is outside the range seen in the data ({lo:.1f} to {hi:.1f}); the estimate is unreliable.")

    row = build_row(day, hour, lights, t_out, rh, t_in)
    energy = max(float(REG.predict(row)[0]), 0.0)
    klass = CLASSES[int(CLF.predict(row)[0])]
    probs = TREE.predict_proba(row)[0]
    period = tariff_period(hour)

    dt = "weekend" if DAYS.index(day) >= 5 else "weekday"
    rows = pd.concat([build_row(day, h, LIGHTS[dt].get(h, 0), t_out, rh, t_in) for h in range(24)], ignore_index=True)
    curve = np.clip(REG.predict(rows), 0, None)
    weighted = curve * HOUR_MULT
    best_hours = sorted(range(6, 23), key=lambda h: weighted[h])[:3]

    return jsonify({"day": day, "hour": hour, "tariff_period": period, "tariff_multiplier": TARIFF[period],
                    "energy_wh": round(energy, 1), "avg_watts": round(energy * 6), "mae": round(B["reg_mae"], 1),
                    "class": klass, "classifier": B["classifier_name"],
                    "tree_probs": {c: round(float(p), 3) for c, p in zip(CLASSES, probs)},
                    "advice": advice_for(klass, period), "lightest_hours": best_hours,
                    "curve": [round(float(v), 1) for v in curve], "warnings": warnings})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
