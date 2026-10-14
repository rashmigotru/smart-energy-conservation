# Smart Resource Conservation – web app

Frontend (`static/index.html`) + Flask backend (`app.py`) wrapping your notebook's Sections 8–9.

## Run it
1. `pip install -r requirements.txt`
2. Put `energydata_complete.csv` in this folder.
3. `python app.py`  → open http://localhost:5000

First run trains the models from the CSV (about a minute) and writes `energy_models.joblib` + `analytics.json`.
Later runs start instantly.

### Using the notebook's own model file instead
Run notebook cell 9C, copy `energy_models.joblib` into this folder, then run `python train.py` once only if you
also want the dashboard (it needs `analytics.json`), or keep the CSV here and delete the joblib to let the app rebuild both.
Use the same scikit-learn version as the notebook when loading its joblib.

## What's inside
| Tab | Backend endpoint | Notebook section |
|---|---|---|
| Dashboard | `GET /api/analytics` | 4, 5, 6, 7, 8B |
| Appliance scheduler | `POST /api/schedule` | 8A, 9A |
| Usage risk check | `POST /api/risk` | 9B |
| Assumptions | `GET /api/meta` | 1, 11 |

Tariff, thresholds and appliance defaults are the notebook's assumptions (edit them in `train.py`).
