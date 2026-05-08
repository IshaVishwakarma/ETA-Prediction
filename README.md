# 🚕 NYC Taxi ETA Prediction System

> Predicting taxi trip duration in New York City using XGBoost — trained on 7.5M+ real trips.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal) ![Dataset](https://img.shields.io/badge/Data-NYC%20Yellow%20Taxi%202019-lightgrey)

---

## 📌 Overview

This project builds a machine learning pipeline to predict how long a NYC yellow taxi trip will take (in seconds), given pickup/dropoff zone, distance, and time-of-day features. The trained model is served via a FastAPI REST endpoint.

**Dataset**: [NYC Yellow Taxi Trip Data 2019–2020](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019) — 1.81 GB, ~7.5M trips (January 2019 used for training).

---

## 📊 Results

| Model | MAE (seconds) | RMSE (seconds) |
|---|---|---|
| Random Forest (20 trees, depth 10) | 183.4 | 282.9 |
| **XGBoost (50 trees, depth 6)** | **181.2** | **276.3** |

> MAE of ~181 seconds = roughly **3 minutes average error** on a median trip of ~10 minutes.

---

## 🧠 Features Used

| Feature | Description |
|---|---|
| `trip_distance` | Distance in miles |
| `hour` | Hour of pickup (0–23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_rush_hour` | 1 if 8–9 AM or 5–7 PM |
| `pickup_hour_traffic` | Zone-level trip count by hour (busyness proxy) |

---

## 🗂️ Project Structure

```
eta-prediction-system/
├── models/
│   └── eta_model.pkl        # Trained XGBoost model
├── src/                     # Extensible source folder
├── main.py                  # FastAPI app with /predict endpoint
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API server

```bash
uvicorn main:app --reload
```

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 2.5,
    "hour": 17,
    "day_of_week": 1,
    "is_weekend": 0,
    "is_rush_hour": 1,
    "pickup_hour_traffic": 45000
  }'
```

---

## 🔄 Data Pipeline

1. Load raw CSV from Kaggle (`yellow_tripdata_2019-01.csv`)
2. Parse pickup/dropoff datetimes → compute `trip_duration` in seconds
3. Filter: keep trips between 60s and 7200s (1 min – 2 hrs)
4. Engineer features: `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`, `pickup_hour_traffic`
5. Drop leakage columns: `fare_amount`, `total_amount`, `tip_amount`
6. Train/test split: 80/20
7. Train XGBoost → evaluate MAE & RMSE → serialize with `joblib`

---

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **Environment**: Google Colab
- **Modelling**: XGBoost, scikit-learn
- **Serving**: FastAPI, Uvicorn
- **Serialization**: joblib

---

## 🚀 Future Improvements

- [ ] Add more months of data for seasonal patterns
- [ ] Include weather data as a feature
- [ ] Add PULocationID / DOLocationID as encoded categoricals
- [ ] Deploy to a cloud endpoint (Render, Railway, or AWS Lambda)
- [ ] Build a simple front-end UI for trip estimation

---

## 📄 License

MIT
