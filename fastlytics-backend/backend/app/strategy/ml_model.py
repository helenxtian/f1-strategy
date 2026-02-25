from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import fastf1

from .replay import build_race_state_timeline
from .schemas import RaceState

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception:
    ColumnTransformer = None
    accuracy_score = None
    brier_score_loss = None
    log_loss = None
    roc_auc_score = None
    train_test_split = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    LogisticRegression = None


MODEL_VERSION = "strategy-pit-now-v2"
MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "strategy_pit_now_model.joblib"


def _driver_features(state: RaceState, driver) -> dict:
    laps_remaining = max(state.total_laps - state.lap, 0)
    return {
        "lap": state.lap,
        "total_laps": state.total_laps,
        "laps_remaining": laps_remaining,
        "laps_remaining_ratio": laps_remaining / max(1, state.total_laps),
        "position": float(driver.position if driver.position is not None else 20),
        "tire_age": float(driver.tire_age),
        "gap_ahead": float(driver.gap_ahead if driver.gap_ahead is not None else 5.0),
        "last_lap_time": float(driver.last_lap_time if driver.last_lap_time is not None else 95.0),
        "compound": (driver.compound or "UNKNOWN").upper(),
        "stint_number": float(driver.stint_number if driver.stint_number is not None else 1),
        "is_pit_lap": float(1 if driver.is_pit_lap else 0),
    }


def _build_training_dataframe(
    years: list[int],
    lap_step: int = 1,
    max_events_per_year: int = 8,
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict] = []
    labels: list[int] = []

    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event_row in schedule.head(max_events_per_year).iterrows():
            event_name = str(event_row["EventName"])
            try:
                timeline = build_race_state_timeline(year=year, event=event_name, session="R", lap_step=lap_step)
            except Exception:
                continue

            by_lap_and_driver = {
                tick.lap: {driver.driver: driver for driver in tick.drivers}
                for tick in timeline
            }

            for state in timeline:
                for driver in state.drivers:
                    lap_now = state.lap
                    if lap_now > state.total_laps - 4:
                        continue

                    next_1_drivers = by_lap_and_driver.get(lap_now + 1)
                    next_2_drivers = by_lap_and_driver.get(lap_now + 2)
                    next_4_drivers = by_lap_and_driver.get(lap_now + 4)
                    if next_1_drivers is None or next_2_drivers is None or next_4_drivers is None:
                        continue

                    next_1_driver = next_1_drivers.get(driver.driver)
                    next_2_driver = next_2_drivers.get(driver.driver)
                    next_4_driver = next_4_drivers.get(driver.driver)
                    if next_1_driver is None or next_2_driver is None or next_4_driver is None:
                        continue

                    pit_soon = bool(next_1_driver.is_pit_lap or next_2_driver.is_pit_lap)

                    pos_now = driver.position if driver.position is not None else 20
                    pos_future = next_4_driver.position if next_4_driver.position is not None else 20
                    improved_position = pos_future <= pos_now

                    now_lap_time = driver.last_lap_time if driver.last_lap_time is not None else 95.0
                    future_lap_time = next_4_driver.last_lap_time if next_4_driver.last_lap_time is not None else now_lap_time
                    improved_pace = future_lap_time < (now_lap_time - 0.3)

                    success_after_pit = pit_soon and (improved_position or improved_pace)

                    rows.append(_driver_features(state, driver))
                    labels.append(1 if success_after_pit else 0)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype="int64")

    return pd.DataFrame(rows), pd.Series(labels, dtype="int64")


def _expected_calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    if data.empty:
        return 0.0

    data["bin"] = pd.qcut(data["p"], q=min(bins, data["p"].nunique()), labels=False, duplicates="drop")
    ece = 0.0
    total = len(data)

    for _, group in data.groupby("bin"):
        if group.empty:
            continue
        avg_conf = float(group["p"].mean())
        avg_acc = float(group["y"].mean())
        weight = len(group) / total
        ece += weight * abs(avg_conf - avg_acc)

    return float(ece)


def train_strategy_ml_model(
    years: list[int],
    lap_step: int = 1,
    max_events_per_year: int = 8,
    model_path: Optional[str] = None,
) -> dict:
    if Pipeline is None or LogisticRegression is None or train_test_split is None:
        raise RuntimeError("scikit-learn is not installed in this environment")

    X, y = _build_training_dataframe(
        years=years,
        lap_step=lap_step,
        max_events_per_year=max_events_per_year,
    )

    if X.empty or y.empty:
        raise RuntimeError("No training data was generated from historical laps")
    if y.nunique() < 2:
        raise RuntimeError("Training labels have only one class; cannot train binary classifier")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = [
        "lap",
        "total_laps",
        "laps_remaining",
        "laps_remaining_ratio",
        "position",
        "tire_age",
        "gap_ahead",
        "last_lap_time",
        "stint_number",
        "is_pit_lap",
    ]
    categorical_features = ["compound"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    training_accuracy = float(accuracy_score(y_train, train_pred))
    validation_accuracy = float(accuracy_score(y_val, val_pred))
    validation_roc_auc = float(roc_auc_score(y_val, val_prob))
    validation_brier = float(brier_score_loss(y_val, val_prob))
    validation_log_loss = float(log_loss(y_val, val_prob, labels=[0, 1]))
    validation_ece = _expected_calibration_error(y_val, pd.Series(val_prob), bins=10)

    payload = {
        "model": model,
        "version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "years": years,
        "lap_step": lap_step,
        "max_events_per_year": max_events_per_year,
        "training_samples": int(len(X)),
        "training_accuracy": round(training_accuracy, 4),
        "validation_samples": int(len(X_val)),
        "validation_accuracy": round(validation_accuracy, 4),
        "validation_roc_auc": round(validation_roc_auc, 4),
        "validation_brier": round(validation_brier, 4),
        "validation_log_loss": round(validation_log_loss, 4),
        "validation_ece": round(validation_ece, 4),
    }

    output_path = Path(model_path) if model_path else MODEL_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    load_strategy_ml_model.cache_clear()

    return {
        "model_path": str(output_path),
        "version": payload["version"],
        "trained_at": payload["trained_at"],
        "years": years,
        "training_samples": payload["training_samples"],
        "training_accuracy": payload["training_accuracy"],
        "validation_samples": payload["validation_samples"],
        "validation_accuracy": payload["validation_accuracy"],
        "validation_roc_auc": payload["validation_roc_auc"],
        "validation_brier": payload["validation_brier"],
        "validation_log_loss": payload["validation_log_loss"],
        "validation_ece": payload["validation_ece"],
    }


@lru_cache(maxsize=1)
def load_strategy_ml_model(model_path: Optional[str] = None) -> Optional[dict]:
    path = Path(model_path) if model_path else MODEL_FILE
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            return None
        return payload
    except Exception:
        return None


def predict_pit_now_probability(state: RaceState, target_driver: str, model_payload: Optional[dict] = None) -> Optional[float]:
    payload = model_payload or load_strategy_ml_model()
    if payload is None:
        return None

    model = payload.get("model")
    if model is None:
        return None

    driver = next((d for d in state.drivers if d.driver == target_driver), None)
    if driver is None:
        return None

    row = pd.DataFrame([_driver_features(state, driver)])
    try:
        if not hasattr(model, "predict_proba"):
            return None
        probabilities = model.predict_proba(row)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            class_index = classes.index(1)
            return float(probabilities[class_index])
        return None
    except Exception:
        return None
