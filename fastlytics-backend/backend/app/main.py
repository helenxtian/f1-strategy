from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import fastf1.plotting
from fastf1.ergast import Ergast
import os
from functools import lru_cache
import re
from typing import Dict, List

from .strategy import (
    RaceState,
    ReplayResponse,
    ScenarioEvaluationRequest,
    StrategyPredictionRequest,
    build_race_state_timeline,
    evaluate_strategy_rules,
    load_strategy_ml_model,
    evaluate_pit_scenarios,
    predict_best_strategy,
    train_strategy_ml_model,
)

app = FastAPI()

default_origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

cors_origins_env = os.getenv("CORS_ORIGINS", "")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()] or default_origins

fastf1_cache_dir = os.getenv("FASTF1_CACHE_DIR", ".cache/fastf1")
os.makedirs(fastf1_cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(fastf1_cache_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_recent_laps_store: Dict[str, List[float]] = {}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return slug.strip("-")


def _is_missing(value) -> bool:
    return value is None or str(value) in {"NaT", "nan", "None"}


def _time_to_str(value):
    return None if _is_missing(value) else str(value)


def _session_type_from_name(name: str) -> str:
    lowered = name.lower()
    if "practice 1" in lowered:
        return "FP1"
    if "practice 2" in lowered:
        return "FP2"
    if "practice 3" in lowered:
        return "FP3"
    if "sprint shootout" in lowered:
        return "SQ"
    if lowered == "sprint" or "sprint race" in lowered:
        return "Sprint"
    if "qualifying" in lowered:
        return "Q"
    if "race" in lowered:
        return "R"
    return name


def _session_api_to_fastf1(session: str) -> str:
    mapping = {
        "R": "R",
        "Q": "Q",
        "FP1": "FP1",
        "FP2": "FP2",
        "FP3": "FP3",
        "SQ": "SQ",
        "Sprint": "S",
        "S": "S",
    }
    return mapping.get(session, session)


def _resolve_event_row(year: int, identifier: str):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    identifier_lower = identifier.lower()

    for _, row in schedule.iterrows():
        event_name = str(row["EventName"])
        official_name = str(row["OfficialEventName"]) if "OfficialEventName" in row else ""

        if event_name.lower() == identifier_lower:
            return row
        if official_name and official_name.lower() == identifier_lower:
            return row
        if _slugify(event_name) == identifier_lower:
            return row
        if official_name and _slugify(official_name) == identifier_lower:
            return row

    raise HTTPException(status_code=404, detail=f"Event not found for identifier '{identifier}' in {year}")

# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Fastlytics Backend API (Fast-F1 powered)"}


@app.get("/api/replay/{year}/{event_slug}", response_model=ReplayResponse)
def get_replay_timeline(year: int, event_slug: str, session: str = "R", lap_step: int = 1, max_laps: int | None = None):
    try:
        event_row = _resolve_event_row(year, event_slug)
        event_name = str(event_row["EventName"])
        ff1_session = _session_api_to_fastf1(session)

        timeline = build_race_state_timeline(year=year, event=event_name, session=ff1_session, lap_step=max(1, lap_step))
        if max_laps is not None:
            timeline = timeline[:max(0, max_laps)]

        return ReplayResponse(year=year, event=event_name, session=session, ticks=timeline)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy/evaluate")
def evaluate_strategy(state: RaceState):
    try:
        for driver in state.drivers:
            if driver.last_lap_time is not None:
                _recent_laps_store.setdefault(driver.driver, []).append(driver.last_lap_time)
                _recent_laps_store[driver.driver] = _recent_laps_store[driver.driver][-5:]

        evaluation = evaluate_strategy_rules(state, recent_laps_by_driver=_recent_laps_store)
        return evaluation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy/simulate")
def simulate_strategy(request: ScenarioEvaluationRequest):
    try:
        outcomes = evaluate_pit_scenarios(
            state=request.state,
            target_driver=request.target_driver,
            pit_loss_seconds=request.pit_loss_seconds,
        )
        return {
            "target_driver": request.target_driver,
            "lap": request.state.lap,
            "total_laps": request.state.total_laps,
            "scenarios": [outcome.model_dump() for outcome in outcomes],
            "best": outcomes[0].model_dump() if outcomes else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prediction/strategy")
def predict_strategy(request: StrategyPredictionRequest):
    try:
        prediction = predict_best_strategy(
            state=request.state,
            target_driver=request.target_driver,
            pit_loss_seconds=request.pit_loss_seconds,
        )
        return prediction.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prediction/strategy/train")
def train_strategy_prediction_model(years: str = "2023,2024", lap_step: int = 1, max_events_per_year: int = 8):
    try:
        parsed_years = [int(value.strip()) for value in years.split(",") if value.strip()]
        if not parsed_years:
            raise HTTPException(status_code=400, detail="No valid years provided")

        result = train_strategy_ml_model(
            years=parsed_years,
            lap_step=max(1, lap_step),
            max_events_per_year=max(1, max_events_per_year),
        )
        return result
    except HTTPException:
        raise
    except RuntimeError as e:
        message = str(e)
        if "No training data was generated" in message:
            message = f"{message}. Try lap_step=1 and/or include years with completed race data."
        raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prediction/strategy/model")
def get_strategy_prediction_model_info():
    payload = load_strategy_ml_model()
    if payload is None:
        return {"available": False}

    return {
        "available": True,
        "version": payload.get("version"),
        "trained_at": payload.get("trained_at"),
        "years": payload.get("years"),
        "training_samples": payload.get("training_samples"),
        "training_accuracy": payload.get("training_accuracy"),
        "validation_samples": payload.get("validation_samples"),
        "validation_accuracy": payload.get("validation_accuracy"),
        "validation_roc_auc": payload.get("validation_roc_auc"),
        "validation_brier": payload.get("validation_brier"),
        "validation_log_loss": payload.get("validation_log_loss"),
        "validation_ece": payload.get("validation_ece"),
    }

# --- Driver Standings ---
@app.get("/api/standings/drivers")
def get_driver_standings(year: int = 2023):
    try:
        standings = Ergast().get_driver_standings(season=year).content[0]
        return [
            {
                "rank": int(s["position"]),
                "code": s["driverCode"],
                "name": f"{s['givenName']} {s['familyName']}",
                "team": s["constructorNames"][0] if s["constructorNames"] else "",
                "points": float(s["points"]),
                "wins": int(s["wins"]),
                "podiums": 0,
                "points_change": None,
                "teamColor": None
            }
            for _, s in standings.iterrows()
        ]
    except Exception as e:
        return {"error": str(e)}

# --- Team Standings ---
@app.get("/api/standings/teams")
def get_team_standings(year: int = 2023):
    try:
        standings = Ergast().get_constructor_standings(season=year).content[0]
        return [
            {
                "rank": int(s["position"]),
                "team": s["constructorName"],
                "points": float(s["points"]),
                "wins": int(s["wins"]),
                "podiums": 0,
                "points_change": None,
                "teamColor": None,
                "shortName": s["constructorName"][:3].upper() if s["constructorName"] else None
            }
            for _, s in standings.iterrows()
        ]
    except Exception as e:
        return {"error": str(e)}

# --- Race Results ---
@lru_cache(maxsize=8)
def _get_race_results_summary(year: int):
    ergast = Ergast()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    results = []

    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        try:
            race_result = ergast.get_race_results(season=year, round=round_number)
            if not race_result.content:
                continue

            round_df = race_result.content[0]
            if round_df.empty:
                continue

            winner = round_df.iloc[0]
            results.append({
                "year": year,
                "event": event["EventName"],
                "round": round_number,
                "driver": winner.get("driverCode") or winner.get("familyName") or "UNK",
                "team": winner.get("constructorName") or "Unknown",
                "teamColor": None,
                "date": str(event["EventDate"]),
                "location": event["Location"],
            })
        except Exception:
            continue

    return results

@app.get("/api/results/races")
def get_race_results(year: int = 2023):
    try:
        return _get_race_results_summary(year)
    except Exception as e:
        return {"error": str(e)}

# --- Race Schedule ---
@app.get("/api/schedule/{year}")
def get_schedule(year: int):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = []
        for _, event in schedule.iterrows():
            events.append({
                "RoundNumber": int(event["RoundNumber"]),
                "Country": event["Country"],
                "Location": event["Location"],
                "EventName": event["EventName"],
                "EventDate": str(event["EventDate"]),
                "EventFormat": event["EventFormat"],
                "Session1": event["Session1"],
                "Session1Date": str(event["Session1Date"]) if event["Session1Date"] is not None else None,
                "Session2": event["Session2"],
                "Session2Date": str(event["Session2Date"]) if event["Session2Date"] is not None else None,
                "Session3": event["Session3"],
                "Session3Date": str(event["Session3Date"]) if event["Session3Date"] is not None else None,
                "Session4": event["Session4"],
                "Session4Date": str(event["Session4Date"]) if event["Session4Date"] is not None else None,
                "Session5": event["Session5"],
                "Session5Date": str(event["Session5Date"]) if event["Session5Date"] is not None else None,
                "F1ApiSupport": bool(event["F1ApiSupport"])
            })
        return events
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sessions")
def get_available_sessions(year: int, event: str):
    try:
        event_row = _resolve_event_row(year, event)
        sessions = []
        for idx in range(1, 6):
            name_key = f"Session{idx}"
            date_key = f"Session{idx}Date"
            session_name = event_row.get(name_key)
            if _is_missing(session_name):
                continue
            sessions.append({
                "name": str(session_name),
                "type": _session_type_from_name(str(session_name)),
                "startTime": _time_to_str(event_row.get(date_key)),
            })
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schedule/{year}/{event}/sessions")
def get_event_session_schedule(year: int, event: str):
    try:
        event_row = _resolve_event_row(year, event)
        sessions = []
        for idx in range(1, 6):
            name_key = f"Session{idx}"
            date_key = f"Session{idx}Date"
            session_name = event_row.get(name_key)
            session_date = event_row.get(date_key)
            if _is_missing(session_name) or _is_missing(session_date):
                continue
            sessions.append({
                "name": str(session_name),
                "date": str(session_date),
                "localTime": str(session_date).split(" ")[1][:5] if " " in str(session_date) else None,
            })

        return {
            "eventName": str(event_row["EventName"]),
            "location": str(event_row["Location"]),
            "country": str(event_row["Country"]),
            "eventFormat": str(event_row["EventFormat"]),
            "sessions": sessions,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/race/{year}/{event_slug}")
def get_specific_race_results(year: int, event_slug: str, session: str = "R"):
    try:
        event_row = _resolve_event_row(year, event_slug)
        event_name = str(event_row["EventName"])
        ff1_session = _session_api_to_fastf1(session)

        race_session = fastf1.get_session(year, event_name, ff1_session)
        race_session.load()

        if race_session.results is None or race_session.results.empty:
            raise HTTPException(status_code=404, detail="No session results available")

        results_df = race_session.results.copy()

        fastest_lap_map = {}
        for _, row in results_df.iterrows():
            driver_code = row.get("Abbreviation")
            if _is_missing(driver_code):
                continue
            try:
                laps = race_session.laps.pick_drivers(driver_code)
                if laps is None or laps.empty:
                    continue
                fastest = laps.pick_fastest()
                fastest_lap_map[driver_code] = {
                    "time": _time_to_str(fastest.get("LapTime")),
                    "lap": int(fastest.get("LapNumber")) if not _is_missing(fastest.get("LapNumber")) else None,
                }
            except Exception:
                continue

        global_fastest_driver = None
        global_fastest_time = None
        for driver_code, data in fastest_lap_map.items():
            lap_time = data.get("time")
            if lap_time is None:
                continue
            if global_fastest_time is None or lap_time < global_fastest_time:
                global_fastest_time = lap_time
                global_fastest_driver = driver_code

        detailed = []
        for _, row in results_df.iterrows():
            driver_code = row.get("Abbreviation")
            team_color = row.get("TeamColor")
            team_color = f"#{team_color}" if team_color and not str(team_color).startswith("#") else team_color

            best_quali = None
            for col in ["Q3", "Q2", "Q1"]:
                if col in row and not _is_missing(row.get(col)):
                    best_quali = _time_to_str(row.get(col))
                    break

            fastest_driver_data = fastest_lap_map.get(driver_code, {})

            detailed.append({
                "position": int(row.get("Position")) if not _is_missing(row.get("Position")) else None,
                "driverCode": driver_code,
                "fullName": row.get("FullName") or f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip(),
                "team": row.get("TeamName"),
                "points": float(row.get("Points")) if not _is_missing(row.get("Points")) else 0,
                "status": row.get("Status") or "",
                "gridPosition": int(row.get("GridPosition")) if not _is_missing(row.get("GridPosition")) else None,
                "teamColor": team_color,
                "isFastestLap": driver_code == global_fastest_driver,
                "fastestLapTime": best_quali or _time_to_str(row.get("Time")),
                "lapsCompleted": int(row.get("Laps")) if not _is_missing(row.get("Laps")) else None,
                "q1Time": _time_to_str(row.get("Q1")),
                "q2Time": _time_to_str(row.get("Q2")),
                "q3Time": _time_to_str(row.get("Q3")),
                "poleLapTimeValue": best_quali,
                "fastestLapTimeValue": fastest_driver_data.get("time"),
                "time": _time_to_str(row.get("Time")),
                "laps": int(row.get("Laps")) if not _is_missing(row.get("Laps")) else None,
                "driverFastestLapTime": fastest_driver_data.get("time"),
                "driverFastestLapNumber": fastest_driver_data.get("lap"),
            })

        return detailed
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Lap-by-lap Position Data ---
@app.get("/api/lapdata/positions")
def get_lap_positions(year: int = 2023, event: str = "Monaco", session: str = "R"):
    try:
        race = fastf1.get_session(year, event, session)
        race.load()
        laps = race.laps
        drivers = laps["Driver"].unique()
        result = []
        for lap_number in sorted(laps["LapNumber"].unique()):
            lap_data = {"LapNumber": int(lap_number)}
            for drv in drivers:
                lap = laps[(laps["LapNumber"] == lap_number) & (laps["Driver"] == drv)]
                lap_data[drv] = int(lap["Position"].values[0]) if not lap.empty else None
            result.append(lap_data)
        return result
    except Exception as e:
        return {"error": str(e)}

# --- Session Weather Data ---
@app.get("/api/weather")
def get_session_weather(year: int = 2023, event: str = "Monaco", session: str = "R"):
    try:
        race_session = fastf1.get_session(year, event, session)
        race_session.load()
        weather = race_session.weather_data
        if weather is None or weather.empty:
            return []

        result = []
        for _, row in weather.iterrows():
            result.append({
                "time": str(row.get("Time")) if row.get("Time") is not None else None,
                "airTemp": float(row.get("AirTemp")) if row.get("AirTemp") is not None else None,
                "trackTemp": float(row.get("TrackTemp")) if row.get("TrackTemp") is not None else None,
                "humidity": float(row.get("Humidity")) if row.get("Humidity") is not None else None,
                "pressure": float(row.get("Pressure")) if row.get("Pressure") is not None else None,
                "rainfall": bool(row.get("Rainfall")) if row.get("Rainfall") is not None else False,
                "windSpeed": float(row.get("WindSpeed")) if row.get("WindSpeed") is not None else None,
                "windDirection": float(row.get("WindDirection")) if row.get("WindDirection") is not None else None,
            })
        return result
    except Exception as e:
        return {"error": str(e)}

# --- Head-to-head Driver Comparison ---
@app.get("/api/comparison/headtohead")
def head_to_head(year: int = 2023, event: str = "Monaco", session: str = "R", driver1: str = "HAM", driver2: str = "VER"):
    try:
        race = fastf1.get_session(year, event, session)
        race.load()
        laps = race.laps.pick_drivers([driver1, driver2])
        comparison = []
        for lap_number in sorted(laps["LapNumber"].unique()):
            lap1 = laps[(laps["LapNumber"] == lap_number) & (laps["Driver"] == driver1)]
            lap2 = laps[(laps["LapNumber"] == lap_number) & (laps["Driver"] == driver2)]
            comparison.append({
                "LapNumber": int(lap_number),
                driver1: float(lap1["LapTime"].dt.total_seconds().values[0]) if not lap1.empty else None,
                driver2: float(lap2["LapTime"].dt.total_seconds().values[0]) if not lap2.empty else None,
            })
        return comparison
    except Exception as e:
        return {"error": str(e)}

# --- Telemetry Endpoints ---
def get_telemetry(year, event, session, driver, lap, channel):
    race = fastf1.get_session(year, event, session)
    race.load()
    tel = race.lap_tel(lap, driver)
    return [{"Distance": float(d), channel: float(val)} for d, val in zip(tel["Distance"], tel[channel])]

@app.get("/api/telemetry/speed")
def telemetry_speed(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "Speed")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telemetry/gear")
def telemetry_gear(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "Gear")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telemetry/throttle")
def telemetry_throttle(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "Throttle")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telemetry/brake")
def telemetry_brake(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "Brake")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telemetry/rpm")
def telemetry_rpm(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "RPM")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telemetry/drs")
def telemetry_drs(year: int, event: str, session: str, driver: str, lap: int):
    try:
        return get_telemetry(year, event, session, driver, lap, "DRS")
    except Exception as e:
        return {"error": str(e)}
