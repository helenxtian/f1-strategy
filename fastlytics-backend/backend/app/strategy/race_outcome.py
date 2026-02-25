from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List
import random

import numpy as np

from .engine import EXPECTED_STINT_LENGTH
from .ml_model import load_strategy_ml_model, predict_pit_now_probability
from .schemas import (
    DriverRaceOutcomeForecast,
    FinishPositionProbability,
    RaceOutcomeForecastResponse,
    RaceState,
)


def _pace_projection(base_time: float, tire_age: int, compound: str) -> float:
    compound = (compound or "MEDIUM").upper()
    compound_delta = {
        "SOFT": -0.3,
        "MEDIUM": 0.0,
        "HARD": 0.4,
        "INTERMEDIATE": 1.5,
        "WET": 3.0,
    }.get(compound, 0.0)
    degradation_per_lap = {
        "SOFT": 0.09,
        "MEDIUM": 0.06,
        "HARD": 0.04,
        "INTERMEDIATE": 0.08,
        "WET": 0.12,
    }.get(compound, 0.06)

    return base_time + compound_delta + (degradation_per_lap * tire_age)


def _next_compound(current_compound: str) -> str:
    current = (current_compound or "MEDIUM").upper()
    if current == "SOFT":
        return "MEDIUM"
    if current == "MEDIUM":
        return "HARD"
    return "MEDIUM"


def _pit_probability(tire_age: int, expected_stint: int, ml_pit_now_probability: float | None) -> float:
    ratio = tire_age / max(1, expected_stint)
    base = 0.05
    if ratio >= 1.2:
        base = 0.85
    elif ratio >= 1.0:
        base = 0.55
    elif ratio >= 0.8:
        base = 0.25

    if ml_pit_now_probability is not None:
        return max(0.0, min(1.0, (0.65 * base) + (0.35 * ml_pit_now_probability)))
    return base


def forecast_race_outcome(
    state: RaceState,
    runs: int = 300,
    pit_loss_seconds: float = 22.0,
    use_ml_correction: bool = True,
) -> RaceOutcomeForecastResponse:
    if not state.drivers:
        return RaceOutcomeForecastResponse(
            lap=state.lap,
            total_laps=state.total_laps,
            runs=0,
            assumptions=["No driver states available"],
            drivers=[],
        )

    sim_runs = max(50, min(2000, runs))
    laps_remaining = max(state.total_laps - state.lap, 0)

    ml_payload = load_strategy_ml_model() if use_ml_correction else None

    finish_positions: Dict[str, List[int]] = defaultdict(list)
    time_deltas: Dict[str, List[float]] = defaultdict(list)

    for _ in range(sim_runs):
        # Global race noise for this continuation (incidents/SC-like pace compression)
        race_noise = random.gauss(0.0, 0.25)
        incident_multiplier = random.choice([1.0, 1.0, 1.0, 0.98, 1.03])

        driver_state = {}
        for driver in state.drivers:
            driver_state[driver.driver] = {
                "position": float(driver.position if driver.position is not None else 20),
                "compound": (driver.compound or "MEDIUM").upper(),
                "tire_age": int(driver.tire_age),
                "base_lap_time": float(driver.last_lap_time if driver.last_lap_time is not None else 95.0),
                "cum_time": 0.0,
                "gap_ahead": float(driver.gap_ahead if driver.gap_ahead is not None else 5.0),
            }

        for _lap_offset in range(1, laps_remaining + 1):
            # Per-lap stochasticity that affects all drivers
            lap_noise = random.gauss(0.0, 0.15)

            lap_results = []
            for driver_code, dstate in driver_state.items():
                expected_stint = EXPECTED_STINT_LENGTH.get(dstate["compound"], 22)

                ml_prob = None
                if ml_payload is not None:
                    ml_prob = predict_pit_now_probability(state, driver_code, model_payload=ml_payload)

                pit_prob = _pit_probability(dstate["tire_age"], expected_stint, ml_prob)
                will_pit = random.random() < pit_prob

                lap_time = _pace_projection(dstate["base_lap_time"], dstate["tire_age"], dstate["compound"])

                # Traffic penalty when cars are close
                traffic_penalty = max(0.0, (2.0 - dstate["gap_ahead"])) * random.uniform(0.0, 0.18)

                # Pit event and resulting state reset
                if will_pit:
                    lap_time += pit_loss_seconds * random.uniform(0.95, 1.05)
                    dstate["compound"] = _next_compound(dstate["compound"])
                    dstate["tire_age"] = 0
                else:
                    dstate["tire_age"] += 1

                total_lap = (lap_time + traffic_penalty + lap_noise + race_noise) * incident_multiplier
                total_lap = max(60.0, total_lap)

                dstate["cum_time"] += total_lap
                dstate["base_lap_time"] = (0.85 * dstate["base_lap_time"]) + (0.15 * total_lap)

                lap_results.append((driver_code, dstate["cum_time"]))

            lap_results.sort(key=lambda item: item[1])
            for idx, (driver_code, _cum) in enumerate(lap_results):
                if idx == 0:
                    driver_state[driver_code]["gap_ahead"] = 3.0
                else:
                    ahead_time = lap_results[idx - 1][1]
                    current_time = lap_results[idx][1]
                    driver_state[driver_code]["gap_ahead"] = max(0.0, current_time - ahead_time)

        ordered = sorted(driver_state.items(), key=lambda kv: kv[1]["cum_time"])
        winner_time = ordered[0][1]["cum_time"]

        for rank, (driver_code, dstate) in enumerate(ordered, start=1):
            finish_positions[driver_code].append(rank)
            time_deltas[driver_code].append(dstate["cum_time"] - winner_time)

    forecasts: List[DriverRaceOutcomeForecast] = []
    total_drivers = len(state.drivers)

    for driver in state.drivers:
        code = driver.driver
        positions = finish_positions.get(code, [])
        deltas = time_deltas.get(code, [])
        if not positions or not deltas:
            continue

        counts = Counter(positions)
        distribution = [
            FinishPositionProbability(position=pos, probability=round(count / len(positions), 4))
            for pos, count in sorted(counts.items(), key=lambda kv: kv[0])
        ]

        expected_finish = float(np.mean(positions))
        expected_delta = float(np.mean(deltas))
        p_win = counts.get(1, 0) / len(positions)
        p_podium = sum(count for pos, count in counts.items() if pos <= 3) / len(positions)
        p_top10 = sum(count for pos, count in counts.items() if pos <= min(10, total_drivers)) / len(positions)
        ci_lower, ci_upper = np.percentile(np.array(deltas), [5, 95]).tolist()

        forecasts.append(
            DriverRaceOutcomeForecast(
                driver=code,
                expected_finish_position=round(expected_finish, 2),
                expected_total_time_delta=round(expected_delta, 3),
                finish_position_distribution=distribution,
                probability_win=round(p_win, 4),
                probability_podium=round(p_podium, 4),
                probability_top_10=round(p_top10, 4),
                time_delta_ci_lower=round(float(ci_lower), 3),
                time_delta_ci_upper=round(float(ci_upper), 3),
            )
        )

    forecasts.sort(key=lambda item: item.expected_finish_position)

    return RaceOutcomeForecastResponse(
        lap=state.lap,
        total_laps=state.total_laps,
        runs=sim_runs,
        assumptions=[
            "Monte Carlo rollout with stochastic pace, incidents, traffic and pit-window sampling",
            "Uses current race state as baseline and short-horizon ML pit-now correction when model is available",
            "Outputs are probabilistic estimates with 5-95% confidence interval on race-time delta to winner",
        ],
        drivers=forecasts,
    )
