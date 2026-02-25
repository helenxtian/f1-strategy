from typing import List, Optional

from .schemas import RaceState, ScenarioOutcome


def _pace_projection(base_time: float, tire_age: int, compound: Optional[str]) -> float:
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


def evaluate_pit_scenarios(state: RaceState, target_driver: str, pit_loss_seconds: float = 22.0) -> List[ScenarioOutcome]:
    target = next((driver for driver in state.drivers if driver.driver == target_driver), None)
    if target is None:
        return []

    base_lap_time = target.last_lap_time or 95.0
    laps_remaining = max(state.total_laps - state.lap, 0)

    scenarios = [
        ("pit_now", state.lap),
        ("pit_in_2_laps", min(state.total_laps, state.lap + 2)),
        ("stay_out", None),
    ]

    results: List[ScenarioOutcome] = []

    for scenario_name, pit_lap in scenarios:
        running_time = 0.0
        tire_age = target.tire_age
        current_compound = target.compound

        for lap_offset in range(1, laps_remaining + 1):
            simulated_lap = state.lap + lap_offset

            if pit_lap is not None and simulated_lap == pit_lap:
                running_time += pit_loss_seconds
                tire_age = 0
                current_compound = "MEDIUM" if (target.compound or "MEDIUM").upper() != "MEDIUM" else "HARD"

            running_time += _pace_projection(base_lap_time, tire_age, current_compound)
            tire_age += 1

        if scenario_name == "stay_out":
            estimated_rejoin = target.position
        else:
            position_penalty = max(1, int(pit_loss_seconds // 8))
            estimated_rejoin = (target.position or 1) + position_penalty

        results.append(
            ScenarioOutcome(
                scenario=scenario_name,
                pit_lap=pit_lap,
                estimated_total_time=round(running_time, 3),
                estimated_rejoin_position=estimated_rejoin,
                summary=f"{scenario_name} projects +{round(running_time, 2)}s over remaining laps",
            )
        )

    results.sort(key=lambda outcome: outcome.estimated_total_time)
    return results
