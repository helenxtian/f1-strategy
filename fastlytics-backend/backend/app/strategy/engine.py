from typing import Dict, List

from .schemas import RaceState, StrategyDecision, StrategyEvaluation


EXPECTED_STINT_LENGTH: Dict[str, int] = {
    "SOFT": 14,
    "MEDIUM": 22,
    "HARD": 30,
    "INTERMEDIATE": 18,
    "WET": 20,
}


def _rolling_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_strategy_rules(state: RaceState, recent_laps_by_driver: Dict[str, List[float]] | None = None) -> StrategyEvaluation:
    recent_laps_by_driver = recent_laps_by_driver or {}
    decisions: List[StrategyDecision] = []

    for driver in state.drivers:
        reasons: List[str] = []
        recommend_pit = False

        compound = (driver.compound or "").upper()
        expected_stint = EXPECTED_STINT_LENGTH.get(compound, 22)

        if driver.tire_age > expected_stint:
            recommend_pit = True
            reasons.append(f"Tire age {driver.tire_age} > expected stint {expected_stint}")

        recent = [lap for lap in recent_laps_by_driver.get(driver.driver, []) if lap is not None][-3:]
        if len(recent) >= 3:
            first_two_avg = _rolling_mean(recent[:2])
            last_avg = recent[-1]
            if (last_avg - first_two_avg) > 0.8:
                recommend_pit = True
                reasons.append("Last lap pace dropped by > 0.8s")

        if driver.gap_ahead is not None and driver.gap_ahead < 2.0:
            recommend_pit = True
            reasons.append("Undercut window under 2.0s")

        confidence = 0.35 + min(0.15 * len(reasons), 0.6)
        decisions.append(
            StrategyDecision(
                driver=driver.driver,
                recommend_pit=recommend_pit,
                reasons=reasons,
                confidence=round(min(confidence, 0.95), 2),
            )
        )

    return StrategyEvaluation(lap=state.lap, total_laps=state.total_laps, decisions=decisions)
