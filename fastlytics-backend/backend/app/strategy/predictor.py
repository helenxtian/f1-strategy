import math

from .engine import EXPECTED_STINT_LENGTH, evaluate_strategy_rules
from .schemas import (
    ConfidenceFactors,
    RaceState,
    ScenarioProbability,
    StrategyPredictionResponse,
)
from .simulator import evaluate_pit_scenarios


TIE_THRESHOLD_SECONDS = 0.2


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _scenario_probabilities(outcomes) -> list[ScenarioProbability]:
    if not outcomes:
        return []

    best_time = outcomes[0].estimated_total_time
    temperature = 0.8

    raw_scores = []
    for outcome in outcomes:
        delta = max(0.0, outcome.estimated_total_time - best_time)
        raw_scores.append(math.exp(-delta / temperature))

    score_sum = sum(raw_scores) or 1.0

    probabilities: list[ScenarioProbability] = []
    for outcome, raw in zip(outcomes, raw_scores):
        delta = max(0.0, outcome.estimated_total_time - best_time)
        probabilities.append(
            ScenarioProbability(
                scenario=outcome.scenario,
                probability=round(raw / score_sum, 3),
                time_delta_to_best=round(delta, 3),
            )
        )

    return probabilities


def _confidence_from_factors(
    state: RaceState,
    best_scenario: str,
    time_delta_to_second: float,
    target_driver: str,
    tie_detected: bool,
) -> tuple[float, ConfidenceFactors]:
    target = next((driver for driver in state.drivers if driver.driver == target_driver), None)

    margin_score = _clamp(time_delta_to_second / 1.5, 0.0, 1.0)

    evaluation = evaluate_strategy_rules(state)
    target_decision = next((decision for decision in evaluation.decisions if decision.driver == target_driver), None)
    pits_expected = best_scenario in {"pit_now", "pit_in_2_laps"}
    if target_decision is None:
        rule_agreement_score = 0.5
    else:
        rule_agreement_score = 1.0 if target_decision.recommend_pit == pits_expected else 0.25

    if target is None:
        tire_age_risk_score = 0.5
        traffic_risk_score = 0.5
    else:
        compound = (target.compound or "MEDIUM").upper()
        expected_stint = EXPECTED_STINT_LENGTH.get(compound, 22)
        tire_ratio = _clamp(target.tire_age / max(1, expected_stint), 0.0, 1.5)
        tire_risk = _clamp(tire_ratio / 1.2, 0.0, 1.0)
        if pits_expected:
            tire_age_risk_score = tire_risk
        else:
            tire_age_risk_score = 1.0 - tire_risk

        if target.gap_ahead is None:
            traffic_risk = 0.5
        else:
            traffic_risk = _clamp((2.0 - target.gap_ahead) / 2.0, 0.0, 1.0)
        if pits_expected:
            traffic_risk_score = traffic_risk
        else:
            traffic_risk_score = 1.0 - traffic_risk

    laps_remaining = max(state.total_laps - state.lap, 0)
    laps_ratio = laps_remaining / max(1, state.total_laps)
    if pits_expected:
        laps_remaining_score = 0.35 if laps_ratio < 0.12 else 0.75
    else:
        laps_remaining_score = 0.85 if laps_ratio < 0.12 else 0.55

    weighted_score = (
        (0.25 * margin_score)
        + (0.25 * rule_agreement_score)
        + (0.2 * tire_age_risk_score)
        + (0.15 * traffic_risk_score)
        + (0.15 * laps_remaining_score)
    )

    confidence = 0.45 + (0.5 * weighted_score)
    if tie_detected:
        confidence = min(confidence, 0.6)

    factors = ConfidenceFactors(
        margin_score=round(margin_score, 3),
        rule_agreement_score=round(rule_agreement_score, 3),
        tire_age_risk_score=round(tire_age_risk_score, 3),
        traffic_risk_score=round(traffic_risk_score, 3),
        laps_remaining_score=round(laps_remaining_score, 3),
    )

    return round(_clamp(confidence, 0.5, 0.95), 2), factors


def predict_best_strategy(
    state: RaceState,
    target_driver: str,
    pit_loss_seconds: float = 22.0,
) -> StrategyPredictionResponse:
    outcomes = evaluate_pit_scenarios(
        state=state,
        target_driver=target_driver,
        pit_loss_seconds=pit_loss_seconds,
    )

    if not outcomes:
        return StrategyPredictionResponse(
            target_driver=target_driver,
            lap=state.lap,
            total_laps=state.total_laps,
            predicted_best_scenario="unknown",
            tie_detected=False,
            tie_threshold_seconds=TIE_THRESHOLD_SECONDS,
            confidence=0.5,
            expected_rejoin_position=None,
            expected_time_delta_to_second_best=0.0,
            recommendation_summary="No simulation outcomes available for this driver.",
            scenarios=[],
            scenario_probabilities=[],
            confidence_factors=ConfidenceFactors(
                margin_score=0.0,
                rule_agreement_score=0.5,
                tire_age_risk_score=0.5,
                traffic_risk_score=0.5,
                laps_remaining_score=0.5,
            ),
        )

    best = outcomes[0]
    second_best = outcomes[1] if len(outcomes) > 1 else None
    time_delta_to_second = (
        round(second_best.estimated_total_time - best.estimated_total_time, 3)
        if second_best is not None
        else 0.0
    )
    tie_detected = time_delta_to_second < TIE_THRESHOLD_SECONDS

    confidence, factors = _confidence_from_factors(
        state=state,
        best_scenario=best.scenario,
        time_delta_to_second=time_delta_to_second,
        target_driver=target_driver,
        tie_detected=tie_detected,
    )
    probabilities = _scenario_probabilities(outcomes)

    if tie_detected and second_best is not None:
        predicted_label = "no_clear_winner"
        summary = (
            f"No clear winner: '{best.scenario}' and '{second_best.scenario}' are within "
            f"{TIE_THRESHOLD_SECONDS:.1f}s ({time_delta_to_second:.2f}s gap)."
        )
    else:
        predicted_label = best.scenario
        summary = (
            f"Predicted best option is '{best.scenario}' with ~{time_delta_to_second:.2f}s "
            f"advantage over the next best scenario."
        )

    return StrategyPredictionResponse(
        target_driver=target_driver,
        lap=state.lap,
        total_laps=state.total_laps,
        predicted_best_scenario=predicted_label,
        tie_detected=tie_detected,
        tie_threshold_seconds=TIE_THRESHOLD_SECONDS,
        confidence=confidence,
        expected_rejoin_position=best.estimated_rejoin_position,
        expected_time_delta_to_second_best=time_delta_to_second,
        recommendation_summary=summary,
        scenarios=outcomes,
        scenario_probabilities=probabilities,
        confidence_factors=factors,
    )
