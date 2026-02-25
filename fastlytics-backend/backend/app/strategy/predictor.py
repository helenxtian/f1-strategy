from .engine import evaluate_strategy_rules
from .schemas import RaceState, StrategyPredictionResponse
from .simulator import evaluate_pit_scenarios


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
            confidence=0.5,
            expected_rejoin_position=None,
            expected_time_delta_to_second_best=0.0,
            recommendation_summary="No simulation outcomes available for this driver.",
            scenarios=[],
        )

    best = outcomes[0]
    second_best = outcomes[1] if len(outcomes) > 1 else None
    time_delta_to_second = (
        round(second_best.estimated_total_time - best.estimated_total_time, 3)
        if second_best is not None
        else 0.0
    )

    base_confidence = 0.55 + min(max(time_delta_to_second, 0.0) / 6.0, 0.35)

    evaluation = evaluate_strategy_rules(state)
    target_decision = next((decision for decision in evaluation.decisions if decision.driver == target_driver), None)

    if target_decision is not None:
        if target_decision.recommend_pit and best.scenario in {"pit_now", "pit_in_2_laps"}:
            base_confidence += 0.05
        elif target_decision.recommend_pit and best.scenario == "stay_out":
            base_confidence -= 0.08

    confidence = round(_clamp(base_confidence, 0.5, 0.95), 2)

    summary = (
        f"Predicted best option is '{best.scenario}' with ~{time_delta_to_second:.2f}s "
        f"advantage over the next best scenario."
    )

    return StrategyPredictionResponse(
        target_driver=target_driver,
        lap=state.lap,
        total_laps=state.total_laps,
        predicted_best_scenario=best.scenario,
        confidence=confidence,
        expected_rejoin_position=best.estimated_rejoin_position,
        expected_time_delta_to_second_best=time_delta_to_second,
        recommendation_summary=summary,
        scenarios=outcomes,
    )
