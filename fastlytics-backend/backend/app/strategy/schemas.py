from pydantic import BaseModel, Field
from typing import List, Optional


class DriverRaceState(BaseModel):
    driver: str
    position: Optional[int] = None
    compound: Optional[str] = None
    tire_age: int = 0
    last_lap_time: Optional[float] = None
    gap_ahead: Optional[float] = None
    stint_number: Optional[int] = None
    is_pit_lap: bool = False


class RaceState(BaseModel):
    lap: int
    total_laps: int
    timestamp: Optional[str] = None
    drivers: List[DriverRaceState] = Field(default_factory=list)


class StrategyDecision(BaseModel):
    driver: str
    recommend_pit: bool
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class StrategyEvaluation(BaseModel):
    lap: int
    total_laps: int
    decisions: List[StrategyDecision]


class ScenarioOutcome(BaseModel):
    scenario: str
    pit_lap: Optional[int] = None
    estimated_total_time: float
    estimated_rejoin_position: Optional[int] = None
    summary: str


class ScenarioEvaluationRequest(BaseModel):
    state: RaceState
    target_driver: str
    pit_loss_seconds: float = 22.0


class StrategyPredictionRequest(BaseModel):
    state: RaceState
    target_driver: str
    pit_loss_seconds: float = 22.0


class ScenarioProbability(BaseModel):
    scenario: str
    probability: float
    time_delta_to_best: float


class ConfidenceFactors(BaseModel):
    margin_score: float
    rule_agreement_score: float
    tire_age_risk_score: float
    traffic_risk_score: float
    laps_remaining_score: float


class StrategyPredictionResponse(BaseModel):
    target_driver: str
    lap: int
    total_laps: int
    predicted_best_scenario: str
    tie_detected: bool = False
    tie_threshold_seconds: float = 0.2
    confidence: float
    expected_rejoin_position: Optional[int] = None
    expected_time_delta_to_second_best: float
    recommendation_summary: str
    scenarios: List[ScenarioOutcome] = Field(default_factory=list)
    scenario_probabilities: List[ScenarioProbability] = Field(default_factory=list)
    confidence_factors: ConfidenceFactors


class ReplayResponse(BaseModel):
    year: int
    event: str
    session: str
    ticks: List[RaceState]
