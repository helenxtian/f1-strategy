import React, { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import Navbar from '@/components/Navbar';
import { useQuery } from '@tanstack/react-query';
import { useSeason } from '@/contexts/SeasonContext';
import {
  evaluateStrategyState,
  fetchStrategyModelInfo,
  fetchPrediction,
  fetchReplayTimeline,
  fetchSchedule,
  simulateStrategyScenarios,
  trainStrategyModel,
  ReplayTick,
  StrategyModelInfoResponse,
  ScheduleEvent,
  StrategyPredictionResponse,
  StrategyEvaluationResponse,
  ScenarioSimulationResponse,
} from '@/lib/api';

const StrategySimulator = () => {
  const { selectedYear, setSelectedYear, availableYears } = useSeason();
  const [roundNumber, setRoundNumber] = useState<string>('');
  const [lapStep, setLapStep] = useState(1);
  const [ticks, setTicks] = useState<ReplayTick[]>([]);
  const [selectedLap, setSelectedLap] = useState<number>(0);
  const [selectedDriver, setSelectedDriver] = useState<string>('');
  const [evalResult, setEvalResult] = useState<StrategyEvaluationResponse | null>(null);
  const [simResult, setSimResult] = useState<ScenarioSimulationResponse | null>(null);
  const [predictionResult, setPredictionResult] = useState<StrategyPredictionResponse | null>(null);
  const [trainStatus, setTrainStatus] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<StrategyModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: schedule = [], isLoading: scheduleLoading } = useQuery<ScheduleEvent[]>({
    queryKey: ['schedule', selectedYear],
    queryFn: () => fetchSchedule(selectedYear),
    staleTime: 1000 * 60 * 30,
  });

  const orderedSchedule = useMemo(
    () => [...schedule].sort((a, b) => Number(a.RoundNumber) - Number(b.RoundNumber)),
    [schedule]
  );

  const selectedEvent = useMemo(
    () => orderedSchedule.find((event) => String(event.RoundNumber) === roundNumber) ?? null,
    [orderedSchedule, roundNumber]
  );

  const eventSlug = useMemo(() => {
    if (!selectedEvent?.EventName) return '';
    return selectedEvent.EventName.toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-');
  }, [selectedEvent]);

  const currentTick = useMemo(() => ticks[selectedLap] ?? null, [ticks, selectedLap]);
  const currentDrivers = currentTick?.drivers ?? [];

  useEffect(() => {
    if (!currentDrivers.length) {
      setSelectedDriver('');
      return;
    }

    const hasSelectedDriver = currentDrivers.some((driverState) => driverState.driver === selectedDriver);
    if (!hasSelectedDriver) {
      setSelectedDriver(currentDrivers[0].driver);
    }
  }, [currentDrivers, selectedDriver]);

  useEffect(() => {
    const loadModelInfo = async () => {
      try {
        const info = await fetchStrategyModelInfo();
        setModelInfo(info);
      } catch {
        setModelInfo(null);
      }
    };
    void loadModelInfo();
  }, []);

  const loadReplay = async () => {
    if (!eventSlug) {
      setError('Select a race round first.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const replay = await fetchReplayTimeline(selectedYear, eventSlug, 'R', lapStep);
      setTicks(replay.ticks ?? []);
      setSelectedLap(0);
      setSelectedDriver('');
      setEvalResult(null);
      setSimResult(null);
      setPredictionResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load replay');
    } finally {
      setLoading(false);
    }
  };

  const runEvaluation = async () => {
    if (!currentTick) return;
    setLoading(true);
    setError(null);
    try {
      const result = await evaluateStrategyState(currentTick);
      setEvalResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to evaluate strategy');
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    if (!currentTick || !selectedDriver) return;
    setLoading(true);
    setError(null);
    try {
      const result = await simulateStrategyScenarios(currentTick, selectedDriver, 22);
      setSimResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to simulate scenarios');
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async () => {
    if (!currentTick || !selectedDriver) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchPrediction({
        state: currentTick,
        target_driver: selectedDriver,
        pit_loss_seconds: 22,
      });
      setPredictionResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to predict strategy');
    } finally {
      setLoading(false);
    }
  };

  const runTrainModel = async () => {
    setLoading(true);
    setError(null);
    setTrainStatus('Training model...');
    try {
      const trainYears = Array.from(new Set([selectedYear, selectedYear - 1])).filter((year) => year >= 2018);
      const result = await trainStrategyModel(trainYears, 1, 8);
      setTrainStatus(
        `Model trained • ${result.training_samples} samples • train acc ${(result.training_accuracy * 100).toFixed(1)}%${typeof result.validation_roc_auc === 'number' ? ` • val AUC ${result.validation_roc_auc.toFixed(3)}` : ''}${typeof result.validation_ece === 'number' ? ` • ECE ${result.validation_ece.toFixed(3)}` : ''}`
      );
      const info = await fetchStrategyModelInfo();
      setModelInfo(info);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to train model';
      setTrainStatus('Model training failed');
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-950 via-black to-gray-950 text-white">
      <Navbar />
      <div className="container mx-auto px-4 py-8 space-y-6">
        <h1 className="text-3xl font-bold">Strategy Simulator</h1>

        <Card className="bg-gray-900/60 border-gray-700">
          <CardHeader>
            <CardTitle>Replay Input</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div className="space-y-1">
              <label className="text-sm font-medium text-gray-200">Season Year</label>
              <Select
                value={String(selectedYear)}
                onValueChange={(value) => {
                  setSelectedYear(Number(value));
                  setRoundNumber('');
                  setTicks([]);
                  setEvalResult(null);
                  setSimResult(null);
                }}
              >
                <SelectTrigger className="w-full bg-gray-800 border border-gray-700 text-white">
                  <SelectValue placeholder="Select year" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  {availableYears.map((year) => (
                    <SelectItem key={year} value={String(year)}>{year}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-gray-200">Race Round</label>
              <Select
                value={roundNumber}
                disabled={scheduleLoading || orderedSchedule.length === 0}
                onValueChange={(value) => {
                  setRoundNumber(value);
                  setTicks([]);
                  setEvalResult(null);
                  setSimResult(null);
                }}
              >
                <SelectTrigger className="w-full bg-gray-800 border border-gray-700 text-white">
                  <SelectValue placeholder="Select round" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  {orderedSchedule.map((event) => (
                    <SelectItem key={`${event.RoundNumber}-${event.EventName}`} value={String(event.RoundNumber)}>
                      {event.RoundNumber}. {event.EventName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-gray-200">Lap Sampling Step</label>
              <Select
                value={String(lapStep)}
                onValueChange={(value) => setLapStep(Number(value))}
              >
                <SelectTrigger className="w-full bg-gray-800 border border-gray-700 text-white">
                  <SelectValue placeholder="Select sampling" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="1">1 (every lap)</SelectItem>
                  <SelectItem value="2">2 (every 2 laps)</SelectItem>
                  <SelectItem value="3">3 (every 3 laps)</SelectItem>
                  <SelectItem value="5">5 (every 5 laps)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="pt-6">
              <Button onClick={loadReplay} disabled={loading} className="w-full">
                Load Replay
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="bg-red-900/20 border-red-500/50">
            <CardContent className="pt-6 text-red-300">{error}</CardContent>
          </Card>
        )}

        {ticks.length > 0 && (
          <Card className="bg-gray-900/60 border-gray-700">
            <CardHeader>
              <CardTitle>Replay Control</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                type="range"
                min={0}
                max={Math.max(0, ticks.length - 1)}
                value={selectedLap}
                onChange={(e) => setSelectedLap(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-sm text-gray-300">Lap {currentTick?.lap} / {currentTick?.total_laps} • Drivers: {currentDrivers.length}</div>
              <div className="text-xs text-gray-400">Lap slider = race timeline position used for rule evaluation and scenario simulation.</div>
              <div className="flex flex-wrap items-end gap-2">
                <Button onClick={runEvaluation} disabled={loading}>Evaluate Pit Rules</Button>
                <Button onClick={runSimulation} disabled={loading || !selectedDriver}>Simulate Pit Options</Button>
                <Button onClick={runPrediction} disabled={loading || !selectedDriver}>Predict Best Strategy</Button>
                <Button onClick={runTrainModel} disabled={loading} variant="secondary" size="sm">Train Model</Button>
                <div className="w-[220px]">
                  <Select
                    value={selectedDriver}
                    onValueChange={(value) => {
                      setSelectedDriver(value);
                      setSimResult(null);
                      setPredictionResult(null);
                    }}
                    disabled={!currentDrivers.length}
                  >
                    <SelectTrigger className="w-full bg-gray-800 border border-gray-700 text-white">
                      <SelectValue placeholder="Select driver" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-900 border-gray-700 text-white">
                      {currentDrivers.map((driverState) => (
                        <SelectItem key={driverState.driver} value={driverState.driver}>
                          {driverState.driver}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              {trainStatus && <div className="text-xs text-gray-400">{trainStatus}</div>}
              {modelInfo && (
                <div className="text-xs text-gray-400">
                  Model Info: {modelInfo.available ? `available${modelInfo.version ? ` • ${modelInfo.version}` : ''}${modelInfo.trained_at ? ` • trained ${new Date(modelInfo.trained_at).toLocaleString()}` : ''}${Array.isArray(modelInfo.years) && modelInfo.years.length ? ` • years ${modelInfo.years.join(', ')}` : ''}${typeof modelInfo.training_samples === 'number' ? ` • samples ${modelInfo.training_samples}` : ''}${typeof modelInfo.training_accuracy === 'number' ? ` • train acc ${(modelInfo.training_accuracy * 100).toFixed(1)}%` : ''}${typeof modelInfo.validation_roc_auc === 'number' ? ` • val AUC ${modelInfo.validation_roc_auc.toFixed(3)}` : ''}${typeof modelInfo.validation_ece === 'number' ? ` • ECE ${modelInfo.validation_ece.toFixed(3)}` : ''}` : 'not trained yet'}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {(evalResult || simResult || predictionResult) && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {evalResult && (
              <Card className={`bg-gray-900/60 border-gray-700 ${!simResult && !predictionResult ? 'xl:col-span-2' : ''}`}>
                <CardHeader>
                  <CardTitle>Rule Decisions (Lap {evalResult.lap})</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {evalResult.decisions.slice(0, 10).map((decision) => (
                    <div key={decision.driver} className="border border-gray-800 rounded p-2 text-sm">
                      <div className="font-semibold">{decision.driver} — {decision.recommend_pit ? 'PIT' : 'STAY OUT'} ({decision.confidence})</div>
                      <div className="text-gray-300">{decision.reasons.length ? decision.reasons.join(' • ') : 'No triggers'}</div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {(simResult || predictionResult) && (
              <div className={`space-y-6 ${!evalResult ? 'xl:col-span-2' : ''}`}>
                {predictionResult && (
                  <Card className="bg-gray-900/60 border-gray-700">
                    <CardHeader>
                      <CardTitle>Prediction</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="font-semibold">
                        Best: {predictionResult.tie_detected ? 'No clear winner' : predictionResult.predicted_best_scenario} ({Math.round(predictionResult.confidence * 100)}% confidence)
                      </div>
                      <div className="text-gray-300">
                        Driver: {predictionResult.target_driver} • Rejoin: P{predictionResult.expected_rejoin_position ?? '-'} • Advantage: {predictionResult.expected_time_delta_to_second_best.toFixed(2)}s
                      </div>
                      <div className="text-gray-300">
                        Tie threshold: {predictionResult.tie_threshold_seconds.toFixed(1)}s • Top probabilities: {predictionResult.scenario_probabilities.slice(0, 3).map((p) => `${p.scenario} ${Math.round(p.probability * 100)}%`).join(' • ')}
                      </div>
                      <div className="text-gray-300">
                        ML: {predictionResult.ml_enabled ? `enabled (${predictionResult.ml_model_version ?? 'unknown model'})` : 'fallback mode'}{predictionResult.ml_pit_now_probability !== null ? ` • Pit-now prob: ${Math.round(predictionResult.ml_pit_now_probability * 100)}%` : ''}
                      </div>
                      <div className="text-gray-400">
                        Factors — margin: {predictionResult.confidence_factors.margin_score.toFixed(2)}, rules: {predictionResult.confidence_factors.rule_agreement_score.toFixed(2)}, tire risk: {predictionResult.confidence_factors.tire_age_risk_score.toFixed(2)}, traffic risk: {predictionResult.confidence_factors.traffic_risk_score.toFixed(2)}, laps: {predictionResult.confidence_factors.laps_remaining_score.toFixed(2)}
                      </div>
                      <div className="text-gray-400">{predictionResult.recommendation_summary}</div>
                    </CardContent>
                  </Card>
                )}

                {simResult && (
                  <Card className="bg-gray-900/60 border-gray-700">
                    <CardHeader>
                      <CardTitle>Scenario Outcomes (Best: {simResult.best?.scenario ?? 'N/A'})</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {simResult.scenarios.map((scenario) => (
                        <div key={scenario.scenario} className="border border-gray-800 rounded p-2 text-sm">
                          <div className="font-semibold">{scenario.scenario} {scenario.pit_lap ? `(Pit L${scenario.pit_lap})` : ''}</div>
                          <div className="text-gray-300">Time: {scenario.estimated_total_time}s • Rejoin: P{scenario.estimated_rejoin_position ?? '-'}</div>
                          <div className="text-gray-400">{scenario.summary}</div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategySimulator;
