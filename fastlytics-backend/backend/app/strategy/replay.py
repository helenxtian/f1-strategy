from typing import List, Dict
import pandas as pd
import fastf1

from .schemas import RaceState, DriverRaceState


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _build_driver_timeline(laps: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    timelines: Dict[str, pd.DataFrame] = {}
    for driver_code, driver_laps in laps.groupby("Driver"):
        sorted_laps = driver_laps.sort_values("LapNumber").copy()
        sorted_laps["LapTimeSeconds"] = sorted_laps["LapTime"].dt.total_seconds()
        sorted_laps["CumRaceTime"] = sorted_laps["LapTimeSeconds"].cumsum()
        timelines[driver_code] = sorted_laps
    return timelines


def build_race_state_timeline(year: int, event: str, session: str = "R", lap_step: int = 1) -> List[RaceState]:
    race_session = fastf1.get_session(year, event, session)
    race_session.load()

    laps = race_session.laps.copy()
    laps = laps.dropna(subset=["LapNumber", "Driver"])  # keep valid rows
    if laps.empty:
        return []

    total_laps = int(laps["LapNumber"].max())
    driver_timelines = _build_driver_timeline(laps)

    timeline: List[RaceState] = []

    for lap in range(1, total_laps + 1, max(1, lap_step)):
        lap_rows = []
        for driver_code, driver_data in driver_timelines.items():
            current = driver_data[driver_data["LapNumber"] == lap]
            if current.empty:
                continue

            row = current.iloc[-1]
            lap_rows.append({
                "driver": driver_code,
                "position": int(row["Position"]) if not pd.isna(row.get("Position")) else None,
                "compound": str(row["Compound"]) if not pd.isna(row.get("Compound")) else None,
                "tire_age": int(row["TyreLife"]) if not pd.isna(row.get("TyreLife")) else 0,
                "last_lap_time": _safe_float(row.get("LapTimeSeconds")),
                "cum_time": _safe_float(row.get("CumRaceTime")),
                "stint_number": int(row["Stint"]) if not pd.isna(row.get("Stint")) else None,
                "is_pit_lap": bool(row.get("PitOutTime") is not pd.NaT and not pd.isna(row.get("PitOutTime"))),
            })

        lap_rows.sort(key=lambda item: item["position"] if item["position"] is not None else 999)

        drivers_state: List[DriverRaceState] = []
        for index, item in enumerate(lap_rows):
            if index == 0:
                gap_ahead = None
            else:
                prev = lap_rows[index - 1]
                if item["cum_time"] is None or prev["cum_time"] is None:
                    gap_ahead = None
                else:
                    gap_ahead = max(0.0, item["cum_time"] - prev["cum_time"])

            drivers_state.append(
                DriverRaceState(
                    driver=item["driver"],
                    position=item["position"],
                    compound=item["compound"],
                    tire_age=item["tire_age"],
                    last_lap_time=item["last_lap_time"],
                    gap_ahead=gap_ahead,
                    stint_number=item["stint_number"],
                    is_pit_lap=item["is_pit_lap"],
                )
            )

        timeline.append(
            RaceState(
                lap=lap,
                total_laps=total_laps,
                timestamp=f"lap-{lap}",
                drivers=drivers_state,
            )
        )

    return timeline
