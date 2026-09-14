from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter


def _duration_seconds(x: pd.DataFrame) -> float:
    if len(x) < 2:
        return float("nan")
    return float((x["Date_Time"].max() - x["Date_Time"].min()).total_seconds())


def _energy_wh(x: pd.DataFrame) -> float:
    x = x.sort_values("Date_Time").dropna(subset=["Date_Time", "Voltage(V)", "Current(A)"])
    if len(x) < 2:
        return float("nan")
    t = (x["Date_Time"] - x["Date_Time"].iloc[0]).dt.total_seconds().to_numpy()
    return float(abs(trapezoid(x["Voltage(V)"].to_numpy() * x["Current(A)"].to_numpy(), t)) / 3600.0)


def _interp_q_at_v(dis: pd.DataFrame, voltage: float) -> float:
    z = dis[["Voltage(V)", "_q"]].dropna().sort_values("Voltage(V)").drop_duplicates("Voltage(V)")
    if len(z) < 2 or voltage < z["Voltage(V)"].min() or voltage > z["Voltage(V)"].max():
        return float("nan")
    return float(np.interp(voltage, z["Voltage(V)"], z["_q"]))


def _ic_peak(dis: pd.DataFrame, vmin: float, vmax: float, step: float, window: int, poly: int):
    z = dis[["Voltage(V)", "_q"]].dropna().sort_values("Voltage(V)").drop_duplicates("Voltage(V)")
    if len(z) < window or z["Voltage(V)"].min() > vmin or z["Voltage(V)"].max() < vmax:
        return float("nan"), float("nan")
    grid = np.arange(vmin, vmax + step / 2, step)
    q = np.interp(grid, z["Voltage(V)"], z["_q"])
    qs = savgol_filter(q, window_length=window, polyorder=poly, mode="interp")
    # Q grows as V falls, so -dQ/dV is the positive discharge IC curve.
    ic = -np.gradient(qs, grid)
    # Exclude boundary samples, where derivative filters are least reliable.
    pad = window // 2
    valid = np.arange(pad, len(grid) - pad)
    if not len(valid):
        return float("nan"), float("nan")
    k = valid[np.nanargmax(ic[valid])]
    return float(ic[k]), float(grid[k])


def _classify_steps(cycle_df: pd.DataFrame) -> tuple[list, list, list]:
    cc, cv, discharge = [], [], []
    for step_id, s in cycle_df.groupby("Step_Index", sort=False):
        cur = s["Current(A)"].dropna()
        volt = s["Voltage(V)"].dropna()
        if len(cur) < 2 or len(volt) < 2:
            continue
        med_i, max_i = float(cur.median()), float(cur.max())
        if med_i < -0.20:
            discharge.append(step_id)
        elif max_i > 0.05:
            # CV charge has a nearly fixed upper-cutoff voltage; CC charge spans
            # a material voltage range. This is invariant to step numbering.
            if float(volt.max() - volt.min()) <= 0.035 and float(volt.median()) >= 4.15:
                cv.append(step_id)
            elif med_i > 0.20:
                cc.append(step_id)
    return cc, cv, discharge


def extract_battery_features(raw: pd.DataFrame, battery: str, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    # File-local Cycle_Index resets. A cycle identity is therefore source file +
    # local index; ordering uses the first timestamp. Duplicate time-overlaps are audited.
    candidate_groups = []
    for (fname, local_cycle), c in raw.groupby(["source_file", "Cycle_Index"], sort=False):
        candidate_groups.append((c["Date_Time"].min(), fname, local_cycle, c.sort_values("Date_Time")))
    # Some exported workbooks overlap (notably late-life files): the same cycle,
    # with the same timestamp, appears in two files. Collapse such exports by
    # cycle start timestamp and retain the more complete copy.
    unique = {}
    for item in candidate_groups:
        start = item[0]
        key = pd.Timestamp(start).round("s") if pd.notna(start) else (item[1], item[2])
        if key not in unique or len(item[3]) > len(unique[key][3]):
            unique[key] = item
    groups = list(unique.values())
    groups.sort(key=lambda x: (pd.Timestamp.max if pd.isna(x[0]) else x[0], x[1], x[2]))

    rows = []
    for global_cycle, (start, fname, local_cycle, c) in enumerate(groups, 1):
        cc_ids, cv_ids, dis_ids = _classify_steps(c)
        # Step groups can contain a zero-current boundary record. Restrict the
        # integrations and capacity range to samples where the corresponding
        # electrochemical operation is actually active.
        cc = c[c["Step_Index"].isin(cc_ids) & (c["Current(A)"] > 0.05)]
        cv = c[c["Step_Index"].isin(cv_ids) & (c["Current(A)"] > 0.01)]
        dis = c[c["Step_Index"].isin(dis_ids) & (c["Current(A)"] < -0.20)].copy()
        # Arbin capacity is cumulative within an export. Use the observed CCDS
        # range itself as the measured discharge capacity. This mirrors the
        # dataset's supplied cycle-level capacity and avoids counting the first
        # logged interval before the CCDS baseline sample.
        qbase = float(dis["Discharge_Capacity(Ah)"].min()) if len(dis) else float("nan")
        dis["_q"] = dis["Discharge_Capacity(Ah)"] - qbase
        capacity = float(dis["_q"].max()) if len(dis) else float("nan")
        hf1, hf2, eout = _energy_wh(cc), _energy_wh(cv), _energy_wh(dis)
        hf4 = _interp_q_at_v(dis, 3.4) - _interp_q_at_v(dis, 4.0)
        hf5, hf6 = _ic_peak(
            dis, cfg.IC_VOLTAGE_MIN, cfg.IC_VOLTAGE_MAX, cfg.IC_GRID_STEP_V,
            cfg.IC_SAVGOL_WINDOW, cfg.IC_SAVGOL_POLYORDER,
        )
        rows.append({
            "battery": battery, "cycle": global_cycle, "source_file": fname,
            "local_cycle": local_cycle, "start_time": start,
            "capacity_Ah": capacity, "SOH": capacity / cfg.NOMINAL_CAPACITY_AH,
            "HF1": hf1, "HF2": hf2,
            "HF3": eout / (hf1 + hf2) if np.isfinite(hf1 + hf2) and hf1 + hf2 > 0 else np.nan,
            "HF4": hf4, "HF5": hf5, "HF6": hf6,
            "HF7": _duration_seconds(cc), "HF8": _duration_seconds(cv),
            "cc_steps": ";".join(map(str, cc_ids)), "cv_steps": ";".join(map(str, cv_ids)),
            "discharge_steps": ";".join(map(str, dis_ids)), "n_points": len(c),
        })
    out = pd.DataFrame(rows)
    first_valid = out["capacity_Ah"].dropna().iloc[0]
    out["SOH_initial"] = out["capacity_Ah"] / first_valid
    audit = pd.DataFrame({
        "battery": battery,
        "field": ["cycles_total", "cycles_valid_capacity", "duplicate_cycle_id", "nonmonotone_time", "nominal_capacity_Ah"],
        "value": [len(out), out["capacity_Ah"].notna().sum(), out["cycle"].duplicated().sum(),
                  int(not out["start_time"].is_monotonic_increasing), cfg.NOMINAL_CAPACITY_AH],
    })
    audit = pd.concat([audit, pd.DataFrame({"battery": [battery], "field": ["overlapping_cycle_exports_collapsed"],
                                            "value": [len(candidate_groups)-len(groups)]})], ignore_index=True)
    return out, audit
