from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "Test_Time(s)", "Date_Time", "Step_Time(s)", "Step_Index",
    "Cycle_Index", "Current(A)", "Voltage(V)", "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)", "Charge_Energy(Wh)", "Discharge_Energy(Wh)",
]


def inspect_workbooks(data_root: Path, batteries: list[str]) -> pd.DataFrame:
    rows = []
    for battery in batteries:
        for path in sorted((data_root / battery).glob("*.xlsx")):
            book = pd.ExcelFile(path)
            sheet = next((s for s in book.sheet_names if s.lower() != "info"), None)
            if sheet is None:
                rows.append({"battery": battery, "file": path.name, "status": "no_data_sheet"})
                continue
            sample = pd.read_excel(path, sheet_name=sheet, nrows=5)
            rows.append({
                "battery": battery, "file": path.name, "data_sheet": sheet,
                "fields": json.dumps(list(sample.columns), ensure_ascii=False),
                "has_required_fields": set(REQUIRED_COLUMNS).issubset(sample.columns),
                "status": "ok",
            })
    return pd.DataFrame(rows)


def read_battery_raw(data_root: Path, battery: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, inventory = [], []
    for path in sorted((data_root / battery).glob("*.xlsx")):
        book = pd.ExcelFile(path)
        sheet = next((s for s in book.sheet_names if s.lower() != "info"), None)
        if sheet is None:
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if missing:
            inventory.append({"file": path.name, "status": "missing_fields", "missing": ";".join(missing)})
            continue
        df = df[REQUIRED_COLUMNS].copy()
        df["source_file"] = path.name
        df["source_row"] = range(2, len(df) + 2)
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
        frames.append(df)
        inventory.append({
            "file": path.name, "status": "ok", "rows": len(df),
            "first_time": df["Date_Time"].min(), "last_time": df["Date_Time"].max(),
            "local_cycles": df["Cycle_Index"].nunique(),
        })
    if not frames:
        raise RuntimeError(f"No usable workbooks found for {battery}")
    raw = pd.concat(frames, ignore_index=True).sort_values(["Date_Time", "source_file", "source_row"])
    return raw.reset_index(drop=True), pd.DataFrame(inventory)

