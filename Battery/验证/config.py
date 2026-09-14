from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"E:\CodeDir\Battery\data\CALCE")
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "cache"

BATTERIES = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
MAIN_HORIZONS = [1, 4, 8, 12, 16, 24, 32]
EXTENDED_HORIZONS = [48, 64]
FEATURES = ["SOH"] + [f"HF{i}" for i in range(1, 9)]
HFS = [f"HF{i}" for i in range(1, 9)]

# CALCE CS2 cells are rated at 1.1 Ah. This common rated-capacity
# denominator is used for all batteries; initial-capacity SOH is also saved.
NOMINAL_CAPACITY_AH = 1.1

# Uniform incremental-capacity extraction rule for every cell and cycle.
IC_VOLTAGE_MIN = 3.0
IC_VOLTAGE_MAX = 4.0
IC_GRID_STEP_V = 0.005
IC_SAVGOL_WINDOW = 21
IC_SAVGOL_POLYORDER = 3

BOOTSTRAP_N = 1000
BLOCK_LENGTHS = [10, 20, 30]
RANDOM_SEED = 42
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

