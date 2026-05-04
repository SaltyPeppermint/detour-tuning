# Bayesian optimisation for Herbie cfg_offset using Optuna.
#
# Each trial:
#   1. Patches `let cfg_offset = N;` in egg-herbie/src/lib.rs.
#   2. Runs `make install` in the herbie directory.
#   3. Runs `run.sh`, producing `out`.
#   4. Parses `out` (lines containing " has cost ") into a list of detour costs D.
#   5. Compares against the precomputed baseline `original` file via the same
#      logic as cmp.py and returns the chosen objective.

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERBIE_ROOT = Path(__file__).resolve().parent / "herbie"
LIB_RS = HERBIE_ROOT / "egg-herbie" / "src" / "lib.rs"
RUN_SH = HERBIE_ROOT / "run.sh"
OUT_FILE = HERBIE_ROOT / "out"
ORIGINAL_FILE = HERBIE_ROOT / "original"

# Objective: "dwins" (maximise detour wins vs original baseline)
#         or "total_cost" (minimise sum of detour costs).
OBJECTIVE = "dwins"

N_TRIALS = 50
RNG_SEED = 42

PARAMS: dict[str, tuple[int, int]] = {
    "offset": (1, 100_000),
}

INITIAL_SAMPLES: list[dict[str, int]] = [
    {"offset": 30},
    {"offset": 50},
]

OFFSET_RE = re.compile(r"^(\s*)let cfg_offset\s*=\s*\d+\s*;", re.MULTILINE)
COST_RE = re.compile(r" has cost ")


def patch_offset(value: int) -> None:
    text = LIB_RS.read_text()
    new_text, n = OFFSET_RE.subn(rf"\1let cfg_offset = {value};", text, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to patch cfg_offset in {LIB_RS}")
    LIB_RS.write_text(new_text)


def make_install() -> None:
    subprocess.run(
        ["make", "install"],
        check=True,
        cwd=HERBIE_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_herbie() -> None:
    if OUT_FILE.exists():
        OUT_FILE.unlink()
    subprocess.run(
        ["bash", str(RUN_SH)],
        check=True,
        cwd=HERBIE_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_costs(path: Path) -> list[int]:
    costs: list[int] = []
    with path.open() as f:
        for line in f:
            if " has cost " not in line:
                continue
            _, c = line.split(" has cost ")
            costs.append(int(c))
    return costs


def evaluate() -> tuple[int, int, int]:
    """Return (dwins, owins, total_detour_cost)."""
    original_file = parse_costs(ORIGINAL_FILE)
    this_run_file = parse_costs(OUT_FILE)
    if len(original_file) != len(this_run_file):
        raise RuntimeError(
            f"length mismatch: original has {len(original_file)} costs, detour has {len(this_run_file)}"
        )
    owins = dwins = 0
    for o, d in zip(original_file, this_run_file):
        if o < d:
            owins += 1
        elif o > d:
            dwins += 1
    return dwins, owins, sum(this_run_file)


def objective(trial: optuna.Trial) -> float:
    params = {
        name: trial.suggest_int(name, low, high) for name, (low, high) in PARAMS.items()
    }
    patch_offset(params["offset"])
    make_install()
    run_herbie()
    dwins, owins, total = evaluate()
    print(
        f"  offset={params['offset']}  dwins={dwins}  owins={owins}  total_cost={total}",
        flush=True,
    )
    if OBJECTIVE == "dwins":
        return float(dwins)
    elif OBJECTIVE == "total_cost":
        return float(total)
    else:
        raise ValueError(f"unknown OBJECTIVE: {OBJECTIVE}")


def optimise() -> None:
    direction = "maximize" if OBJECTIVE == "dwins" else "minimize"
    start = datetime.now()
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}  objective={OBJECTIVE} ({direction})")

    sampler = optuna.samplers.GPSampler(
        seed=RNG_SEED, n_startup_trials=10, deterministic_objective=True
    )
    study = optuna.create_study(direction=direction, sampler=sampler)

    for params_dict in INITIAL_SAMPLES:
        study.enqueue_trial(params_dict)
        params_str = "  ".join(f"{k}={v}" for k, v in params_dict.items())
        print(f"[warm-start] {params_str}")
    if INITIAL_SAMPLES:
        print()

    study.optimize(
        objective, n_trials=N_TRIALS + len(INITIAL_SAMPLES), show_progress_bar=True
    )

    end = datetime.now()
    print(
        f"\nStart: {start:%Y-%m-%d %H:%M:%S}  End: {end:%Y-%m-%d %H:%M:%S}  Duration: {end - start}"
    )
    print(f"Best: {study.best_params}  value={study.best_value}")
    print("\nFull results (sorted by value):")
    reverse = direction == "maximize"
    for t in sorted(
        (t for t in study.trials if t.value is not None),
        key=lambda t: t.value,  # type: ignore[arg-type,return-value]
        reverse=reverse,
    ):
        params_str = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  {params_str}  value={t.value}")


if __name__ == "__main__":
    sys.exit(optimise())
