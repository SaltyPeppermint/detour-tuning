# Bayesian optimisation for caviar detour parameters using Optuna.
#
# Maximises the number of proved expressions over integer parameters defined in PARAMS.

import subprocess
import tempfile
from pathlib import Path

import optuna
import polars as pl

optuna.logging.set_verbosity(optuna.logging.WARNING)

CAVIAR_ROOT = Path(__file__).resolve().parent.parent

EXPRESSIONS_FILE = "./data/prefix/evaluation.csv"
ITER_LIMIT = 10_000_000
NODE_LIMIT = 10_000_000
TIME_LIMIT = 3

N_TRIALS = 50
RNG_SEED = 42

# Parameters to optimise: name -> (inclusive_low, inclusive_high).
PARAMS: dict[str, tuple[int, int]] = {
    "offset": (1, 100_000),
}

# Warm-start samples: list of (params_dict, solved_count) pairs.
INITIAL_SAMPLES: list[tuple[dict[str, int], int]] = [
    ({"offset": 3}, 4516),
    ({"offset": 9}, 4539),
]


def _count_solved(csv_path: str) -> int:
    df = pl.read_csv(
        csv_path,
        schema_overrides={"result": pl.Boolean},
        null_values=["", "null", "NULL"],
    )
    return int(df["result"].sum())


def run_caviar(params: dict[str, int], out_path: str) -> int:
    cmd = [
        str(CAVIAR_ROOT / "target" / "release" / "caviar"),
        "--expressions-file",
        EXPRESSIONS_FILE,
        "-i",
        str(ITER_LIMIT),
        "-n",
        str(NODE_LIMIT),
        "-t",
        str(TIME_LIMIT),
        "--out-path",
        out_path,
        "prove",
        "detour",
    ]
    for name, value in params.items():
        cmd += [f"--{name}", str(value)]
    subprocess.run(
        cmd,
        check=True,
        cwd=CAVIAR_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return _count_solved(out_path)


def objective(trial: optuna.Trial) -> float:
    params = {
        name: trial.suggest_int(name, low, high) for name, (low, high) in PARAMS.items()
    }
    out_path = (
        str(trial.study.user_attrs["tmpdir"])
        + f"/detour_{'_'.join(str(v) for v in params.values())}.csv"
    )
    # n_total = N_TRIALS + len(INITIAL_SAMPLES)
    # params_str = "  ".join(f"{k}={v}" for k, v in params.items())
    # print(f"[{trial.number + 1:3d}/{n_total}] {params_str} ...", end=" ", flush=True)
    solved = run_caviar(params, out_path)
    print(f"solved={solved}")
    return float(solved)


def optimise() -> None:
    from datetime import datetime

    start = datetime.now()
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}")

    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = optuna.samplers.GPSampler(
            seed=RNG_SEED, n_startup_trials=10, deterministic_objective=True
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.set_user_attr("tmpdir", tmpdir)

        # Enqueue warm-start samples.
        for params_dict, solved in INITIAL_SAMPLES:
            study.enqueue_trial(params_dict)
            params_str = "  ".join(f"{k}={v}" for k, v in params_dict.items())
            print(f"[warm-start] {params_str}  solved={solved}")
        if INITIAL_SAMPLES:
            print()

        study.optimize(
            objective, n_trials=N_TRIALS + len(INITIAL_SAMPLES), show_progress_bar=True
        )

    end = datetime.now()
    print(
        f"\nStart: {start:%Y-%m-%d %H:%M:%S}  End: {end:%Y-%m-%d %H:%M:%S}  Duration: {end - start}"
    )
    print(f"Best: {study.best_params}  solved={int(study.best_value)}")
    print("\nFull results (sorted by solved desc):")
    for t in sorted(study.trials, key=lambda t: -t.value):  # type: ignore[arg-type]
        params_str = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  {params_str}  solved={int(t.value)}")  # type: ignore[arg-type]


if __name__ == "__main__":
    optimise()
