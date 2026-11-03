"""Checkpoint discovery, complete-evaluation ranking and video names; no GPU imports."""
import json
import math
from pathlib import Path
import re


CHECKPOINT = re.compile(r"model_(\d+)\.pt")
FAILURES = ("robot_fall", "base_failure", "wrong_contact")
METRICS = ("success", *FAILURES, "head_contact", "top_fallen", "fast_strike",
           "base_max_xy_m", "base_max_tilt_rad", "max_head_force_n",
           "strike_head_speed_m_s", "strike_contact_duration_s", "strike_separation_delay_s")


def checkpoint_iteration(path):
    match = CHECKPOINT.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"Expected model_<iteration>.pt: {path}")
    return int(match[1])


def discover_checkpoints(directory):
    paths = [p for p in Path(directory).glob("model_*.pt") if p.is_file() and CHECKPOINT.fullmatch(p.name)]
    paths.sort(key=checkpoint_iteration)
    iterations = [checkpoint_iteration(p) for p in paths]
    if not paths or len(iterations) != len(set(iterations)):
        raise ValueError("Need saved checkpoints with unique numeric iterations")
    return paths


def progression(paths):
    paths = sorted(paths, key=checkpoint_iteration)
    if not paths:
        raise ValueError("No checkpoints")
    if len(paths) == 1:
        return [("only_available", paths[0])]
    if len(paths) == 2:
        return [("early", paths[0]), ("late_available", paths[-1])]
    return [("early", paths[0]), ("middle", paths[len(paths)//2]), ("late_available", paths[-1])]


def wilson(successes, count):
    if not count:
        return [0.0, 1.0]
    z, p = 1.96, successes / count
    denominator = 1 + z*z/count
    middle = (p + z*z/(2*count)) / denominator
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0.0, middle-half), min(1.0, middle+half)]


def summarize(episodes, expected_seeds):
    seeds = [row["seed"] for row in episodes]
    expected = list(expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected seeds must be nonempty and unique")
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(expected):
        raise ValueError("Duplicate or unexpected episode seeds")
    for row in episodes:
        if any(not math.isfinite(float(row[key])) for key in METRICS):
            raise ValueError("Nonfinite episode metric")
        if row["success"] and (not row["fast_strike"] or any(row[key] for key in FAILURES)):
            raise ValueError("Inconsistent success/failure metrics")
        if row["success"] and row["strike_separation_delay_s"] < 0:
            raise ValueError("Successful strike must have an observed separation time")
    count = len(episodes)
    wins = sum(bool(row["success"]) for row in episodes)
    failures = sum(any(bool(row[k]) for k in FAILURES) for row in episodes)
    means = {f"mean_{key}": sum(float(row[key]) for row in episodes)/count if count else 0.0
             for key in METRICS}
    delays = [row["strike_separation_delay_s"] for row in episodes if row["success"]]
    return {"episodes":count, "complete":set(seeds) == set(expected), "successes":wins,
            "success_rate":wins/count if count else 0.0, "success_95ci":wilson(wins,count),
            "failure_rate":failures/count if count else 0.0,
            "successful_mean_separation_delay_s":sum(delays)/len(delays) if delays else None,
            **means}


RANKING = ["higher successful fast-strike rate", "lower union failure rate",
           "lower mean lower-block horizontal displacement", "lower mean lower-block tilt",
           "lower sampled head contact force", "shorter successful separation delay",
           "earlier checkpoint iteration for exact ties"]


def select_best(records):
    eligible = [r for r in records if r["summary"]["complete"]]
    if not eligible:
        raise ValueError("No completely evaluated checkpoint; do not select from partial results")
    def key(record):
        s = record["summary"]
        return (-s["success_rate"], s["failure_rate"], s["mean_base_max_xy_m"],
                s["mean_base_max_tilt_rad"], s["mean_max_head_force_n"],
                s["successful_mean_separation_delay_s"] if s["successful_mean_separation_delay_s"] is not None else math.inf,
                record["checkpoint_iteration"])
    winner = min(eligible, key=key)
    return winner, "best" if winner["summary"]["successes"] else "best_available_no_success"


def video_filename(task, iteration, role, seed):
    match = re.fullmatch(r"Mjlab-JengaHead-(Vertical2|Vertical3|Pedestal)-MicroDuck", task)
    if not match or not re.fullmatch(r"[a-zA-Z0-9_-]+", role) or iteration < 0:
        raise ValueError("Invalid task, iteration, or role")
    return f"jenga_{match[1]}_checkpoint_{iteration:05d}_{role}_seed_{seed}.mp4"


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
