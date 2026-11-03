"""Evaluate saved checkpoints, then record full-scene progression and best videos.

No auto-reset, no skipped failures, and no assumption that the latest saved
checkpoint completed the training plan. Runtime imports happen only in main().
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import shutil

from mjlab_microduck.jenga_evaluation import (
    METRICS, RANKING, checkpoint_iteration, discover_checkpoints, progression,
    save_json, select_best, summarize, video_filename, wilson)
from mjlab_microduck.jenga_strike import criteria_dict


def pick_checkpoints(directory):
    """Compatibility helper for the original early/middle/late CPU test."""
    paths = discover_checkpoints(directory)
    if len(paths) < 3:
        raise ValueError("Need three checkpoints for early/middle/late comparison")
    return [("early", paths[0]), ("middle", paths[len(paths)//2]), ("late", paths[-1])]


def run_episode(env, wrapped, policy, seed, torch_module, writer=None):
    env.reset(seed=seed)
    obs = wrapped.get_observations()
    assert obs["actor"].shape == (1, 61), obs["actor"].shape
    trace = []
    if writer is not None:
        writer.append_data(env.render())
    for step in range(env.max_episode_length + 1):
        with torch_module.inference_mode():
            action = policy(obs)
        # The simulator's persistent state buffers must remain mutable on later
        # resets; inference mode is appropriate for the policy only.
        obs, reward, done, extras = wrapped.step(action.clone())
        state = env._jenga
        trace.append({"time_s": round((step+1)*env.step_dt, 6),
                      "reward": float(reward[0]),
                      **{key: float(state[key][0]) for key in METRICS}})
        if writer is not None:
            writer.append_data(env.render())
        if bool(done[0]):
            break
    else:
        raise RuntimeError("Episode did not terminate within configured horizon")
    return {"seed":seed, "duration_s":trace[-1]["time_s"],
            **{key:trace[-1][key] for key in METRICS},
            "terminated":bool(env.reset_terminated[0]),
            "timeout":bool(env.reset_time_outs[0])}, trace


def save_trace(path, trace):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)


def center_episode_camera(env):
    """Center the existing offscreen camera after randomized placement."""
    robot = env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
    tower = env.scene["jenga_0"].data.root_link_pos_w[0].detach().cpu().numpy()
    env._offline_renderer._cam.lookat[:] = ((robot[0] + tower[0]) / 2,
                                             (robot[1] + tower[1]) / 2, 0.12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Mjlab-JengaHead-Vertical3-MicroDuck")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", action="append", metavar="LABEL=PATH")
    source.add_argument("--checkpoint-dir", type=Path,
                        help="Evaluate ALL numeric saved checkpoints before selecting best")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 74)))
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--best-video-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Save full metrics and select a checkpoint without rendering videos")
    parser.add_argument("--tower-offset", type=float, default=0.20,
                        help="Tower reset distance in metres (default: original 0.20)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.14 <= args.tower_offset <= 0.20:
        parser.error("Tower offset must be between 0.14 and 0.20 metres")
    if args.checkpoint_dir:
        paths = discover_checkpoints(args.checkpoint_dir)
        requests = progression(paths)
    else:
        requests = [(label, Path(path)) for label, path in
                    (item.split("=", 1) for item in args.checkpoint)]
        paths = [path for _, path in requests]
    for label, path in requests:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or label.startswith("best") or not path.is_file():
            raise ValueError(f"Invalid/reserved label or missing checkpoint: {label}={path}")
    if len(set(label for label, _ in requests)) != len(requests):
        raise ValueError("Checkpoint labels must be unique")
    iterations = [checkpoint_iteration(path) for path in paths]
    if len(set(iterations)) != len(iterations):
        raise ValueError("Checkpoint iterations must be unique")
    seed_groups = (args.seeds,) if args.evaluation_only else (args.seeds, args.video_seeds, args.best_video_seeds)
    for seeds in seed_groups:
        if len(seeds) != len(set(seeds)) or not set(seeds) <= set(args.seeds):
            raise ValueError("Seeds must be unique; every video seed must be evaluated")
    video_filename(args.task, 0, "validation", args.seeds[0])
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v2 as imageio
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    import mjlab_microduck.tasks  # register pinned plugin tasks

    def load(checkpoint, render):
        cfg = load_env_cfg(args.task, play=True)
        cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, 41, False
        cfg.events["reset_jenga"].params["offset_x"] = args.tower_offset
        agent = load_rl_cfg(args.task)
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device,
                                render_mode="rgb_array" if render else None)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = load_runner_cls(args.task)(wrapped, asdict(agent), device=args.device)
            runner.load(str(checkpoint), load_cfg={"actor":True}, strict=True, map_location=args.device)
            return env, wrapped, runner, runner.get_inference_policy(device=args.device)
        except BaseException:
            env.close()
            raise

    manifest = {"task":args.task, "tower_offset_m":args.tower_offset,
                "actor":"unchanged 61D blind placement prototype; no camera/ToF input",
                "fast_strike_criteria":criteria_dict(), "sampling_hz":50,
                "sampling_limit":"Impacts shorter than one 20 ms sample may be missed; force maxima are sampled.",
                "startup_randomization_seed":41, "evaluation_seeds":args.seeds,
                "ranking":RANKING, "checkpoint_results":[], "videos":[],
                "evaluation_complete":False, "video_complete":False,
                "comparison_limit":"Best is selected on these evaluation seeds, not an independent held-out test. Video replays may differ on CUDA."}
    save_json(args.output/"manifest.json", manifest)
    all_rows = []
    for checkpoint in paths:
        iteration = checkpoint_iteration(checkpoint)
        record = {"checkpoint":str(checkpoint.resolve()), "checkpoint_iteration":iteration,
                  "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "episodes":[],
                  "summary":summarize([], args.seeds)}
        manifest["checkpoint_results"].append(record)
        env, wrapped, runner, policy = load(checkpoint, False)
        try:
            for seed in args.seeds:
                row, trace = run_episode(env, wrapped, policy, seed, torch)
                row.update(checkpoint_iteration=iteration)
                stem = f"evaluation_checkpoint_{iteration:05d}_seed_{seed}"
                save_trace(args.output/f"{stem}.csv", trace)
                save_json(args.output/f"{stem}.json", row)
                record["episodes"].append(row)
                all_rows.append(row)
                record["summary"] = summarize(record["episodes"], args.seeds)
                save_json(args.output/"manifest.json", manifest)
                print(f"Evaluated checkpoint={iteration} seed={seed} success={row['success']}", flush=True)
        finally:
            env.close()
            del runner, policy, wrapped, env
    best, best_role = select_best(manifest["checkpoint_results"])
    manifest["evaluation_complete"] = True
    manifest["best"] = {"checkpoint_iteration":best["checkpoint_iteration"], "sha256":best["sha256"],
                        "label":best_role, "success_rate":best["summary"]["success_rate"],
                        "selection_scope":"completely evaluated provided/saved checkpoints only"}
    save_json(args.output/"manifest.json", manifest)
    if args.evaluation_only:
        save_trace(args.output/"episodes.csv", all_rows)
        return
    requests.append((best_role, Path(best["checkpoint"])))
    # Render each checkpoint/seed once; a distinct best MP4 can be an explicitly
    # documented byte-for-byte copy when it is also a progression checkpoint.
    for checkpoint in paths:
        roles = [role for role, path in requests if path.resolve() == checkpoint.resolve()]
        if not roles:
            continue
        seeds = sorted({seed for role in roles for seed in
                        (args.best_video_seeds if role == best_role else args.video_seeds)})
        iteration = checkpoint_iteration(checkpoint)
        env, wrapped, runner, policy = load(checkpoint, True)
        try:
            for seed in seeds:
                seed_roles = [role for role in roles if seed in
                              (args.best_video_seeds if role == best_role else args.video_seeds)]
                name = video_filename(args.task, iteration, seed_roles[0], seed)
                writer = imageio.get_writer(args.output/name, fps=round(1/env.step_dt),
                                            codec="libx264", quality=7)
                try:
                    row, trace = run_episode(env, wrapped, policy, seed, torch, writer)
                finally:
                    writer.close()
                for role in seed_roles:
                    target = video_filename(args.task, iteration, role, seed)
                    if target != name:
                        shutil.copyfile(args.output/name, args.output/target)
                    video = {**row, "role":role, "checkpoint_iteration":iteration,
                             "video":target, "video_sha256":hashlib.sha256((args.output/target).read_bytes()).hexdigest(),
                             "copied_from":name if target != name else None,
                             "frames":len(trace)+1, "fps":round(1/env.step_dt)}
                    save_trace(args.output/f"{Path(target).stem}.csv", trace)
                    save_json(args.output/f"{Path(target).stem}.json", video)
                    manifest["videos"].append(video)
                    save_json(args.output/"manifest.json", manifest)
        finally:
            env.close()
            del runner, policy, wrapped, env
    save_trace(args.output/"episodes.csv", all_rows)
    manifest["video_complete"] = True
    save_json(args.output/"manifest.json", manifest)


if __name__ == "__main__":
    main()
