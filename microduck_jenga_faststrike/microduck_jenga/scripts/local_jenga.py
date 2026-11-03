"""Bounded local PPO runner; no provider API, upload, or physical robot access."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['smoke', 'train'])
    parser.add_argument('--envs', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tower-offset', type=float, default=0.20,
                        help='Experimental reset distance in metres (default: original 0.20)')
    parser.add_argument('--task', default='Mjlab-JengaHead-Vertical3-MicroDuck',
                        choices=['Mjlab-JengaHead-Vertical3-MicroDuck',
                                 'Mjlab-JengaHead-Vertical2-MicroDuck',
                                 'Mjlab-JengaHead-Pedestal-MicroDuck'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    envs = args.envs or (32 if args.phase == 'smoke' else 64)
    iterations = args.iterations or (5 if args.phase == 'smoke' else 2000)
    if not 1 <= envs <= 128 or iterations < 1 or not 1 <= args.seconds <= 86400:
        parser.error('Use 1-128 environments and an explicit 1-86400 second bound')
    if not 0.14 <= args.tower_offset <= 0.20:
        parser.error('Experimental tower offset must be between 0.14 and 0.20 metres')
    if args.phase == 'smoke' and (envs not in range(32, 65) or iterations != 5):
        parser.error('Smoke requires 32-64 environments and exactly five iterations')
    checkpoint = (args.checkpoint or ROOT/'pretrained/velstand_model_2749.pt').resolve()
    if not checkpoint.is_file():
        parser.error('Restore a provenance-checked balance or resume checkpoint first')
    if args.resume and args.checkpoint is None:
        parser.error('--resume requires an explicit same-task checkpoint')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Select a single local GPU with CUDA_VISIBLE_DEVICES')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_name = f'local_{args.phase}_{stamp}'
    command = [sys.executable, str(Path(sys.executable).parent/'train'), args.task,
               '--env.scene.num-envs', str(envs), '--agent.max-iterations', str(iterations),
               '--agent.save-interval', '20' if args.phase == 'train' else '1',
               '--agent.seed', '42', '--agent.logger', 'tensorboard',
               '--agent.upload-model', 'False', '--agent.run-name', run_name,
               '--enable-nan-guard', 'True', '--init-checkpoint', str(checkpoint)]
    if args.tower_offset != 0.20:
        command += ['--env.events.reset-jenga.params.offset-x', str(args.tower_offset)]
    child_env = os.environ.copy()
    child_env.update(PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2',
                     MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled',
                     PYTHONUNBUFFERED='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    if args.resume:
        child_env.pop('JENGA_ACTOR_ONLY_INIT', None)
        child_env.pop('MICRODUCK_WARM_START', None)
    else:
        child_env.update(JENGA_ACTOR_ONLY_INIT='1', MICRODUCK_WARM_START='1')
    info = dict(task=args.task, phase=args.phase, envs=envs, iterations=iterations,
                run_name=run_name, python=sys.version, cuda=torch.version.cuda,
                gpu=props.name, capability=list(torch.cuda.get_device_capability()),
                gpu_total_bytes=props.total_memory, resume=args.resume, tower_offset_m=args.tower_offset,
                lock_sha256=hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                versions={p:metadata.version(p) for p in
                          ['torch','mujoco','mujoco-warp','warp-lang','mjlab','rsl-rl-lib']},
                command=command, status='running', wall_time_cap_s=args.seconds)
    write_json(args.output/'run.json', info)
    started = time.monotonic()
    timed_out = False
    with (args.output/'train.log').open('w') as log, (args.output/'hardware.csv').open('w', newline='') as hw:
        writer = csv.writer(hw)
        writer.writerow(['elapsed_s','gpu_util_pct','gpu_memory_used_mib','gpu_temperature_c','gpu_power_w',
                         'system_load_1m','system_ram_available_kib','training_rss_kib'])
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        info['pid'] = process.pid
        write_json(args.output/'run.json', info)
        print(f'Running {run_name} pid={process.pid}; evidence: {args.output}', flush=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= args.seconds:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
                    break
                sample = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                if sample.returncode == 0:
                    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                    try:
                        proc_status = dict(line.split(':', 1) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines())
                        rss = int(proc_status.get('VmRSS', '0 kB').split()[0])
                    except FileNotFoundError:
                        rss = 0
                    writer.writerow([round(elapsed, 2), *next(csv.reader([sample.stdout.strip()])),
                                     os.getloadavg()[0], int(memory['MemAvailable'].split()[0]), rss])
                    hw.flush()
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            info.update(status='timeout' if timed_out else ('completed' if process.returncode == 0 else 'failed'),
                        returncode=process.returncode, elapsed_s=round(time.monotonic()-started, 3))
            info['checkpoints'] = [str(p.relative_to(ROOT)) for p in (ROOT/'logs').rglob('model_*.pt')
                                   if run_name in str(p)]
            write_json(args.output/'run.json', info)
    print(json.dumps({k:info[k] for k in ['status','returncode','elapsed_s','checkpoints']}), flush=True)
    return process.returncode if not timed_out else 124


if __name__ == '__main__':
    sys.exit(main())
