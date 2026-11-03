"""Replay the archived CEM controller with independent seeds, after the GPU is free.

Run from the old experiment's work directory with its reviewed_snapshot/src in
PYTHONPATH and PYTHONDONTWRITEBYTECODE=1. This only reads the old workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from train_strike_search import build, episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-seed', type=int, default=61000)
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--resume', action='store_true', help='Continue an incomplete results file without repeating seeds')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    ckpt = json.loads(args.checkpoint.read_text())
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.resume:
        output=json.loads((args.output/'results.json').read_text())
        seeds=[row['seed'] for row in output['episodes']]
        if output['complete'] or output['checkpoint_sha256'] != checkpoint_hash or seeds != list(range(args.first_seed,args.first_seed+len(seeds))) or len(seeds)>args.count:
            raise ValueError('Existing results do not match this incomplete, contiguous evaluation')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        output = dict(checkpoint_sha256=checkpoint_hash,
                  generation=ckpt['generation'], tower_offset_m=ckpt['tower_offset_m'],
                  closed_loop=ckpt.get('closed_loop', False), stationary=ckpt.get('stationary', False),
                  contact_sampling_hz=400, episodes=[], complete=False,
                  limitation='Historical high-rate monitor requires independent freshness verification.')
    def save():
        temp = args.output/'results.tmp'
        temp.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        temp.replace(args.output/'results.json')
    save()
    torch.set_num_threads(2)
    env, wrapped, runner, policy = build(1, highrate=True, tower_offset=output['tower_offset_m'])
    try:
        for seed in range(args.first_seed+len(output['episodes']), args.first_seed+args.count):
            _, rows, _ = episode(env, wrapped, policy, ckpt['parameters'], seed,
                                 closed_loop=ckpt.get('closed_loop', False),
                                 approach_home=ckpt.get('approach_home', False),
                                 stationary=ckpt.get('stationary', False))
            output['episodes'].append({'seed':seed, **rows[0]})
            save()
            print(f"seed={seed} success={rows[0]['success']}", flush=True)
    finally:
        env.close()
    output['complete'] = True
    output['successes'] = sum(bool(row['success']) for row in output['episodes'])
    save()


if __name__ == '__main__':
    main()
