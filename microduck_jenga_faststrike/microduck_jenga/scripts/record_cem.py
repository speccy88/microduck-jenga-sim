"""Record full independent-seed CEM success/failure replays, without skipping frames."""
import argparse, hashlib, json
from pathlib import Path
import imageio.v2 as imageio
import torch
from train_strike_search import build, episode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[61000,61001])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    ckpt=json.loads(args.checkpoint.read_text())
    torch.set_num_threads(2)
    for seed in args.seeds:
        env,wrapped,runner,policy=build(1,render=True,highrate=True,tower_offset=ckpt['tower_offset_m'])
        path=args.output/f'cem_generation100_seed_{seed}.mp4'
        try:
            with imageio.get_writer(path,fps=50,codec='libx264',quality=8) as writer:
                _,rows,trace=episode(env,wrapped,policy,ckpt['parameters'],seed,writer,
                                     closed_loop=ckpt.get('closed_loop',False),
                                     approach_home=ckpt.get('approach_home',False),
                                     stationary=ckpt.get('stationary',False))
        finally: env.close()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        metadata={'seed':seed,'sha256':digest,'episode':rows[0],'frames':len(trace)+1,
                  'fps':50,'controller':'CEM-optimized timed head trajectory over frozen VelStand; ideal simulator bearing',
                  'layout':'two blocks, 14 cm','trace':trace}
        (args.output/f'{path.stem}.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in metadata.items() if k!='trace'}),flush=True)

if __name__=='__main__': main()
