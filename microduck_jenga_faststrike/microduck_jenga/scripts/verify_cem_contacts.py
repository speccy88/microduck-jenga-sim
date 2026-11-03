"""Compare a historical CEM run's Warp contacts with CPU MuJoCo at matching qpos.

Uses an identical model and copies qpos/qvel after selected physics substeps.
This is simulation instrumentation verification, not physical validation.
Run only after the ongoing GPU evaluation has finished.
"""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco
import torch
import strike_highrate
from train_strike_search import build, episode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=61000)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('Output already exists')
    checkpoint=json.loads(args.checkpoint.read_text())
    original=strike_highrate.monitor
    comparison={'seed':args.seed,'checked_substeps':0,'gpu_contacts':0,'cpu_contacts':0,
                'mismatches':[],'max_mouth_position_error_m':0.0,
                'window_s':[1.2,1.7],
                'caveat':'CPU MuJoCo versus Warp at matching simulated qpos, one episode; not hardware truth.'}
    holder={}
    def checked(env):
        result=original(env)
        t=env._sim_step_counter*env.physics_dt
        if not 1.2 <= t <= 1.7: return result
        model=env.sim.mj_model
        if not holder:
            holder['data']=mujoco.MjData(model)
            holder['head']=model.body('robot/jaw_soft').id
            holder['top']=model.body('jenga_1/block').id
            holder['mouth']=model.site('robot/mouth_tip').id
        data=holder['data']
        data.qpos[:]=env.sim.data.qpos[0].detach().cpu().numpy()
        data.qvel[:]=env.sim.data.qvel[0].detach().cpu().numpy()
        mujoco.mj_forward(model,data)
        cpu=any(set(map(int,model.geom_bodyid[c.geom])) == {holder['head'],holder['top']} for c in data.contact)
        gpu=bool(result[0].item())
        gpu_mouth=env.scene['robot'].data.site_pos_w[0,env.scene['robot'].find_sites('mouth_tip')[0][0]].detach().cpu().numpy()
        error=float(np.linalg.norm(data.site_xpos[holder['mouth']]-gpu_mouth))
        comparison['checked_substeps']+=1
        comparison['gpu_contacts']+=int(gpu); comparison['cpu_contacts']+=int(cpu)
        comparison['max_mouth_position_error_m']=max(comparison['max_mouth_position_error_m'],error)
        if cpu != gpu: comparison['mismatches'].append({'time_s':float(t),'cpu':cpu,'gpu':gpu})
        return result
    strike_highrate.monitor=checked
    torch.set_num_threads(2)
    env,wrapped,runner,policy=build(1,highrate=True,tower_offset=checkpoint['tower_offset_m'])
    try:
        _,rows,_=episode(env,wrapped,policy,checkpoint['parameters'],args.seed,
                         closed_loop=checkpoint.get('closed_loop',False),
                         approach_home=checkpoint.get('approach_home',False),
                         stationary=checkpoint.get('stationary',False))
        comparison['episode']=rows[0]
    finally: env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__': main()
