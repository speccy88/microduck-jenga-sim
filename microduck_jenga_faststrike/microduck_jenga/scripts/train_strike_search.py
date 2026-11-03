"""Local CEM training of a head-strike controller over frozen VelStand.

Simulation only. Original free-block geometry, dynamics and strict success
metrics are unchanged. No camera/ToF input. Optional closed-loop mode uses
ideal simulator geometry for the approach trigger. Eight learned parameters
drive actuator targets and a velocity command, never robot/block poses.
"""
import argparse, json, os, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
import mjlab_microduck.tasks
from mjlab_microduck.jenga_evaluation import METRICS
TRACKED=METRICS+('strike_peak_head_forward_m_s','strike_first_contact_s','strike_head_total_speed_m_s')

TASK='Mjlab-JengaHead-Vertical2-MicroDuck'
ROOT=Path(__file__).resolve().parents[1]
NAMES=['forward_speed','approach_duration','pre_neck','pre_head','strike_neck','strike_head','strike_duration','hold_duration']
LOW=np.array([.015,.3,-1.1,-.9,-1.88,-1.7,.06,.02])
HIGH=np.array([.16,2.2,.1,.9,-.6,.7,.35,.24])

def build(n,render=False,highrate=False,tower_offset=None):
 cfg=load_env_cfg(TASK,play=True); cfg.scene.num_envs=n; cfg.seed=41; cfg.auto_reset=False
 if tower_offset is not None:
  cfg.events['reset_jenga'].params['offset_x']=tower_offset
  for name,entity in cfg.scene.entities.items():
   if name.startswith('jenga_'):
    _,y,z=entity.init_state.pos; entity.init_state.pos=(tower_offset,y,z)
 if highrate:
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from strike_highrate import monitor
  cfg.metrics['physics_strike_contact']=MetricsTermCfg(func=monitor,per_substep=True)
 env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array' if render else None)
 agent=load_rl_cfg(TASK); wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 runner=load_runner_cls(TASK)(wrapped,asdict(agent),device='cuda:0')
 runner.load(str(ROOT/'pretrained/velstand_model_2749.pt'),load_cfg={'actor':True},strict=True,map_location='cuda:0')
 return env,wrapped,runner,runner.get_inference_policy(device='cuda:0')

def episode(env,wrapped,policy,parameters,seed,writer=None,closed_loop=False,approach_home=False,stationary=False):
 env.reset(seed=seed); obs=wrapped.get_observations(); n=env.num_envs
 p=torch.as_tensor(parameters,dtype=torch.float32,device=env.device)
 if p.ndim==1: p=p.repeat(n,1)
 ids=env.scene['robot'].find_joints(['neck_pitch','head_pitch'],preserve_order=True)[0]
 yaw_id=env.scene['robot'].find_joints('head_yaw')[0][0]
 alive=torch.ones(n,dtype=torch.bool,device=env.device)
 min_dist=torch.full((n,),1.,device=env.device); max_move=torch.zeros(n,device=env.device)
 rows={k:torch.zeros(n,device=env.device) for k in TRACKED}; rows['duration_s']=torch.zeros(n,device=env.device)
 trace=[]
 strike_start=torch.full((n,),float('inf'),device=env.device)
 if writer:
  from sys import path
  path.insert(0,str(ROOT/'scripts'))
  from record_jenga import center_episode_camera
  center_episode_camera(env)
  q=env.scene['robot'].data.root_link_quat_w[0].detach().cpu().numpy()
  yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
  env._offline_renderer._cam.azimuth=float(np.degrees(yaw)+135)
  writer.append_data(env.render())
 for step in range(env.max_episode_length):
  t=step*env.step_dt
  start=.5+p[:,1]; dur=p[:,6]; hold=p[:,7]
  if closed_loop:
   relative=env._jenga['spawn'][:,-1,:2]-env.scene['robot'].data.root_link_pos_w[:,:2]
   gap=(relative*env._jenga['direction']).sum(-1)
   ready=((gap<=p[:,1]) & (t>=.5)) | (t>=3.5)
   strike_start=torch.where(ready & torch.isinf(strike_start),torch.full_like(strike_start,t+(.65 if approach_home else .15)),strike_start)
   start=strike_start
  if stationary: start=torch.full_like(start,1.2)
  pre=p[:,2:4]; strike=p[:,4:6]
  ramp=min(1.,t/.5)
  target=pre*ramp
  if closed_loop and approach_home:
   ramp=((t-(start-.65))/.5).clamp(0,1)
   target=pre*ramp[:,None]
  blend=((t-start)/dur).clamp(0,1)
  target=target+(strike-pre)*blend[:,None]
  withdraw=((t-start-dur-hold)/.45).clamp(0,1)
  target=target*(1-withdraw[:,None])
  obs['actor'][:,48:]=0
  approaching=(t>=.4)&(t<start)
  if closed_loop: approaching=(t>=.4)&torch.isinf(strike_start)
  if stationary: approaching=torch.zeros_like(approaching)
  obs['actor'][:,48]=torch.where(approaching,p[:,0],0.)
  obs['actor'][:,51:53]=target
  if closed_loop:
   q=env.scene['robot'].data.root_link_quat_w
   yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
   desired=torch.atan2(relative[:,1],relative[:,0])
   yaw_error=torch.atan2(torch.sin(desired-yaw),torch.cos(desired-yaw))
   obs['actor'][:,50]=(2*yaw_error).clamp(-.4,.4)
   obs['actor'][:,53]=yaw_error.clamp(-.45,.45)
  with torch.inference_mode(): action=policy(obs)
  action=action.clone()
  action[:,ids]=target
  if closed_loop: action[:,yaw_id]=yaw_error.clamp(-.45,.45)
  obs,rew,done,extras=wrapped.step(action)
  s=env._jenga
  if step==0 and getattr(env,'_jenga_highrate_enabled',False):
   assert bool((s['strike_samples']==env.cfg.decimation).all()), 'Strike monitor did not run exactly once per physics step'
   assert bool(torch.allclose(s['strike_elapsed_s'],torch.full_like(s['strike_elapsed_s'],env.step_dt),atol=1e-6)), 'Strike monitor timebase mismatch'
  mouth=env.scene['robot'].data.site_pos_w[:,env.scene['robot'].find_sites('mouth_tip')[0][0]]
  aim=s['spawn'][:,-1].clone(); aim[:,:2]-=s['direction']*.0175
  distance=(mouth-aim).norm(dim=-1)
  min_dist=torch.where(alive,torch.minimum(min_dist,distance),min_dist)
  move=((env.scene['jenga_1'].data.root_link_pos_w[:,:2]-s['spawn'][:,-1,:2])*s['direction']).sum(-1).clamp(0,.10)
  max_move=torch.where(alive,torch.maximum(max_move,move),max_move)
  for key in TRACKED: rows[key]=torch.where(alive,s.get(key,torch.zeros_like(rows[key])),rows[key])
  rows['duration_s']=torch.where(alive,torch.full_like(rows['duration_s'],(step+1)*env.step_dt),rows['duration_s'])
  if writer:
   writer.append_data(env.render()); trace.append({'time_s':(step+1)*env.step_dt,**{k:float(v[0]) for k,v in rows.items()},'mouth_world_m':mouth[0].tolist(),'top_world_m':env.scene['jenga_1'].data.root_link_pos_w[0].tolist(),'robot_world_m':env.scene['robot'].data.root_link_pos_w[0].tolist()})
  alive &= ~done.bool()
  if not alive.any(): break
  # Batch members finish independently. Preserve their first-episode metrics;
  # reset only to satisfy the vector simulator while remaining members finish.
  finished=done.nonzero().flatten()
  if finished.numel():
   env.reset(env_ids=finished)
   obs=wrapped.get_observations()
 fail=torch.maximum(rows['robot_fall'],torch.maximum(rows['base_failure'],rows['wrong_contact']))
 # Dense geometric approach helps discover contact. Actual task success is
 # measured separately using the original fast-strike and stability criteria.
 score=30*torch.exp(-min_dist/.05)+25*rows['head_contact']+100*max_move+40*rows['fast_strike']+1000*rows['success']-60*fail
 score+=(50*(rows['strike_head_speed_m_s']/.25).clamp(0,1)*rows['head_contact']+20*rows['top_fallen'])*(1-fail)
 return score.cpu().numpy(),[{**{k:float(v[i]) for k,v in rows.items()},'minimum_mouth_target_distance_m':float(min_dist[i]),'max_top_forward_m':float(max_move[i])} for i in range(n)],trace

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--generations',type=int,default=20); ap.add_argument('--envs',type=int,default=32); ap.add_argument('--resume',type=Path); ap.add_argument('--generation-offset',type=int,default=0); ap.add_argument('--repeats',type=int,default=1); ap.add_argument('--closed-loop',action='store_true'); ap.add_argument('--highrate',action='store_true'); ap.add_argument('--approach-home',action='store_true'); ap.add_argument('--stationary',action='store_true'); ap.add_argument('--tower-offset',type=float); args=ap.parse_args()
 assert args.envs%args.repeats==0
 candidates=args.envs//args.repeats
 torch.set_num_threads(2); args.output.mkdir(parents=True,exist_ok=False)
 env,wrapped,runner,policy=build(args.envs,highrate=args.highrate,tower_offset=args.tower_offset)
 rng=np.random.default_rng(20261029); mean=np.array([.075,1.1,-.45,0,-1.6,-.45,.15,.08]); std=(HIGH-LOW)*.26
 if args.resume:
  old=json.loads(args.resume.read_text()); mean=np.array(old['parameters']); std=(HIGH-LOW)*.13
 if args.closed_loop:
  NAMES[1]='ideal_geometry_trigger_distance_m'; LOW[1]=.095; HIGH[1]=.16
  mean[1]=.135; std[1]=.016
  if args.resume and old.get('closed_loop'): mean[1]=old['parameters'][1]
 history=[]; start=time.monotonic(); best=-1e9; best_params=mean.copy()
 try:
  for generation in range(args.generation_offset,args.generation_offset+args.generations):
   population=np.clip(rng.normal(mean,std,(candidates,len(mean))),LOW,HIGH); population[0]=mean; population[1]=best_params
   individual_score,individual_rows,_=episode(env,wrapped,policy,np.repeat(population,args.repeats,axis=0),100+generation,closed_loop=args.closed_loop,approach_home=args.approach_home,stationary=args.stationary)
   if not np.isfinite(individual_score).all(): raise RuntimeError('Nonfinite candidate score; stopping training')
   score=individual_score.reshape(candidates,args.repeats).mean(1)
   rows=[{k:float(np.mean([r[k] for r in individual_rows[i*args.repeats:(i+1)*args.repeats]])) for k in individual_rows[0]} for i in range(candidates)]
   elite=np.argsort(score)[-max(3,candidates//5):]
   mean=.25*mean+.75*population[elite].mean(0); std=np.maximum(.25*std+.75*population[elite].std(0),(HIGH-LOW)*.04)
   winner=int(np.argmax(score))
   if score[winner]>best: best=float(score[winner]); best_params=population[winner].copy()
   record={'generation':generation,'parameter_names':NAMES,'parameters':population[winner].tolist(),'score':float(score[winner]),'metrics':rows[winner],'mean':mean.tolist(),'std':std.tolist(),'seed':100+generation,'all_scores':score.tolist(),'all_metrics':rows,'population':population.tolist(),'individual_metrics':individual_rows,'repeats':args.repeats,'elapsed_s':time.monotonic()-start}
   record['closed_loop']=args.closed_loop
   record['contact_sampling_hz']=400 if args.highrate else 50
   record['approach_home']=args.approach_home
   record['stationary']=args.stationary
   record['tower_offset_m']=args.tower_offset if args.tower_offset is not None else .20
   (args.output/f'checkpoint_{generation:03}.json').write_text(json.dumps(record,indent=2))
   history.append(record)
   print(json.dumps({k:record[k] for k in ['generation','score','metrics','elapsed_s']}),flush=True)
  (args.output/'training_summary.json').write_text(json.dumps({'algorithm':'cross-entropy trajectory search over frozen learned balance controller','parameters':NAMES,'generations':args.generations,'parallel_envs':args.envs,'repeats_per_candidate':args.repeats,'best_training_score':best,'best_training_parameters':best_params.tolist(),'sensor_input':'proprioception and ideal simulator geometry; no camera/ToF' if args.closed_loop else 'proprioception only; timed outer controller','success_definition':'unchanged original strict fast head strike, landed top, stable lower block, no robot fall/wrong contact','training_seeds':list(range(100+args.generation_offset,100+args.generation_offset+args.generations))},indent=2))
 finally: env.close()

if __name__=='__main__': main()
