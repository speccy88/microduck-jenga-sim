"""400 Hz contact/strike instrumentation with an explicit dynamics refresh.

No integration, state teleportation, or external force is added. The original
thresholds are used at physics_dt rather than control_dt. Remaining episode
success/settling logic runs at 50 Hz as before.
"""
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab_microduck.tasks import mdp
from mjlab_microduck.jenga_strike import update_strike

def monitor(env):
 env._jenga_highrate_enabled=True
 if not hasattr(env,'_jenga'): return torch.zeros(env.num_envs,device=env.device)
 env.sim.forward()
 env.sim.sense()
 # Sensor/entity caches already advance through scene.update before this hook.
 mdp.jenga_sample_contacts(env)
 s=env._jenga; robot=env.scene['robot']; q=robot.data.root_link_quat_w
 rot=matrix_from_quat(q).transpose(1,2)
 mouth=robot.data.site_pos_w[:,robot.find_sites('mouth_tip')[0][0]]
 mouth_b=torch.bmm(rot,(mouth-robot.data.root_link_pos_w).unsqueeze(-1)).squeeze(-1)
 direction=torch.cat((s['direction'],torch.zeros_like(s['direction'][:,:1])),-1)
 direction_b=torch.bmm(rot,direction.unsqueeze(-1)).squeeze(-1)
 contact=mdp._sensor_any_contact(env,'jenga_head_top')
 top=env.scene['jenga_1'].data.root_link_pos_w
 # Diagnostic alongside the original forward-only criterion: a fast downward
 # nod can have high head speed but small forward speed. Do not overwrite the
 # original success or forward-speed fields.
 if 'strike_head_total_speed_m_s' not in s:
  s['strike_head_total_speed_m_s']=torch.zeros(env.num_envs,device=env.device)
  s['strike_previous_total_speed_m_s']=torch.zeros(env.num_envs,device=env.device)
 total=(mouth_b-s['strike_previous_mouth_b']).norm(dim=-1)/env.physics_dt
 total=torch.where(s['strike_samples']>0,total,0.)
 first=contact & (s['strike_seen_contact']==0)
 s['strike_head_total_speed_m_s']=torch.where(first,torch.maximum(total,s['strike_previous_total_speed_m_s']),s['strike_head_total_speed_m_s'])
 s['strike_previous_total_speed_m_s']=total
 s.update(update_strike(s,mouth_b,direction_b,top-s['spawn'][:,-1],contact,env.physics_dt,torch))
 return contact.float()
