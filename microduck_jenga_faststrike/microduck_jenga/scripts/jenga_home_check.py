"""Three-second, CPU BAM equilibrium check, not a learned controller test.

No DR, action delay, encoder noise or policy is exercised by this check. It
only determines whether the initial pose is plausible under nominal BAM.
"""
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController
from mjlab.scene import Scene
from mjlab_microduck.tasks.microduck_jenga_env_cfg import make_microduck_jenga_env_cfg

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES='' for CPU validation")
cfg = make_microduck_jenga_env_cfg()
cfg.scene.num_envs = 1
scene = Scene(cfg.scene,device="cpu")
model = scene.compile()
model.opt.timestep = cfg.sim.mujoco.timestep
data = mujoco.MjData(model)
bam = load_model(motor_name="xl330",model="m6")
bam.actuator.kp, bam.actuator.vin = 200., 7.4
bam.actuator.max_current = None
names = [model.actuator(i).name for i in range(model.nu)]
controller = MujocoController(bam,names,model,data,vin_drop_gain=0.1,vin_min=6.0)
results = []
for seed in (42,43,44):
    rng = np.random.default_rng(seed)
    mujoco.mj_resetDataKeyframe(model,data,0)
    data.qpos[controller.qpos_indexes] += rng.uniform(-0.02,0.02,len(names))
    mujoco.mj_forward(model,data)
    controller.reset(data.qpos)
    controller.q_target = model.key_qpos[0,controller.qpos_indexes].copy()
    controller.last_ts = data.time
    max_tilt, minimum_z = 0., float("inf")
    for _ in range(round(3/model.opt.timestep)):
        controller.update()
        mujoco.mj_step(model,data)
        q = data.qpos[3:7]
        tilt = math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))
        max_tilt = max(max_tilt,tilt)
        minimum_z = min(minimum_z,float(data.qpos[2]))
    final_tilt = math.degrees(tilt)
    results.append({"seed":seed,"max_tilt_deg":math.degrees(max_tilt),
        "final_tilt_deg":final_tilt,"minimum_trunk_z_m":minimum_z,
        "final_trunk_z_m":float(data.qpos[2]),
        "passed":bool(max_tilt < math.radians(20) and minimum_z>0.085
                      and np.isfinite(data.qpos).all())})
root = Path(__file__).resolve().parents[1]
(root/"evidence/home_equilibrium.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
raise SystemExit(0 if all(r["passed"] for r in results) else 1)
