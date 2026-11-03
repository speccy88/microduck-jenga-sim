"""Bounded CPU-only balance validation of official learned policies in Jenga scenes.

This does not train, use Warp/GPU, or move a robot. CPU BAM with a physics-rate
delay, proprioception delays/noise and small start noise is an approximation of
the target environment; cloud Warp smoke and full DR validation remain required.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController
from mjlab.scene import Scene
from mjlab_microduck.tasks.microduck_jenga_env_cfg import make_microduck_jenga_env_cfg


def verify_checkpoint(root):
    """Verify original training checkpoint provenance and its deterministic actor."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    source = root/"pretrained"
    bundle = json.loads((source/"velstand_bundle.json").read_text())
    checkpoint = source/"velstand_model_2749.pt"
    model_file = source/"velstand.onnx"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(checkpoint) == bundle["checkpoint_sha256"]
    assert sha(model_file) == bundle["onnx"][0]["sha256"]
    loaded = torch.load(checkpoint,map_location="cpu",weights_only=True)
    state = loaded.get("actor_state_dict")
    if state is None:
        state = {}
        for key,value in loaded["model_state_dict"].items():
            if key.startswith("actor."):
                state[key.replace("actor.","mlp.",1)] = value
            elif key.startswith("actor_obs_normalizer."):
                state[key.replace("actor_obs_normalizer.","obs_normalizer.",1)] = value
            elif key in ("std","log_std"):
                state[key] = value
    state = dict(state)
    for old,new in (("std","distribution.std_param"),("log_std","distribution.log_std_param")):
        if old in state:
            state[new] = state.pop(old)
    actor = MLPModel(TensorDict({"actor":torch.zeros(1,61)},batch_size=[1]),
        {"actor":["actor"]},"actor",14,hidden_dims=[512,256,128],activation="elu",
        obs_normalization=True,distribution_cfg={"class_name":"GaussianDistribution",
        "init_std":1.0,"std_type":"scalar"})
    actor.load_state_dict(state,strict=True)
    actor.eval()
    initializers = {v.name:numpy_helper.to_array(v) for v in onnx.load(model_file).graph.initializer}
    exact = all(np.array_equal(state[name].numpy(),value) for name,value in initializers.items()
                if name in state)
    assert exact
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 1,1
    session = ort.InferenceSession(str(model_file),options,providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261029)
    mean = state["obs_normalizer._mean"].numpy()
    scale = state["obs_normalizer._std"].numpy()
    observations = (mean + rng.normal(size=(256,61))*scale*0.5).astype(np.float32)
    maximum_error = 0.
    for vector in observations:
        with torch.inference_mode():
            actual = actor(TensorDict({"actor":torch.from_numpy(vector[None])},batch_size=[1])).numpy()
        expected = session.run(None,{session.get_inputs()[0].name:vector[None]})[0]
        maximum_error = max(maximum_error,float(np.abs(actual-expected).max()))
    assert maximum_error < 1e-4,maximum_error
    report = {"checkpoint_sha256":sha(checkpoint),"onnx_sha256":sha(model_file),
        "iteration":loaded["iter"],"actor_load_strict":True,"shared_initializers_bit_exact":exact,
        "cpu_comparison_vectors":256,"maximum_action_absolute_error":maximum_error,
        "normalizer_from_original_checkpoint":True}
    (root/"evidence/balance_checkpoint_compatibility.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


def inverse_rotate(quaternion, vector):
    w, xyz = quaternion[0],quaternion[1:]
    cross = 2*np.cross(xyz,vector)
    return vector-w*cross+np.cross(xyz,cross)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",default="velstand.onnx")
    parser.add_argument("--layout",default="Vertical3",choices=["Vertical2","Vertical3","Pedestal"])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--seconds",type=float,default=6.)
    parser.add_argument("--sensor-noise",action="store_true")
    parser.add_argument("--verify-checkpoint",action="store_true")
    parser.add_argument("--video-seed",type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES='' for this CPU-only check")
    if args.seconds <= 0 or args.seconds>6 or len(args.seeds)>3:
        raise SystemExit("Lightweight check limited to three episodes of at most six seconds")
    root = Path(__file__).resolve().parents[1]
    if args.verify_checkpoint:
        verify_checkpoint(root)
    options = ort.SessionOptions()
    options.intra_op_num_threads,options.inter_op_num_threads = 1,1
    policy_file = root/"pretrained"/args.policy
    session = ort.InferenceSession(str(policy_file),options,providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert session.get_inputs()[0].shape == [1,61]
    assert session.get_outputs()[0].shape == [1,14]
    cfg = make_microduck_jenga_env_cfg(layout=args.layout)
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene,device="cpu")
    model = scene.compile()
    model.opt.timestep = cfg.sim.mujoco.timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = cfg.sim.mujoco.iterations
    data = mujoco.MjData(model)
    names = [model.actuator(i).name for i in range(model.nu)]
    expected = metadata["joint_names"].split(",")
    assert [name.removeprefix("robot/") for name in names] == expected
    bam = load_model(motor_name="xl330",model="m6")
    bam.actuator.kp,bam.actuator.vin = 200.,7.4
    bam.actuator.max_current = None
    controller = MujocoController(bam,names,model,data,vin_drop_gain=0.1,vin_min=6.)
    qidx,didx = controller.qpos_indexes,controller.dof_indexes
    home = model.key_qpos[0,qidx].copy()
    trunk = model.body("robot/trunk_base").id
    sensor = model.sensor("robot/imu_ang_vel").id
    sensor_address = model.sensor_adr[sensor]
    body_ids = [model.body(f"jenga_{i}/block").id for i in range(len(cfg.scene.entities)-1)]
    robot_body_ids = {i for i in range(model.nbody) if model.body(i).name.startswith("robot/")}
    report = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(model,data,0)
        data.qpos[qidx] += rng.uniform(-0.02,0.02,14)
        mujoco.mj_forward(model,data)
        controller.reset(data.qpos)
        controller.last_ts = data.time
        action = np.zeros(14)
        target = home.copy()
        lag = int(rng.integers(6,13))
        history = deque([home.copy() for _ in range(13)],maxlen=13)
        prior_velocity = np.zeros(14)
        encoder_bias = rng.uniform(-0.015,0.015,14) if args.sensor_noise else np.zeros(14)
        # Small per-episode IMU mounting rotation, as in the task's noise model.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-math.radians(6),math.radians(6)) if args.sensor_noise else 0.
        misalignment = np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
        spawn = data.xpos[body_ids].copy()
        min_z,max_tilt,peak_base_xy,peak_base_tilt = float("inf"),0.,0.,0.
        wrong_contacts, head_contacts, finite, fall = 0,0,True,False
        frames = []
        renderer = mujoco.Renderer(model,height=240,width=320) if seed == args.video_seed else None
        camera = mujoco.MjvCamera()
        camera.lookat[:],camera.distance,camera.azimuth,camera.elevation = [0.08,0,0.12],.85,135,-18
        if renderer:
            renderer.update_scene(data,camera=camera)
            frames.append(renderer.render())
        try:
            for step in range(round(args.seconds/.02)):
                gravity = inverse_rotate(data.xquat[trunk],np.array([0.,0,-1.]))
                gyro = data.sensordata[sensor_address:sensor_address+3].copy()
                gyro = inverse_rotate(misalignment,gyro)
                gravity = inverse_rotate(misalignment,gravity)
                joint_pos = data.qpos[qidx]-home+encoder_bias
                if args.sensor_noise:
                    gyro += rng.uniform(-.03,.03,3)
                    gravity += rng.uniform(-.01,.01,3)
                    joint_pos += rng.uniform(-.001,.001,14)
                velocity = prior_velocity.copy()
                prior_velocity = data.qvel[didx].copy()
                if args.sensor_noise:
                    velocity += rng.uniform(-.25,.25,14)
                obs = np.r_[gyro,gravity,joint_pos,velocity,action,np.zeros(13)].astype(np.float32)
                action = session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0]
                finite &= bool(np.isfinite(action).all())
                target = home+action
                for _ in range(cfg.decimation):
                    history.append(target.copy())
                    controller.q_target[:] = history[-lag-1]
                    controller.update()
                    mujoco.mj_step(model,data)
                    for contact in data.contact:
                        a,b = map(int,model.geom_bodyid[contact.geom])
                        if a in body_ids and b in robot_body_ids:
                            a,b = b,a
                        if b in body_ids and a in robot_body_ids:
                            is_head = model.body(a).name == "robot/jaw_soft"
                            if b == body_ids[-1] and is_head:
                                head_contacts += 1
                            else:
                                wrong_contacts += 1
                mujoco.mj_forward(model,data)
                q = data.xquat[trunk]
                tilt = math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))
                min_z,max_tilt = min(min_z,float(data.qpos[2])),max(max_tilt,tilt)
                base = data.xpos[body_ids[:-1]]
                peak_base_xy = max(peak_base_xy,float(np.linalg.norm(base[:,:2]-spawn[:-1,:2],axis=1).max()))
                qs = data.xquat[body_ids[:-1]]
                peak_base_tilt = max(peak_base_tilt,float(np.arccos(np.clip(1-2*(qs[:,1]**2+qs[:,2]**2),-1,1)).max()))
                finite &= bool(np.isfinite(data.qpos).all())
                fall |= bool(tilt>math.radians(55) or data.qpos[2]<.055)
                # Optional local preview is capped at 30 frames (5 fps / six s).
                # Full 50 fps checkpoint videos belong on the approved cloud GPU.
                if renderer and step%10 == 9:
                    renderer.update_scene(data,camera=camera)
                    frames.append(renderer.render())
                if fall or not finite:
                    break
        finally:
            if renderer:
                renderer.close()
        row = {"seed":seed,"duration_s":round(data.time,6),"lag_physics_steps":lag,
            "sensor_noise":args.sensor_noise,"finite":finite,"robot_fall":fall,
            "max_tilt_deg":math.degrees(max_tilt),"minimum_trunk_z_m":min_z,
            "final_trunk_z_m":float(data.qpos[2]),"base_peak_xy_m":peak_base_xy,
            "base_peak_tilt_deg":math.degrees(peak_base_tilt),
            "head_contact_substeps":head_contacts,"wrong_contact_substeps":wrong_contacts,
            "passed":bool(finite and not fall and data.time>=args.seconds-.01
                          and peak_base_xy<.008 and peak_base_tilt<math.radians(12))}
        if frames:
            import imageio.v2 as imageio
            video = root/"evidence"/f"balance_baseline_{args.layout.lower()}_seed_{seed}.mp4"
            imageio.mimsave(video,frames,fps=5,codec="libx264",quality=7,macro_block_size=1)
            row["video"] = str(video.relative_to(root))
        report.append(row)
        print(json.dumps(row),flush=True)
    suffix = "noisy" if args.sensor_noise else "nominal"
    output = root/"evidence"/f"balance_{Path(args.policy).stem}_{args.layout.lower()}_{suffix}.json"
    output.write_text(json.dumps({"policy":args.policy,"layout":args.layout,
        "backend":"CPU MuJoCo + BAM + ONNX Runtime CPU; no training",
        "limitations":"Approximate sensor/delay path; no full DR or Warp validation",
        "episodes":report},indent=2)+"\n")
    raise SystemExit(0 if all(r["passed"] for r in report) else 1)


if __name__ == "__main__":
    main()
