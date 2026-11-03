"""CPU-only regression checks: mechanics, contracts, and false success gates."""
from types import SimpleNamespace as NS
import math
import torch
import mujoco

from mjlab.scene import Scene
from mjlab_microduck.tasks import mdp
from mjlab_microduck.jenga_strike import initial_strike_state
from mjlab_microduck.tasks.microduck_jenga_env_cfg import (
    LAYOUTS, make_microduck_jenga_env_cfg, make_jenga_rl_cfg, block_spec)


def test_layouts_compile_and_preserve_original_task():
    from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import make_microduck_ball_kick_env_cfg
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    for layout, sizes in LAYOUTS.items():
        cfg = make_microduck_jenga_env_cfg(layout=layout)
        cfg.scene.num_envs = 1
        scene = Scene(cfg.scene, device="cpu")
        model = scene.compile()
        assert model.nu == 14
        assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1 + len(sizes)
        assert "reset_jenga" in cfg.events and "reset_ball" not in cfg.events
        assert list(cfg.events).index("reset_jenga") > list(cfg.events).index("set_ground_state")
        actor = cfg.observations["actor"].terms
        assert list(actor) == list(make_microduck_ball_kick_env_cfg().observations["actor"].terms)
        assert not any("jenga" in k or "ball" in k for k in actor)
        assert "jenga_state" in cfg.observations["critic"].terms
        assert "push_robot" not in cfg.events and cfg.metrics["jenga_success"].reduce == "last"
        assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        assert (act.delay_min_lag,act.delay_max_lag) == (6,12)
        assert make_jenga_rl_cfg(layout).algorithm.symmetry_cfg is None
        assert cfg.rewards["jenga_base_motion"].weight < 0
    assert "ball" in make_microduck_ball_kick_env_cfg().scene.entities
    assert "ball" not in make_microduck_velocity_env_cfg().scene.entities


def test_towers_stand_without_robot_or_reward():
    for layout, sizes in LAYOUTS.items():
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.0025
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1,1,0.1])
        height = 0
        for i, size in enumerate(sizes):
            child = block_spec(size, i == len(sizes)-1)
            height += size[2]
            child.body("block").pos = [0,0,height]
            spec.attach(child, prefix=f"b{i}/", frame=spec.worldbody.add_frame())
            height += size[2]
        model = spec.compile()
        data = mujoco.MjData(model)
        initial = data.qpos.copy()
        for _ in range(1200):  # three simulated seconds, tiny prop-only CPU model
            mujoco.mj_step(model, data)
        assert abs(data.qpos-initial).max() < 0.003, layout
        assert all(abs(data.qpos[7*i+3]) > math.cos(math.radians(2)/2) for i in range(len(sizes)))


class Prop:
    def __init__(self, pos):
        self.data = NS(root_link_pos_w=torch.tensor([pos], dtype=torch.float),
            root_link_quat_w=torch.tensor([[1.,0,0,0]]),
            root_link_lin_vel_w=torch.zeros(1,3), root_link_ang_vel_w=torch.zeros(1,3))


class MockScene(dict):
    pass


def mock_env():
    scene = MockScene(robot=Prop([0,0,0.115]), jenga_0=Prop([0.2,0,0.0375]),
                      jenga_1=Prop([0.2,0,0.1125]))
    scene["robot"].data.site_pos_w = torch.tensor([[[0.1,0,0.12]]])
    scene["robot"].find_sites = lambda name: ([0], [name])
    scene.terrain = NS(env_origins=torch.zeros(1,3))
    scene.sensors = {n: NS(data=NS(found=torch.zeros(1,1), force=torch.zeros(1,1,3)))
        for n in ("jenga_head_top", "jenga_other_top", "jenga_robot_base_0", "jenga_top_floor")}
    zeros = torch.zeros(1)
    keys = ("head_contact", "wrong_contact", "base_failure", "robot_fall", "failure",
            "success", "success_awarded", "top_fallen", "base_max_xy_m", "base_max_tilt_rad", "max_head_force_n",
            "settle_s", "approach_previous", "push_frontier", "approach_delta", "push_delta",
            "success_delta", "base_cost", "force_cost")
    state = {k: zeros.clone() for k in keys}
    state.update(initial_strike_state(zeros, torch.zeros(1,3)))
    state.update(names=("jenga_0","jenga_1"), half_sizes=LAYOUTS["Vertical2"],
        spawn=torch.tensor([[[0.2,0,0.0375],[0.2,0,0.1125]]]),
        direction=torch.tensor([[1.,0]]), last_step=-1)
    state["approach_previous"][:] = -1
    return NS(scene=scene, _jenga=state, common_step_counter=0, num_envs=1, device="cpu", step_dt=0.02,
              sim=NS(forward=lambda:None, sense=lambda:None))


def advance(env, count=30):
    for _ in range(count):
        env.common_step_counter += 1
        mdp.jenga_failure(env)


def test_requires_head_contact_and_half_second_settle():
    env = mock_env()
    env.scene["jenga_1"].data.root_link_pos_w[:] = torch.tensor([[0.28,0,0.0125]])
    env.scene.sensors["jenga_top_floor"].data.found[:] = 1
    advance(env)
    assert env._jenga["success"].item() == 0  # spontaneous fall doesn't count
    env = mock_env()
    advance(env,1)  # seed head finite difference before contact
    env.scene["robot"].data.site_pos_w[0,0,0] += 0.006  # 0.30 m/s neck/head motion
    env.scene.sensors["jenga_head_top"].data.found[:] = 1
    advance(env,1)
    env.scene.sensors["jenga_head_top"].data.found[:] = 0
    env.scene["jenga_1"].data.root_link_pos_w[:] = torch.tensor([[0.28,0,0.0125]])
    env.scene.sensors["jenga_top_floor"].data.found[:] = 1
    advance(env, 24)
    assert env._jenga["success"].item() == 0
    advance(env, 2)
    assert env._jenga["success"].item() == 1
    # Step guard prevents repeated manager reads from advancing the timer.
    before = env._jenga["settle_s"].clone()
    for _ in range(5):
        mdp.jenga_metric(env, "success")
    assert torch.equal(before, env._jenga["settle_s"])
    # No repeat jackpot once successful.
    advance(env)
    assert env._jenga["success_delta"].item() == 0


def test_near_floor_without_contact_is_not_success():
    env = mock_env()
    env.scene["jenga_1"].data.root_link_pos_w[:] = torch.tensor([[0.28,0,0.04]])
    env.scene.sensors["jenga_head_top"].data.found[:] = 1
    advance(env)
    assert env._jenga["success"].item() == 0


def test_base_contact_and_foot_contact_disqualify():
    for name in ("jenga_other_top", "jenga_robot_base_0"):
        env = mock_env()
        env.scene.sensors[name].data.found[:] = 1
        advance(env, 1)
        assert env._jenga["wrong_contact"].item() == 1
        assert env._jenga["failure"].item() == 1


def test_base_movement_tilt_robot_fall_and_nan_fail():
    for failure in ("shift", "tilt", "fall", "nan", "robot_nan"):
        env = mock_env()
        if failure == "shift":
            env.scene["jenga_0"].data.root_link_pos_w[0,0] += 0.01
        elif failure == "tilt":
            env.scene["jenga_0"].data.root_link_quat_w[:] = torch.tensor([[math.cos(0.2),math.sin(0.2),0,0]])
        elif failure == "fall":
            env.scene["robot"].data.root_link_pos_w[0,2] = 0.03
        elif failure == "robot_nan":
            env.scene["robot"].data.root_link_lin_vel_w[0,0] = float("nan")
        else:
            env.scene["jenga_1"].data.root_link_pos_w[0,0] = float("nan")
        advance(env, 1)
        assert env._jenga["failure"].item() == 1, failure
        assert env._jenga["success"].item() == 0
        assert torch.isfinite(mdp.jenga_reward(env, "base_cost")).all()


def test_actor_only_warm_start_keeps_new_critic_and_resets_counters():
    import os
    from mjlab_microduck.tasks import JengaOnPolicyRunner, MicroduckOnPolicyRunner
    original = MicroduckOnPolicyRunner.load
    previous = os.environ.get("JENGA_ACTOR_ONLY_INIT")
    calls = []
    def fake_load(self,path,**kwargs):
        calls.append(kwargs)
        self.current_learning_iteration = 800
        self.env.unwrapped.common_step_counter = 19200
        return {"source":"mock"}
    runner = object.__new__(JengaOnPolicyRunner)
    runner.env = NS(unwrapped=NS(common_step_counter=19200))
    MicroduckOnPolicyRunner.load = fake_load
    try:
        os.environ["JENGA_ACTOR_ONLY_INIT"] = "1"
        assert runner.load("mock.pt") == {"source":"mock"}
        assert calls[-1]["load_cfg"] == {"actor":True}
        assert runner.current_learning_iteration == 0
        assert runner.env.unwrapped.common_step_counter == 0
        # Explicit playback selection and ordinary full resume are preserved.
        runner.load("mock.pt",load_cfg={"actor":True})
        assert runner.current_learning_iteration == 800
        del os.environ["JENGA_ACTOR_ONLY_INIT"]
        runner.load("mock.pt")
        assert calls[-1]["load_cfg"] is None
    finally:
        MicroduckOnPolicyRunner.load = original
        if previous is None:
            os.environ.pop("JENGA_ACTOR_ONLY_INIT",None)
        else:
            os.environ["JENGA_ACTOR_ONLY_INIT"] = previous


def test_recording_checkpoint_order_and_intervals():
    import runpy
    import tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root/"scripts/record_jenga.py"))
    with tempfile.TemporaryDirectory() as directory:
        for it in (1000,1999,0,100):
            (Path(directory)/f"model_{it}.pt").touch()
        chosen = module["pick_checkpoints"](Path(directory))
        assert [(label,path.name) for label,path in chosen] == [
            ("early","model_0.pt"),("middle","model_1000.pt"),("late","model_1999.pt")]
    lo,hi = module["wilson"](0,32)
    assert lo == 0 and hi > 0


def test_reset_rotates_whole_tower_and_preserves_other_environments():
    class WritableProp:
        def write_root_link_pose_to_sim(self,pose,ids):
            self.pose = pose.clone()
        def write_root_link_velocity_to_sim(self,velocity,ids):
            self.velocity = velocity.clone()
    scene = MockScene(robot=NS(indexing=NS(free_joint_q_adr=slice(0,7))),
                      jenga_0=WritableProp(),jenga_1=WritableProp())
    scene.terrain = NS(env_origins=torch.tensor([[0.,0,0],[3.,5,0]]))
    roots = torch.tensor([[0.,0,.115,1,0,0,0],
        [3.,5,.115,math.cos(math.pi/4),0,0,math.sin(math.pi/4)]])
    env = NS(scene=scene,num_envs=2,device="cpu",sim=NS(data=NS(qpos=roots)))
    mdp.reset_jenga_tower(env,torch.tensor([1]),("jenga_0","jenga_1"),
                         LAYOUTS["Vertical2"],noise_xy=0)
    assert torch.allclose(scene["jenga_1"].pose[0,:3],torch.tensor([3.,5.2,.1125]),atol=1e-6)
    assert torch.equal(scene["jenga_1"].velocity,torch.zeros(1,6))
    env._jenga["success"][1] = 1
    mdp.reset_jenga_tower(env,torch.tensor([0]),("jenga_0","jenga_1"),
                         LAYOUTS["Vertical2"],noise_xy=0)
    assert env._jenga["success"][1].item() == 1
    assert torch.allclose(scene["jenga_1"].pose[0,:3],torch.tensor([.2,0,.1125]),atol=1e-6)
