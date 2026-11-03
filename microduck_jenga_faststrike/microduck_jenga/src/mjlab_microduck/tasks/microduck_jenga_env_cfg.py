"""Placement-controlled Jenga prototype. 61D actor; privileged critic/rewards.

This is deliberately not a ToF policy. See docs/jenga.md for the sensor gap.
All props have free joints: the base is never welded, made heavier, or pinned.
"""
from copy import deepcopy
from functools import partial
import math

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from . import mdp
from .microduck_ball_kick_env_cfg import make_microduck_ball_kick_env_cfg, MicroduckBallKickRlCfg

EPISODE_LENGTH_S = 6.0
BLOCK_MASS_KG = 0.012  # provisional; measure the operator's blocks before transfer
BLOCK_FRICTION = 0.45  # provisional wood contact, not a fitted material model
TOWER_OFFSET_X = 0.20
PLACEMENT_NOISE_M = 0.005
LAYOUTS = {
    "Vertical2": ((0.0125, 0.0075, 0.0375),) * 2,
    "Vertical3": ((0.0125, 0.0075, 0.0375),) * 3,
    "Pedestal": ((0.0375, 0.0125, 0.0075),) * 2 + ((0.0125, 0.0075, 0.0375),),
}
HEAD_BODY = "jaw_soft"  # includes head shells and jaw on pinned groundcontact model


def block_spec(half_size, is_top=False):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="block")
    body.add_freejoint(name="block_free")
    body.add_geom(name="block_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=half_size, mass=BLOCK_MASS_KG, condim=3,
                  friction=[BLOCK_FRICTION, 0.002, 0.0001],
                  solref=[0.006, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
                  rgba=[0.95, 0.40, 0.09, 1] if is_top else [0.69, 0.48, 0.25, 1])
    return spec


def _contact(name, primary, entity):
    return ContactSensorCfg(name=name, primary=primary,
        secondary=ContactMatch(mode="body", pattern="block", entity=entity),
        fields=("found", "force"), reduce="netforce", num_slots=1,
        secondary_policy="first")


def make_microduck_jenga_env_cfg(play=False, layout="Vertical3"):
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}")
    cfg = deepcopy(make_microduck_ball_kick_env_cfg(play=play))
    sizes = LAYOUTS[layout]
    names = tuple(f"jenga_{i}" for i in range(len(sizes)))
    cfg.scene.entities.pop("ball")
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.115)
    z = 0.0
    for i, (name, half_size) in enumerate(zip(names, sizes)):
        z += half_size[2]
        cfg.scene.entities[name] = EntityCfg(
            spec_fn=partial(block_spec, half_size, i == len(names) - 1),
            init_state=EntityCfg.InitialStateCfg(pos=(TOWER_OFFSET_X, 0, z)))
        z += half_size[2]
    sensors = [s for s in cfg.scene.sensors if s.name != "support_foot_ground_contact"]
    sensors += [_contact("jenga_head_top", ContactMatch(
        mode="body", pattern=HEAD_BODY, entity="robot"), names[-1])]
    sensors += [_contact("jenga_other_top", ContactMatch(
        mode="body", pattern=".*", entity="robot", exclude=(HEAD_BODY,)), names[-1])]
    sensors += [ContactSensorCfg(name="jenga_top_floor",
        primary=ContactMatch(mode="body",pattern="block",entity=names[-1]),
        secondary=ContactMatch(mode="body",pattern="terrain"),
        fields=("found",),reduce="netforce",num_slots=1)]
    sensors += [_contact(f"jenga_robot_base_{i}", ContactMatch(
        mode="subtree", pattern="trunk_base", entity="robot"), name)
        for i, name in enumerate(names[:-1])]
    cfg.scene.sensors = tuple(sensors)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 160
    # Better resolved small block contacts than the walking recipe's 5 ms.
    cfg.sim.mujoco.timestep = 0.0025
    cfg.decimation = 8  # retain 50 Hz policy control
    # BAM lag counts physics steps: double lags when halving the timestep.
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag *= 2
        actuator.delay_max_lag *= 2
    cfg.viewer.distance = 0.85
    cfg.viewer.elevation = -18
    cfg.viewer.azimuth = 135
    cfg.viewer.width, cfg.viewer.height = 960, 720
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = (0.08, 0.0, 0.12)

    for name in ("ball_forward_velocity", "ball_speed_overshoot", "support_foot_grounded",
                 "pose_stand_neck", "height_stand"):
        cfg.rewards.pop(name, None)
    cfg.rewards["pose_stand_legs"].weight = 0.4  # permit a deliberate crouch
    cfg.rewards["upright"].weight = 0.7
    cfg.rewards["jenga_approach"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=2.0, params={"term": "approach_delta"})
    cfg.rewards["jenga_push"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=10.0, params={"term": "push_delta"})
    cfg.rewards["jenga_success"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=8.0, params={"term": "success_delta"})
    # Costs below are nonnegative; negative weights are intentional.
    cfg.rewards["jenga_base_motion"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=-2.0, params={"term": "base_cost"})
    cfg.rewards["jenga_failure"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=-25.0, params={"term": "failure_impulse"})
    cfg.rewards["jenga_head_force"] = RewardTermCfg(func=mdp.jenga_reward,
        weight=-0.05, params={"term": "force_cost"})
    for key in ("ball_position", "ball_velocity"):
        cfg.observations["critic"].terms.pop(key)
    cfg.observations["critic"].terms["jenga_state"] = ObservationTermCfg(
        func=mdp.jenga_privileged_state, params={"names": names})

    cfg.events.pop("reset_ball")
    cfg.events.pop("push_robot", None)  # isolate manipulation discovery
    cfg.curriculum.pop("push_magnitude", None)
    # Avoid inheriting an unrelated kick curriculum on small, fragile towers.
    cfg.curriculum.clear()
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.events["set_ground_state"].params["sitting_tilt_max"] = math.radians(1)
    cfg.events["set_ground_state"].params["standing_z_min"] = 0.115
    cfg.events["set_ground_state"].params["standing_z_max"] = 0.115
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.events["reset_jenga"] = EventTermCfg(func=mdp.reset_jenga_tower, mode="reset",
        params={"names": names, "half_sizes": sizes, "offset_x": TOWER_OFFSET_X,
                "noise_xy": PLACEMENT_NOISE_M})
    cfg.terminations["jenga_failure"] = TerminationTermCfg(func=mdp.jenga_failure)
    cfg.terminations.pop("fell_over", None)  # one documented fall criterion / metric
    cfg.terminations["jenga_nan"] = TerminationTermCfg(func=mdp.jenga_nonfinite,
        params={"names": names})
    # Contacts are refreshed and sampled once in _jenga_update. Per-substep
    # metrics in mjlab 1.3.0 cannot supply fresh contact/site data.
    cfg.metrics = {}
    for metric in ("success", "robot_fall", "base_failure", "wrong_contact",
                   "head_contact", "top_fallen", "base_max_xy_m", "base_max_tilt_rad",
                   "max_head_force_n", "fast_strike", "strike_head_speed_m_s",
                   "strike_peak_head_forward_m_s", "strike_contact_duration_s",
                   "strike_first_contact_s", "strike_separation_delay_s"):
        cfg.metrics[f"jenga_{metric}"] = MetricsTermCfg(func=mdp.jenga_metric,
            params={"key": metric}, reduce="last")
    return cfg


def make_jenga_rl_cfg(layout="Vertical3"):
    cfg = deepcopy(MicroduckBallKickRlCfg)
    cfg.experiment_name = f"jenga_head_{layout.lower()}_blind_v1"
    cfg.run_name = cfg.experiment_name
    cfg.max_iterations = 2000
    cfg.save_interval = 100
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
