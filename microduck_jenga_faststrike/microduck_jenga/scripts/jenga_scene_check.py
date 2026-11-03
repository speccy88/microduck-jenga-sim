"""CPU scene and ray geometry check; optional software-rendered start image.

No RL environment, Warp stepping, BAM approximation, or learned motion is run.
The 64 rays are geometry diagnostics, not a calibrated VL53L5CX sensor model.
"""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab_microduck.tasks.microduck_jenga_env_cfg import LAYOUTS, make_microduck_jenga_env_cfg


def tof_geometry(model, data):
    sid = model.site("robot/tof").id
    rotation = data.site_xmat[sid].reshape(3,3)
    origin = data.site_xpos[sid]
    output = []
    # Match pinned microduck kinematics/src/{head,tof}.rs: sensor +x forward,
    # +y left, +z up; angular zone centres across the 45x45 degree square.
    # SITE_TO_CV2 * SENSOR_IN_CV2_Q = identity on that source revision.
    for row in range(8):
        line = []
        for col in range(8):
            azimuth,elevation = np.radians([(4-col-0.5)*45/8, (4-row-0.5)*45/8])
            direction = rotation @ np.array([np.cos(elevation)*np.cos(azimuth),
                np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
            direction /= np.linalg.norm(direction)
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, origin, direction,
                np.array([1,0,0,1,0,0], dtype=np.uint8), True,
                int(model.site_bodyid[sid]), geom_id)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])) if geom_id[0]>=0 else None
            line.append({"range_m": float(distance), "hit": name})
        output.append(line)
    hits = [z["hit"] for row in output for z in row]
    return {"origin_world_m": origin.tolist(), "optical_x_world": rotation[:,0].tolist(),
            "assumed_fov_degrees":45, "rays":output,
            "block_zone_hits": {name:hits.count(name) for name in sorted({x for x in hits if x})
                                if name and name.startswith("jenga_")},
            "warning":"Ideal rays only; FOV/axis/zone integration/noise/latency uncalibrated."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES='' for CPU validation.")
    out = Path(__file__).resolve().parents[1]/"evidence"
    out.mkdir(exist_ok=True)
    results = {}
    for layout in LAYOUTS:
        cfg = make_microduck_jenga_env_cfg(play=True, layout=layout)
        cfg.scene.num_envs = 1
        scene = Scene(cfg.scene, device="cpu")
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0)
        mujoco.mj_forward(model,data)
        mouth = data.site_xpos[model.site("robot/mouth_tip").id]
        contacts = []
        for contact in data.contact:
            a,b = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or "" for g in contact.geom]
            if ((a.startswith("robot/") and b.startswith("jenga_")) or
                (b.startswith("robot/") and a.startswith("jenga_"))):
                contacts.append({"a":a,"b":b,"distance_m":float(contact.dist)})
        results[layout] = {"nq":model.nq,"nu":model.nu,"mouth_world_m":mouth.tolist(),
            "block_centers_world_m":[data.xpos[model.body(f"jenga_{i}/block").id].tolist()
                                    for i in range(len(LAYOUTS[layout]))],
            "initial_robot_block_contacts":contacts, "tof_geometry":tof_geometry(model,data)}
        if args.render:
            from PIL import Image
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.08,0,0.12]
            camera.distance, camera.azimuth, camera.elevation = 0.80, 135, -18
            with mujoco.Renderer(model,height=720,width=960) as renderer:
                from OpenGL.GL import glGetString, GL_RENDERER
                results[layout]["gl_renderer"] = glGetString(GL_RENDERER).decode()
                renderer.update_scene(data,camera=camera)
                Image.fromarray(renderer.render()).save(out/f"scene_{layout.lower()}.png")
    (out/"scene_geometry.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({k:{x:v for x,v in val.items() if x!="tof_geometry"}
                      for k,val in results.items()},indent=2))


if __name__ == "__main__":
    main()
