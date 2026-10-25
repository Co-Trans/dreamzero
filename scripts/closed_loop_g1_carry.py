#!/usr/bin/env python3
"""Closed-loop test of a g1_carry checkpoint in WAM_CT's MuJoCo scene: both G1s driven by it (one model, one KV cache
per robot), each through its own Converter and ScaleBFM (scalebfm_ee_posttrain) as in WAM_CT's
scripts/scalebfm_wbc/run_episode.py, which collected the data. An episode is a dataset npz of the export's validation
split, a carry export of WAM_CT's scripts/dreamzero/export_lerobot.py (its payload, object and seed rebuild the same
scene); each robot's plan tokens are that npz's recorded plan_gt at the same expert step (open loop: how the payload
moved next under the expert). The expert drives both robots (run_episode.relative_actions()) until the carry starts
(step k0), the checkpoint drives the carry, and the run ends where the plan's carry ends (k1; putting the payload down
is the expert's). Calls as in eval_g1_carry.py from k0: at expert step k0 + 8c each robot sends its head-camera frames
k0 + 8(c-1) .. k0 + 8c (frame k0 alone at c = 0) and its state then (q and g from the simulation, its odometry since
step 0, its palms and hands' wrenches from its own joints and F/T sensors as export_lerobot.body_palms() and hands()),
and executes the 8 actions it gets over steps k0 + 8c .. k0 + 8c+7.

Writes OUT/<name>.npz (run_episode.record()'s 50 Hz log, the executed actions (K, 2, 18), k0, k1 and the result),
OUT/<name>.mp4 (overview) and OUT/<name>_r1.mp4, _r2.mp4 (the head cameras), and prints the result (carry_result()).

Usage (from the repo root, one GPU; ScaleBFM's files under $HOME, as WAM_CT's scalebfm_wbc/policy.py expects):
  MUJOCO_GL=egl PYTHONPATH=<WAM_CT>:<WAM_CT>/scripts/scalebfm_wbc:<WAM_CT>/scripts/dreamzero python scripts/closed_loop_g1_carry.py \
      --ckpt <run>/checkpoint-5000 --agibot <DreamZero-AgiBot> --data <lerobot_carry2/val> --out <dir> \
      data_v3/box/box_s931462.npz ...
"""

from eval_g1_carry import BLOCK, lora_on_agibot  # first: it disables torch._dynamo

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import pandas as pd
import torch.distributed as dist
from scipy.spatial.transform import Rotation
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.base_vla import VLA
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

import export_lerobot as X
import payloads as P
import scalebfm_wbc.expert as E
import scalebfm_wbc.policy as BFM
from planner.data import rot6, rot_z
from run_episode import (EGO, PHYSICS_DT, SUB, WBC_DT, Converter, build, heading, plan, record, relative_actions, wrap,
                         wrists_in_yaw_frame)

# The action head's state of one sequence (its KV caches and first frame), kept per robot between its calls
CACHE = ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg", "clip_feas", "ys", "current_start_frame", "language")


def act(policy, cache, obs):
    """One robot's 8 actions (8, 18) from one call, on its own cache (a dict of CACHE, empty before its first call)."""
    head = policy.trained_model.action_head
    for k, v in cache.items():
        setattr(head, k, v)
    out, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
    cache.update({k: getattr(head, k) for k in CACHE})
    return np.concatenate([out.act["action.pelvis"], out.act["action.palms"]], axis=-1).reshape(BLOCK, -1)


def g_now(row, i):
    """planner.data's g of robot i at a record() row: the payload in the robot's heading frame (position, rotation 6D,
    linear and angular velocity)."""
    Rh = rot_z(heading(row["root_quat"][i]))
    R = Rotation.from_quat(row["payload_quat"], scalar_first=True).as_matrix()
    return np.concatenate([Rh.T @ (row["payload_pos"] - row["root_pos"][i]), rot6(Rh.T @ R), Rh.T @ row["payload_linvel"],
                           Rh.T @ row["payload_angvel"]])


def odom(start, row, i):
    """planner.data's odom of robot i at a record() row: its pose in its heading frame at the start row (x, y, cos yaw,
    sin yaw)."""
    yaw0 = heading(start["root_quat"][i])
    d = rot_z(yaw0).T @ (row["root_pos"][i] - start["root_pos"][i])
    dyaw = heading(row["root_quat"][i]) - yaw0
    return np.array([d[0], d[1], np.cos(dyaw), np.sin(dyaw)])


def carry_result(ep, log, k0, k1):
    """The carry the checkpoint drove (steps k0 .. k1): held (the payload never 5 cm under its height when the carry
    started), carried (m: how far the payload went until then, or to the end) of planned (m), end (the payload's
    distance, m, and yaw, deg, from the plan's at the last step) and tilt (deg, the largest while held)."""
    p, z0 = log["payload_pos"][SUB * k0 :], log["payload_pos"][SUB * k0, 2]
    low = np.flatnonzero(p[:, 2] < z0 - 0.05)
    n = low[0] if len(low) else len(p)
    w, x, y, _ = log["payload_quat"][SUB * k0 : SUB * k0 + n].T
    k = k0 + (len(p) - 1) // SUB  # the plan's step at the last row
    return dict(held=not len(low), carried=float(np.linalg.norm(np.diff(p[:n, :2], axis=0), axis=1).sum()),
                planned=float(np.linalg.norm(np.diff(ep["pos"][k0:k1], axis=0), axis=1).sum()),
                end_xy=float(np.linalg.norm(p[-1, :2] - ep["pos"][k])),
                end_yaw=float(np.degrees(abs(wrap(heading(log["payload_quat"][-1]) - ep["yaw"][k])))),
                tilt=float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))).max()))


def run(policy, ep, plans, modality, trackers, out, name, fk):
    """One episode, plans (2, k1 - k0, 240) the robots' plan tokens per carry step; the result of carry_result()."""
    k0, k1 = (np.flatnonzero(ep["phase"] == E.PHASES.index(p))[0] for p in ("carry", "lower"))
    m, d, idx, act_ids, limit = build(ep, trackers[0].default)
    kp, kd = trackers[0].kp, trackers[0].kd
    for i, x in enumerate(idx["robots"]):
        trackers[i].reset(d.qpos[x["root_q"] : x["root_q"] + 3], d.qpos[x["root_q"] + 3 : x["root_q"] + 7],
                          d.qvel[x["root_v"] + 3 : x["root_v"] + 6], d.qpos[x["qpos"]], d.qvel[x["dof"]])
    conv = [Converter(d.qpos[x["root_q"] : x["root_q"] + 2], heading(d.qpos[x["root_q"] + 3 : x["root_q"] + 7]),
                      wrists_in_yaw_frame(m, d, i)) for i, x in enumerate(idx["robots"])]
    pl = plan(ep)
    ego, view = mujoco.Renderer(m, *EGO), mujoco.Renderer(m, 960, 1280)
    cam = mujoco.MjvCamera()  # overview camera following the payload
    cam.distance, cam.elevation, cam.azimuth = 6.0, -30.0, 135.0
    view_video = iio.get_writer(out / f"{name}.mp4", fps=25)
    frames, caches, rows, acts = [[], []], [{}, {}], [], []
    for n in range(k1 * SUB):
        row = record(m, d, idx)
        if n % SUB == 0:
            k = n // SUB
            for i in range(2):
                ego.update_scene(d, camera=f"robot_{i + 1}/head_camera")
                frames[i].append(ego.render())
            if k >= k0 and (k - k0) % BLOCK == 0:
                block = []
                for i in range(2):
                    p, R, w = X.hands(fk, row["root_pos"][i], row["root_quat"][i], row["q"][i], row["wrench"][i])
                    s = np.concatenate([row["q"][i], g_now(row, i), odom(rows[0], row, i),
                                        X.body_palms(row["root_pos"][i], row["root_quat"][i], p, R), w.ravel()]).astype(np.float32)
                    obs = {"video.head": np.array(frames[i][-1:] if k == k0 else frames[i][k - BLOCK :]),
                           "annotation.task": ep["instruction"], "state.plan": plans[i][k - k0 : k - k0 + 1]}
                    for key, sl in modality["state"].items():
                        if key != "plan":
                            obs[f"state.{key}"] = s[None, sl["start"] : sl["end"]]
                    block.append(act(policy, caches[i], obs))
            xy, yaw = row["root_pos"][:, :2], np.array([heading(q) for q in row["root_quat"]])
            if k < k0:
                a = relative_actions(k, pl, xy, yaw)
            else:
                a = np.array([block[i][(k - k0) % BLOCK] for i in range(2)])
            for i in range(2):
                conv[i].act(n * WBC_DT, a[i], xy[i], yaw[i])
            acts.append(a)
        rows.append(row)
        targets = []
        for i, x in enumerate(idx["robots"]):
            lp, lq = conv[i].targets(n * WBC_DT, d.qpos[x["root_q"] : x["root_q"] + 2], heading(d.qpos[x["root_q"] + 3 : x["root_q"] + 7]))
            targets.append(trackers[i].step(d.qpos[x["root_q"] : x["root_q"] + 3], d.qpos[x["root_q"] + 3 : x["root_q"] + 7],
                                            d.qvel[x["root_v"] + 3 : x["root_v"] + 6], d.qpos[x["qpos"]], d.qvel[x["dof"]], lp, lq))
        for _ in range(round(WBC_DT / PHYSICS_DT)):
            for i, x in enumerate(idx["robots"]):
                tau = kp * (targets[i] - d.qpos[x["qpos"]]) - kd * d.qvel[x["dof"]]
                d.ctrl[act_ids[i]] = np.clip(tau, -limit[i], limit[i])
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        if n % 2 == 0:  # 25 fps
            cam.lookat[:] = [*row["payload_pos"][:2], 0.6]
            view.update_scene(d, cam)
            view_video.append_data(view.render())
        if min(d.qpos[x["root_q"] + 2] for x in idx["robots"]) < 0.45:
            print(f"  a robot fell at {d.time:.2f} s ({E.PHASES[ep['phase'][n // SUB]]})", flush=True)
            break
    view_video.close()
    for i in range(2):
        iio.mimsave(out / f"{name}_r{i + 1}.mp4", frames[i], fps=1 / E.DT, codec="libx264", macro_block_size=8)
    log = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    result = carry_result(ep, log, k0, k1) | dict(fell=len(rows) < k1 * SUB)
    np.savez_compressed(out / f"{name}.npz", **log, action=np.array(acts), k0=k0, k1=k1, **result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--agibot", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("npz", nargs="+", help="validation episodes by their dataset npz, e.g. data_v3/box/box_s931462.npz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29534")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    VLA.load_lora = lora_on_agibot(args.agibot)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.G1_CARRY, model_path=args.ckpt, device="cuda")

    data = Path(args.data)
    modality = json.loads((data / "meta/modality.json").read_text())
    sources = json.loads((data / "meta/sources.json").read_text())
    BFM.torch.set_num_threads(2)
    bfm, md = BFM.load_policy("scalebfm_ee_posttrain")
    m0 = mujoco.MjModel.from_xml_path(str(BFM.SIM_XML))  # its hinge order is the order of record()
    names = [m0.joint(j).name for j in range(m0.njnt) if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    trackers = [BFM.Tracker(bfm, md, names) for _ in range(2)]
    fk = X.g1_model()
    for f in args.npz:
        folder, stem = Path(f).parent.name, Path(f).stem
        payload = "table" if folder == "table" or folder in P.DESKS else "box"
        obj = None if folder in ("box", "table") else folder
        ep = E.random_episode(np.random.default_rng(int(stem.split("_s")[-1])), payload, "scalebfm", obj)
        plans = [np.stack(pd.read_parquet(data / f"data/chunk-000/episode_{s['episode_index']:06d}.parquet")[modality["state"]["plan"]["original_key"]])
                 for s in sources if s["npz"] == f]  # robot 1, robot 2
        r = run(policy, ep, plans, modality, trackers, out, stem, fk)
        print(f"{f}: held {r['held']}, carried {r['carried']:.2f} of {r['planned']:.2f} m, end {100 * r['end_xy']:.1f} cm, "
              f"{r['end_yaw']:.1f} deg from the plan, tilt {r['tilt']:.1f} deg, a robot fell {r['fell']}", flush=True)


if __name__ == "__main__":
    main()
