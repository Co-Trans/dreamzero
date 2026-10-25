# DreamZero for Collaborative Transport (g1_carry)

This fork post-trains DreamZero on two Unitree G1 humanoids carrying a payload together, from the
[WAM_CT](https://github.com/Co-Trans/Collaborate_Transport) project. Each robot runs the same model on its own
sensors only: its head camera, its proprioception and hand wrenches, and a plan of how the payload should move next.
There is no communication between the robots.

It is a LoRA post-training of the released DreamZero-AgiBot checkpoint (Wan2.1-I2V-14B backbone). Everything else
(installation, checkpoints, the inference server, DROID training) is as in the [main README](../README.md).

## The task as DreamZero sees it

- **One robot's view per episode.** Every recorded two-robot episode becomes two training episodes, one per robot.
- **Only the carry.** The data holds the carry phase only, from lift-off to the start of putting the payload down.
  Walking in, grasping, putting down and regrasping are left to the expert.
- **A block is 8 rows at 6.25 Hz (1.28 s):** 8 head-camera frames (2 latent frames) and 8 actions. A training clip
  is a first frame and 4 blocks (33 frames).

| | keys | dims |
|---|---|---|
| video | `video.head`: the head camera, RGB 640 x 352 | |
| state | `state.joint_pos` (29), `state.g` (15: the payload's pose and velocity in the robot's heading frame), `state.odom` (4: the robot's pose since the start), `state.palms` (18: the measured palms, position and rotation 6D, in the heading frame), `state.h_int` (12: each hand's 6-axis wrench) | 78, one state token per block |
| plan | `state.plan`: the payload at the next 16 steps in its own frame (position, rotation 6D, linear and angular velocity) | 16 x 15, 16 more state tokens per block |
| action | `action.pelvis` (4: dx, dy, dyaw of the next step in the heading frame, pelvis height), `action.palms` (14: each palm's position and wxyz quaternion in the heading frame) | 18 per row |
| text | with a plan, the fixed "carry the payload together with the other robot." (the task reaches the model only through the plan) | |

All states and actions are normalized with their 1 % and 99 % quantiles.

## Changes from upstream

| file | change |
|---|---|
| `groot/vla/data/schema/embodiment_tags.py`, `configs/model/dreamzero/transform/base.yaml`, `scripts/data/convert_lerobot_to_gear.py` | the embodiment tag `g1_carry` (projector index 33) |
| `configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | `modality_config_g1_carry` and `transform_g1_carry`: the keys above and their normalization |
| `configs/data/dreamzero/g1_carry.yaml` (new) | the dataset: path, 8 rows per block, a frame every row |
| `data/dataset/lerobot_sharded.py` | `chunk_rows` and `video_stride` instead of DROID's fixed 24 rows and every 3rd frame |
| `model/dreamzero/transform/dreamzero_cotrain.py` | the plan as 16 extra state tokens (`plan_steps`, `plan_step_dim`), each step followed by a one-hot of its index; the g1_carry text prompt |
| `model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`, `action_head/wan_flow_matching_action_tf.py` | more than one state token per block; per-dimension action loss weights (`action_loss_weights`) and a video loss weight (`video_loss_weight`) |
| `experiment/base.py` | pretrained weights of another shape are skipped (AgiBot's state encoder expects a different state size, so its first layer starts fresh) |
| `scripts/train/g1_carry_training.sh` (new) | the training script |
| `scripts/eval_g1_carry.py` (new) | open-loop evaluation |
| `scripts/closed_loop_g1_carry.py` (new) | closed-loop test in WAM_CT's MuJoCo scene |

## 1. Data

The episodes come from WAM_CT's privileged expert:
[G1_Collaborate_Transport_v3](https://huggingface.co/datasets/Co-Trans/G1_Collaborate_Transport_v3) (220 clean
episodes) and G1_Collaborate_Transport_dist (275 episodes with noise and disturbances in the carry; the label is
the expert's corrective action). WAM_CT exports their carry to LeRobot v2, one robot per episode. Per payload the
last 4 seeds of v3 validate; all disturbed episodes train. From the WAM_CT repo root:

```bash
PYTHONPATH=scripts/scalebfm_wbc .venv/bin/python scripts/dreamzero/export_lerobot.py \
    --data outputs/data_v3 --extra outputs/data_dist/* --out outputs/lerobot_carry2   # --plan gt (default)
```

Then, here, the GEAR metadata (statistics, tasks, episodes) for each split:

```bash
for s in train val; do
  python scripts/data/convert_lerobot_to_gear.py --dataset-path <data>/lerobot_carry2/$s \
      --embodiment-tag g1_carry --task-key annotation.task
done
```

The result: train 900 episodes (146,138 rows, 6.5 h), val 88 episodes (14,192 rows). An episode that was dropped
and regrasped is left out.

## 2. Training

```bash
G1_DATA_ROOT=<data>/lerobot_carry2/train OUTPUT_DIR=<run> \
WAN_CKPT_DIR=<Wan2.1-I2V-14B-480P> TOKENIZER_DIR=<umt5-xxl> AGIBOT_DIR=<DreamZero-AgiBot> \
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 BATCH=1 MAX_STEPS=20000 SAVE_STEPS=1000 \
PLAN=gt PALM_WEIGHT=3 VIDEO_WEIGHT=0.5 \
bash scripts/train/g1_carry_training.sh
```

| variable | default | |
|---|---|---|
| `PLAN` | `gt` | `gt` / `pred`: the plan tokens from the export's `plan_gt` (the payload's recorded future) or `plan_pred` (the object-level planner's) column; `none`: no plan, the instruction text instead |
| `PALM_WEIGHT` | 1 | weight of the 14 palm dimensions in the action loss (the 4 pelvis dimensions: 1) |
| `VIDEO_WEIGHT` | 1 | weight of the video loss |
| `LR` | 1e-4 | peak learning rate: 5 % warmup, then cosine decay to 0 at `MAX_STEPS` |
| `MAX_STEPS`, `SAVE_STEPS`, `BATCH`, `NUM_GPUS` | 100000, 5000, 1, all | |

- **Trained:** LoRA (rank 4) on the DiT's attention and feed-forward layers, the state encoder, the action encoder
  and decoder. The text encoder, image encoder, VAE and the DiT itself stay frozen.
- **Checkpoints** (`<run>/checkpoint-<step>`, about 0.2 GB) hold only these trained weights, without the optimizer
  state, so a stopped run cannot be resumed exactly.
- **Cost:** on two B200 GPUs with one clip each, 74 GB per GPU and 5.3 s per step: 20k steps take about 29.5 h.

## 3. Open-loop evaluation

The model gets the recorded frames and states of held-out episodes, block by block as at deployment (one robot,
its KV cache kept), and its actions are compared with the expert's. Nothing is executed.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_g1_carry.py --ckpt <run>/checkpoint-20000 --agibot <DreamZero-AgiBot> \
    --data <data>/lerobot_carry2/val --out <dir> --pick 42 53 24 80 26 1 51 17 9 21 10 72 12 22 18 63 45 15 --videos 4
```

It writes `metrics.json` (per episode and overall: pelvis step xy error cm, yaw deg, height cm, palm position cm,
palm rotation deg), `episodes/<episode>.npz` (predicted and true actions) and `videos/<episode>.mp4` (the real
head camera left, the imagined one right). About 20 minutes for 18 episodes on one B200.

## 4. Closed-loop test

Both robots are simulated in WAM_CT's MuJoCo scene, each driven by the same model with its own KV cache, through
WAM_CT's action converter and the ScaleBFM whole-body controller (`scalebfm_ee_posttrain`)
that collected the data. The expert walks in, grasps and lifts; the model carries; the run ends where the plan
starts putting the payload down. The plan tokens are the episode's recorded `plan_gt`. It needs the WAM_CT repo
on the `PYTHONPATH` and ScaleBFM's files under `$HOME` (as WAM_CT's `scalebfm_wbc/policy.py` expects):

```bash
MUJOCO_GL=egl PYTHONPATH=<WAM_CT>:<WAM_CT>/scripts/scalebfm_wbc:<WAM_CT>/scripts/dreamzero \
python scripts/closed_loop_g1_carry.py --ckpt <run>/checkpoint-20000 --agibot <DreamZero-AgiBot> \
    --data <data>/lerobot_carry2/val --out <dir> data_v3/box/box_s931462.npz data_v3/table/table_s956507.npz
```

Per episode it prints whether the payload stayed in the hands, how far it was carried of the planned distance,
the end error from the plan and the largest tilt, and writes the 50 Hz log, the executed actions and the videos.

## Results so far

Run with `PLAN=gt PALM_WEIGHT=3 VIDEO_WEIGHT=0.5`, two B200 GPUs, open loop on the 18 validation episodes above:

| step | pelvis xy cm | yaw deg | palm position cm | palm rotation deg |
|---|---|---|---|---|
| 5k | 5.05 | 1.09 | 2.20 | 48.9 |
| 10k | 3.43 | 1.04 | 2.10 | 34.5 |
| 15k | 2.88 | 0.92 | 1.99 | 23.6 |

The pelvis height is constant in the carry (0.789 m), so its error is 0. The palm rotation is learned slowly: the
palms are held inward for boxes and bins and turned 90 deg (palms up) for tables. At 15k the tables are within
4.3 deg, the others still 24.9 deg off, but improving steadily.
