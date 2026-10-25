#!/bin/bash
# DreamZero on WAM_CT collaborative transport (two Unitree G1), LoRA from DreamZero-AgiBot.
#
# Usage:
#   G1_DATA_ROOT=/path/to/lerobot_carry2/train OUTPUT_DIR=... PLAN=gt [PALM_WEIGHT=3 VIDEO_WEIGHT=0.5] bash scripts/train/g1_carry_training.sh
#
# PLAN=gt | pred: each block gets its plan as 16 extra state tokens (state.plan, from the dataset's plan_gt or plan_pred
#                 column: meta/modality.json "plan" -> original_key), and a fixed text instead of the instruction
#                 (the controller gets the task only through the plan).
# PLAN=none:      no plan; the instruction text (DreamZero's own conditioning), as the control.
# PALM_WEIGHT:    weight of the 14 palm dims in the action loss (the 4 pelvis dims: 1); VIDEO_WEIGHT: of the video loss.
#
# Prerequisites: the dataset converted with scripts/data/convert_lerobot_to_gear.py --embodiment-tag g1_carry and
# meta/modality.json given the "plan" state key; Wan2.1-I2V-14B-480P (T5, CLIP, VAE), the umt5-xxl tokenizer and
# DreamZero-AgiBot (see docs/DATASET_TO_GEAR_AND_TRAIN.md).
# Checkpoints hold only the LoRA weights (save_lora_only, save_only_model: about 0.2 GB; with DeepSpeed's
# optimizer state one is 87 GB), so a run restarts from them without its optimizer state.

export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-disabled}  # report_to=none; wandb_project is still a required setting

G1_DATA_ROOT=${G1_DATA_ROOT:?"Set G1_DATA_ROOT to the GEAR-converted dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_g1_carry_lora"}
PLAN=${PLAN:-gt}
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
AGIBOT_DIR=${AGIBOT_DIR:-"./checkpoints/DreamZero-AgiBot"}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-5000}
BATCH=${BATCH:-1}
LR=${LR:-1e-4}
PALM_WEIGHT=${PALM_WEIGHT:-1}
VIDEO_WEIGHT=${VIDEO_WEIGHT:-1}

if [ "$PLAN" = "none" ]; then
  PLAN_ARGS=(data=dreamzero/g1_carry num_state_per_block=1 "g1_carry_state_keys=[state.joint_pos,state.g,state.odom,state.palms,state.h_int]"
             'g1_carry_state_normalization=${g1_carry_norm_noplan}')
else
  PLAN_ARGS=(data=dreamzero/g1_carry num_state_per_block=17 ++model_specific_transform.plan_steps=16 ++model_specific_transform.plan_step_dim=15
             ++model_specific_transform.always_use_default_instruction=true
             "++model_specific_transform.default_instruction='carry the payload together with the other robot.'")
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=8 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=8 \
    "${PLAN_ARGS[@]}" \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=20 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=640 \
    image_resolution_height=352 \
    save_lora_only=true \
    ++training_args.save_only_model=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    g1_carry_data_root=$G1_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$AGIBOT_DIR \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    '++action_head_cfg.config.diffusion_model_cfg.max_state_dim=${max_state_dim}' \
    "++action_head_cfg.config.action_loss_weights=[1,1,1,1$(printf ",$PALM_WEIGHT%.0s" {1..14})]" \
    ++action_head_cfg.config.video_loss_weight=$VIDEO_WEIGHT
