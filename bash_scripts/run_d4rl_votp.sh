#!/bin/bash
# Run VOTP (semi-supervised OT) on the D4RL gym-locomotion environments.
#
# Usage:
#   bash bash_scripts/run_d4rl_votp.sh                                         # all envs, seeds 1-5
#   bash bash_scripts/run_d4rl_votp.sh hopper-medium-replay-v2                 # a single env
#   bash bash_scripts/run_d4rl_votp.sh hopper-medium-expert-v2 walker2d-medium-replay-v2   # a subset
#   GPU=1 SEEDS="1 2 3" bash bash_scripts/run_d4rl_votp.sh                     # override GPU / seeds
#   PTHR=0.2 bash bash_scripts/run_d4rl_votp.sh hopper-medium-replay-v2        # override preference threshold

GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3 4 5"}

CONFIG="VOTP_IQL.yaml"
REWARD_LOSS="ce"
N_LABELS=${N_LABELS:-10}
PTHR=${PTHR:-}    # if set, overrides the per-env preference threshold below
EQTHR=${EQTHR:-}  # if set, overrides the per-env equal-teacher threshold below
WANDB="online"                          # wandb mode: online | offline | disabled
export WANDB_PROJECT="VOTP-LOCO"     # set your wandb project name here

# Per-environment tuned config
declare -A PREFERENCE_THRESHOLD=(
  ["hopper-medium-expert-v2"]=0.15
  ["hopper-medium-replay-v2"]=0.2
  ["walker2d-medium-expert-v2"]=0.2
  ["walker2d-medium-replay-v2"]=0.2
)
declare -A EQUAL_PREF_THRESHOLD=(
  ["hopper-medium-expert-v2"]=0
  ["hopper-medium-replay-v2"]=0
  ["walker2d-medium-expert-v2"]=0
  ["walker2d-medium-replay-v2"]=0
)

# Run all envs by default, or only the ones passed as arguments.
if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=(hopper-medium-expert-v2 hopper-medium-replay-v2 walker2d-medium-expert-v2 walker2d-medium-replay-v2)
fi

for ENV in "${TASKS[@]}"; do
  THR=${PREFERENCE_THRESHOLD[$ENV]}
  if [ -z "$THR" ]; then
    echo "Error: unknown env '$ENV'. Valid: ${!PREFERENCE_THRESHOLD[*]}"
    exit 1
  fi
  THR=${PTHR:-$THR}   # PTHR overrides the per-env default when set
  EQ=${EQTHR:-${EQUAL_PREF_THRESHOLD[$ENV]}}   # EQTHR overrides the per-env default when set

  EXP_NAME="${ENV}_votp_N${N_LABELS}_pth${THR}_eqthr${EQ}_${REWARD_LOSS}"

  for seed in $SEEDS; do
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --config "configs/locomotion/${CONFIG}" \
      --env_name ${ENV} \
      --exp_name ${EXP_NAME} \
      --reward_loss_type ${REWARD_LOSS} \
      --equal_pref_threshold_teacher ${EQ} \
      --use_pseudo_label True \
      --preference_threshold ${THR} \
      --n_labels ${N_LABELS} \
      --wandb ${WANDB} \
      --seed ${seed}
  done
done
