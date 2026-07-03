#!/bin/bash
# Run VOTP (semi-supervised OT) on the MetaWorld environments.
#
# Usage:
#   bash bash_scripts/run_mw_votp.sh                                    # all envs, seeds 1-5
#   bash bash_scripts/run_mw_votp.sh mw_door-open-v2                    # a single env
#   bash bash_scripts/run_mw_votp.sh mw_door-open-v2 mw_sweep-into-v2   # a subset
#   GPU=1 SEEDS="1 2 3" bash bash_scripts/run_mw_votp.sh                # override GPU / seeds
#   PTHR=0.4 bash bash_scripts/run_mw_votp.sh mw_door-open-v2           # override preference threshold

GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3 4 5"}

CONFIG="VOTP_IQL.yaml"
REWARD_LOSS="linear"
N_LABELS=${N_LABELS:-10}
PTHR=${PTHR:-}    # if set, overrides the per-env preference threshold below
EQTHR=${EQTHR:-}  # if set, overrides the per-env equal-teacher threshold below

# Per-environment tuned config
declare -A PREFERENCE_THRESHOLD=(
  ["mw_door-open-v2"]=0.4
  ["mw_drawer-open-v2"]=0.4
  ["mw_plate-slide-v2"]=0.35
  ["mw_sweep-into-v2"]=0.45
)
declare -A EQUAL_PREF_THRESHOLD=(
  ["mw_door-open-v2"]=0
  ["mw_drawer-open-v2"]=0
  ["mw_plate-slide-v2"]=0.05
  ["mw_sweep-into-v2"]=0
)

# Run all envs by default, or only the ones passed as arguments.
if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=(mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2 mw_sweep-into-v2)
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
      --config "configs/metaworld/${CONFIG}" \
      --env_name ${ENV} \
      --exp_name ${EXP_NAME} \
      --reward_loss_type ${REWARD_LOSS} \
      --equal_pref_threshold_teacher ${EQ} \
      --use_pseudo_label True \
      --preference_threshold ${THR} \
      --n_labels ${N_LABELS} \
      --seed ${seed}
  done
done
