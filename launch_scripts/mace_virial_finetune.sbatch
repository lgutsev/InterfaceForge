#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=mace.virial
#SBATCH --output=mace_virial.%A_%a.out
#SBATCH --error=mace_virial.%A_%a.err

# A1 virial-aware fine-tune of the existing four-member SiN/TiN/TiO MACE
# committee. Submit from the Periodic_MLIPs campaign root:
#
#   sbatch --array=0-3%2 \
#     /project/lgutsev/git_develop/InterfaceForge/launch_scripts/mace_virial_finetune.sbatch
#
# A short seed-11 smoke can be run first with:
#
#   sbatch --export=ALL,MACE_SEED=11,MACE_MAX_EPOCHS=2,MACE_RUN_TAG=virial_smoke \
#     /project/lgutsev/git_develop/InterfaceForge/launch_scripts/mace_virial_finetune.sbatch
#
# Production output is isolated under
# models/mace_committee_520eV_virial_a1/mace_finetune_committee/seed_<seed>/.

set -euo pipefail

CAMP="${MACE_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the campaign root}}"
IFACE_ROOT="${INTERFACEFORGE_ROOT:-/project/lgutsev/git_develop/InterfaceForge}"
RUN_TAG="${MACE_RUN_TAG:-virial_a1}"
SEEDS=(11 23 37 53)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    INDEX="$SLURM_ARRAY_TASK_ID"
    [[ "$INDEX" =~ ^[0-3]$ ]] || {
        echo "Array index must be 0, 1, 2, or 3; received $INDEX" >&2
        exit 2
    }
    SEED="${SEEDS[$INDEX]}"
else
    SEED="${MACE_SEED:?Set MACE_SEED=11|23|37|53 or submit --array=0-3}"
fi

case "$SEED" in
    11|23|37|53) ;;
    *)
        echo "MACE_SEED must be one of 11, 23, 37, 53; received $SEED" >&2
        exit 2
        ;;
esac
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "Invalid MACE_RUN_TAG: $RUN_TAG" >&2
    exit 2
}

CAMP="$(cd "$CAMP" && pwd -P)"
DATASET="$CAMP/models/mace_committee_520eV"
SOURCE="$DATASET/mace_committee/seed_$SEED/mace_model/SiN_TiN_TiO_periodic_mace_seed${SEED}_stagetwo.model"
OUTPUT="$CAMP/models/mace_committee_520eV_$RUN_TAG"
LAUNCHER="$IFACE_ROOT/launch_scripts/mace_finetune_committee.sh"

[[ -s "$SOURCE" ]] || {
    echo "Missing source MACE member: $SOURCE" >&2
    exit 2
}
for split in train valid test; do
    [[ -s "$DATASET/${split}.extxyz" ]] || {
        echo "Missing canonical MACE split: $DATASET/${split}.extxyz" >&2
        exit 2
    }
done
[[ -s "$LAUNCHER" ]] || {
    echo "Missing InterfaceForge launcher: $LAUNCHER" >&2
    exit 2
}

export MACE_SEED="$SEED"
export MACE_FOUNDATION_MODEL="$SOURCE"
export MACE_DATASET_DIR="$DATASET"
export MACE_OUTPUT_ROOT="$OUTPUT"
export MACE_MODEL_PREFIX="${MACE_MODEL_PREFIX:-SiN_TiN_TiO_periodic_mace_$RUN_TAG}"

# A1 changes only the training objective: the exact original split is reused,
# and the REF_virial labels already present in every frame enter the loss.
export MACE_LOSS="${MACE_LOSS:-virials}"
export MACE_VIRIALS_KEY="${MACE_VIRIALS_KEY:-REF_virial}"
export MACE_VIRIALS_WEIGHT="${MACE_VIRIALS_WEIGHT:-1.0}"
export MACE_ENERGY_WEIGHT="${MACE_ENERGY_WEIGHT:-1.0}"
export MACE_FORCES_WEIGHT="${MACE_FORCES_WEIGHT:-100.0}"

# Conservative fine-tuning settings. Keep the loss weights constant throughout
# A1 rather than entering MACE's from-scratch stage-two/SWA weight schedule.
export MACE_LR="${MACE_LR:-0.001}"
export MACE_MAX_EPOCHS="${MACE_MAX_EPOCHS:-30}"
export MACE_PATIENCE="${MACE_PATIENCE:-10}"
export MACE_DEFAULT_DTYPE="${MACE_DEFAULT_DTYPE:-float64}"
export MACE_USE_STAGE_TWO="${MACE_USE_STAGE_TWO:-False}"
export MACE_EMA_DECAY="${MACE_EMA_DECAY:-0.999}"
export MACE_WEIGHT_DECAY="${MACE_WEIGHT_DECAY:-0.0}"
export MACE_CLIP_GRAD="${MACE_CLIP_GRAD:-1.0}"
export MACE_E0S="${MACE_E0S:-foundation}"
export MACE_MULTIHEADS="${MACE_MULTIHEADS:-False}"

echo "MACE virial A1 relaunch"
echo "  campaign: $CAMP"
echo "  seed:     $SEED"
echo "  source:   $SOURCE"
echo "  dataset:  $DATASET"
echo "  output:   $OUTPUT"
echo "  loss:     $MACE_LOSS"
echo "  virials:  $MACE_VIRIALS_KEY weight=$MACE_VIRIALS_WEIGHT"
echo "  E/F:      $MACE_ENERGY_WEIGHT / $MACE_FORCES_WEIGHT"
echo "  lr:       $MACE_LR"
echo "  epochs:   $MACE_MAX_EPOCHS"
echo "  stage2:   $MACE_USE_STAGE_TWO"

exec bash "$LAUNCHER"
