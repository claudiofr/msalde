SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONF_ID=$1
export DATASET=$2
export NUM_VARS=$3
bsub -J job${CONF_ID} -oo job${CONF_ID}.out -eo job${CONF_ID}.err < $SCRIPT_DIR/runs.lsf
