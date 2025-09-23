SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONF_ID=$1
bsub -J jobb${CONF_ID} -oo jobb${CONF_ID}.out -eo jobb${CONF_ID}.err < $SCRIPT_DIR/runs_multi_gpu.lsf
