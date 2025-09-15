SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONF_ID=$1
bsub < $SCRIPT_DIR/runs.lsf $1
