SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bsub < $SCRIPT_DIR/runs.lsf $1
