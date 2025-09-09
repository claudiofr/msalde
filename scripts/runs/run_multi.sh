for arg in "$@"; do
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    . $SCRIPT_DIR/runs.sh "$arg"
done