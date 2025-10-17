SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bsub -J job_run_maves -oo job_run_maves.out -eo job_run_maves.err < $SCRIPT_DIR/run_maves.lsf
