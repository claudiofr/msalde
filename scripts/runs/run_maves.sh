SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
filename_prefix="job_run_maves_$(date +%Y%m%d_%H%M%S)"
outfile="${filename_prefix}.out"
errfile="${filename_prefix}.err"
bsub -J job_run_maves -oo $outfile -eo $errfile < $SCRIPT_DIR/run_maves_gpu.lsf