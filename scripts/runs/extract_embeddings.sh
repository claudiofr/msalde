SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bsub -J extract_embeddings -oo extract_embeddings.out -eo extract_embeddings.err < $SCRIPT_DIR/extract_embeddings.lsf
