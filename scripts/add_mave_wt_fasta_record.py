import os
from pathlib import Path
import shutil

print("start")

wt_fasta_dir = Path('/hpc/users/fratac01/gitrepo/EvolvePro/data/dms/wt_fasta')
dir = Path('/sc/arion/work/fratac01/data/al/dms')
to_dir = '/sc/arion/work/fratac01/data/al/dms/work/'

# wt_fasta_dir = Path('/home/claudiof/gitrepo/EvolvePro/data/dms/wt_fasta')
# dir = Path('/home/claudiof')
# to_dir = '/home/claudiof/work/'

# Example: Get all .txt files
wt_fasta_files = list(wt_fasta_dir.glob('*.fasta'))

for file in wt_fasta_files:
    dataset_name = file.stem.replace("_WT", "")

    fasta_file = dir / (dataset_name + ".fasta")
    with open(fasta_file, 'r') as infile:
        wt_found = False
        for line in infile:
            if line.startswith(">WT"):
                wt_found = True
                break
        if wt_found:
            continue

    # Open the input file for reading
    with open(file, 'r') as infile:
        lines = infile.readlines()
        wt_sequence = "".join([line.strip() for line in lines if line[0] != ">"])

    shutil.copy(fasta_file, to_dir)

    with open(os.path.join(to_dir, dataset_name + ".fasta"), "a") as f:
        f.write(">WT\n")
        f.write(wt_sequence + "\n")

    break

