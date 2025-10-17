import os
import SeqIO
import pandas as pd
from pathlib import Path


print("start")
dir = "/sc/arion/work/fratac01/data/al/input/EvolvePro/"
to_dir = "/sc/arion/work/fratac01/data/al/dms"

# Define the directory
dir = Path('/home/claudiof')
to_dir = '/home/claudiof/work'

dir = Path('/sc/arion/work/fratac01/data/al/dms')
to_dir = '/sc/arion/work/fratac01/data/al/dms/work'

# Example: Get all .txt files
files = list(dir.glob('*.fasta'))

def update_row(row):
    # Example update: Add line numbers
    if row[0] != ">":
        return row
    variant_id = row[1:].strip().split()[0]
    return f">{variant_id}\n"


for file in files:
    # Open the input file for reading
    with open(file, 'r') as infile:
        lines = infile.readlines()

    # Update each line (e.g., add line numbers)
    updated_lines = [update_row(line) for line in lines]

    # Write the updated lines to a new output file
    with open(os.path.join(to_dir, os.path.basename(file)), 'w') as outfile:
        outfile.writelines(updated_lines)

dir = Path('/home/claudiof')
dir = Path('/sc/arion/work/fratac01/data/al/dsm')
to_dir = '/home/claudiof/work'
to_dir = '/sc/arion/work/fratac01/data/al/dsm/work'

# Example: Get all .txt files
files = list(dir.glob('*.fasta'))

def update_esm_row(i, row):
    if i == 0:
        return row
    split_row = row.split(",", 1)
    variant_id = split_row[0].split()[0]
    embeddings = split_row[1]
    return f"{variant_id},{embeddings}"

# Find all keys in fasta_dict that have a given value (could be duplicates)
def find_keys_by_value(d, value):
    return [k for k, v in d.items() if v == value]


for file in files:
    # Open the input file for reading
    fasta_dict = {record.id: str(record.seq)
                  for record in SeqIO.parse(file, "fasta")}
    wt_sequence = fasta_dict.get("WT")
    silent_variant_ids = find_keys_by_value(fasta_dict, wt_sequence)
    labels_file = file.replace(".fasta", "_labels.csv")
    labels_df = pd.read_csv(labels_file)
    silent_variant_labels = labels_df[labels_df['mutant'].isin(silent_variant_ids)]
    labels_df.to_csv(labels_file, index=False)

    with open(file, 'r') as infile:
        lines = infile.readlines()

    # Update each line (e.g., add line numbers)
    updated_lines = [update_esm_row(i, line) for i, line in enumerate(lines)]

    # Write the updated lines to a new output file
    with open(os.path.join(to_dir, os.path.basename(file)), 'w') as outfile:
        outfile.writelines(updated_lines)



