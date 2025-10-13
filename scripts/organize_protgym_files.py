import os

print("start")
dir = "/sc/arion/work/fratac01/data/al/input/EvolvePro/"
to_dir = "/sc/arion/work/fratac01/data/al/dms"

subdirs = [d for d in os.listdir(dir)
    if os.path.isdir(os.path.join(dir,d))]
# subdirs = [d for d in os.listdir(dir)]
print(len(subdirs))

for subdir in subdirs:
    print(subdir, os.path.isdir(os.path.join(dir,subdir)))
    # break
    path = os.path.join(dir, subdir)
    dataset_name = subdir.split('_', 1)[0]
    to_labels_path = os.path.join(to_dir, f"{dataset_name}_labels.csv")
    to_fasta_path = os.path.join(to_dir, f"{dataset_name}.fasta")
    from_labels_path = os.path.join(path, "activity_score.csv")
    from_fasta_path = os.path.join(path, "sequences.fasta")
    print(f"Copying from {from_labels_path} to {to_labels_path}")
    print(f"Copying from {from_fasta_path} to {to_fasta_path}")
    os.system(f"cp {from_labels_path} {to_labels_path}")
    os.system(f"cp {from_fasta_path} {to_fasta_path}")
    # break