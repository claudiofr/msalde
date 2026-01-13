import context
from msalde.container import ALDEContainer

fasta_file = '/home/claudiof/data/al/dms/MC4R.fasta'
labels_file = '/home/claudiof/data/al/dms/MC4R_labels.csv'

container = ALDEContainer()

loader = container.variant_ref_loader

variant_assay, input_df = loader.load_mc4r_dataset(pathway='Gs')

with open(fasta_file, "w") as f:
    for variant_id, sequence in input_df[['variant_id', 'sequence']].values:
        f.write(f">{variant_id}\n")   # header line
        f.write(f"{sequence}\n")       # sequence line

input_df[['variant_id', 'z_score']].to_csv(
    labels_file, index=False
)

repo = container.variant_repository
repo.add_variant_assay(variant_assay)
repo.add_variant_assays_bulk(input_df)
