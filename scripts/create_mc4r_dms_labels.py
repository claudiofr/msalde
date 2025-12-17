import pandas as pd

dms_file = '/home/claudiof/data/al/dms/mc4r-dms.tsv'
wt_file = '/home/claudiof/data/al/dms/mc4r-wt.tsv'

fasta_file = '/home/claudiof/data/al/dms/MC4R.fasta'
labels_file = '/home/claudiof/data/al/dms/MC4R_labels.csv'

input_df = pd.read_csv(dms_file, sep='\t')
input_df = input_df[(input_df['aa'] != '*')
                    & (input_df['pathway'] == 'Gs')
                    & (input_df['compound'].isna())]
wt_df = pd.read_csv(wt_file, sep='\t')

input_df = input_df.merge(
    wt_df,
    left_on='pos',
    right_on='Pos',
    suffixes=('', '_wt')
)

input_df['variant_id'] = (
    input_df['WT_AA_Short'] + input_df['pos'].astype(str) +
    input_df['aa']
)
input_df.rename(columns={'statistic': 'z_score'}, inplace=True)

wt_sequence = ''.join(wt_df.sort_values('Pos')['WT_AA_Short'].tolist())

input_df['sequence'] = input_df.apply(
    lambda row: wt_sequence[:row['pos'] - 1] +
    row['aa'] +
    wt_sequence[row['pos']:],
    axis=1
)

wt_row = pd.DataFrame({"variant_id": ["WT"], "z_score": [0],
                       "sequence": [wt_sequence]})
input_df = pd.concat([input_df, wt_row], ignore_index=True)

with open(fasta_file, "w") as f:
    for variant_id, sequence in input_df[['variant_id', 'sequence']].values:
        f.write(f">{variant_id}\n")   # header line
        f.write(f"{sequence}\n")       # sequence line

input_df[['variant_id', 'z_score']].to_csv(
    labels_file, index=False
)
