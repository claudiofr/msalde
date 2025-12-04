import pandas as pd

ds = pd.read_csv('junk_datasets.csv')

cv = pd.read_csv("data/clinvar/clinvar_labels.csv")
gs = cv['gene_symbol'].drop_duplicates()
merged = ds.merge(gs, left_on='dataset', right_on='gene_symbol', how='left')
missing = merged[merged['gene_symbol'].isnull()]
pass


