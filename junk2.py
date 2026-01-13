import pandas as pd
import matplotlib.pyplot as plt


file = "/home/claudiof/urn_mavedb_00000013-a-1_scores.csv"
file = "/home/claudiof/data/al/dms/SRC_NIHMS1520956-supplement-4.xlsx"
file = "/home/claudiof/data/al/dms/kaggle_training_set_20230316.tsv"

df = pd.read_csv(file, sep="\t", skiprows=0)

dfg  = df.groupby("GENE_SYMBOL")["LABEL"].value_counts().reset_index()

dfg2 = dfg.pivot(index="GENE_SYMBOL", columns="LABEL", values="LABEL").fillna(0)

df4 = df[df["abundance_class"] == 4]
df3 = df[df["abundance_class"] == 3]
df2 = df[df["abundance_class"] == 2]


df4["score"].hist(alpha=0.3, color="blue", bins=50)
df3["score"].hist(alpha=0.3, color="orange", bins=50)
#df2["score"].hist(alpha=0.5, color="green", bins=50)

a = df4["score"].max()
b = df3["score"].max()
c = df2["score"].max()
d = df4["score"].min()
e = df3["score"].min()
f = df2["score"].min()

print(a,b,c,d,e,f)

plt.show()

pass