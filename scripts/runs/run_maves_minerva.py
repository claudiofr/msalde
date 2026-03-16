import context  # noqa: F401 E402
from scripts.runs.run_maves import run_maves

datasets = [
    "ADRB2",
    "AICDA",
    "BRCA1",
    "BRCA2",
    "CALM1",
    "CAR11",
    "CASP3",
    "CASP7",
    "CBS",
    "GDIA",
    "GRB2",
    "HEM3",
    "HMDH",
    "HXK4",
    "KCNE1",
    "KCNH2",
    "MET",
    "MK01",
    "MSH2",
    "MTHR",
    "NPC1",
    "OTC",
    "P53",
    "PAI1",
    "PPARG",
    "PPM1D",
    "PTEN",
    "RAF1",
    "RASH",
    "S22A1",
    "SC6A4",
    "SCN5A",
    "SERC",
    "SHOC2",
    "SRC",
    "SUMO1",
    "SYUA",
    "TADBP",
    "TPK1",
    "TPOR",
    "UBC9",
    "VKOR1",
    "MC4R",
    "brenan",
    "cas12f",
    "cov2_S",
    "doud",
    "giacomelli",
    "haddox",
    "jones",
    "kelsic",
    "lee",
    "markin",
    "stiffler",
    "zikv_E"]

# minerva change
datasets1 = [
    "ADRB2",
    "AICDA",
]

datasets = [
    "MC4R",
    "HXK4",
    "PTEN",
    "SRC",
]

datasets1 = [
    "ACVRL1",
]
# minerva change
label_dir = "/sc/arion/work/fratac01/data/al/dms"
# label_dir = "/home/claudiof"

if __name__ == "__main__":
    run_maves(label_dir, datasets, "./config/msaldem.yaml", 5)

