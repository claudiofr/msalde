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
    "VKOR1"]

with open("config/work.yaml", "a") as f:
    for dataset in datasets:
        template = f"""
  {dataset}:
    data_loader_type: file_loader
    input_path: /sc/arion/work/fratac01/data/al/dms/{dataset}_labels.csv
    wild_type_id: "WT"
    column_names:
      id_col: "mutant"
      sequence_col: "mutant"
      score_col: "DMS_score"
    embeddings_file: /sc/arion/work/fratac01/data/al/esm/{dataset}_esm2_t33_650M_UR50D.csv
    fasta_file: /sc/arion/work/fratac01/data/al/dms/{dataset}.fasta"""
        f.write(template)
