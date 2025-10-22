import os
import re
import pandas as pd
from pathlib import Path

amino_acids = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V"
}


def parse_protein_mutation_string(s):
    match = re.match(r'^([A-Za-z]+)(\d+)([A-Za-z]+)$', s)
    if match:
        leading, number, trailing = match.groups()
        return leading, int(number), trailing
    else:
        return None


def extract_variant_id(protein_change: str) -> str:
    """
    Extracts the variant ID from a protein change string.

    Parameters
    ----------
    protein_change : str
        The protein change string (e.g., "MAGOHB:p.Leu9Gln").

    Returns
    -------
    str
        The extracted variant ID (e.g., "L9Q").
    """
    try:
        # Split the string to get the mutation part
        mutation_part = protein_change.split('.')[1]

        ref_aa3, position, alt_aa3 = parse_protein_mutation_string(mutation_part)
        if alt_aa3 == "Ter":
            return None
        ref_aa1 = amino_acids.get(ref_aa3)
        alt_aa1 = amino_acids.get(alt_aa3)
        if ref_aa1 is None or alt_aa1 is None:
            raise ValueError(f"Unknown amino acid code in {mutation_part}")

        variant_id = f"{ref_aa1}{position}{alt_aa1}"
        return variant_id
    except Exception as e:
        print(f"Error extracting variant ID from {protein_change}: {e}")
        return None

def extract_label(clinvar_category: str) -> int:
    """
    Extracts a binary label from the ClinVar variant category.

    Parameters
    ----------
    clinvar_category : str
        The ClinVar variant category (e.g., "Pathogenic", "Benign").

    Returns
    -------
    int
        The extracted label (1 for pathogenic, 0 for benign, -1 for others).
    """
    pathogenic_categories = {"LB", "B"}
    benign_categories = {"LP", "P"}

    if clinvar_category in pathogenic_categories:
        return 1
    elif clinvar_category in benign_categories:
        return 0
    else:
        return -1

dir = Path('/home/claudiof/gitrepo/msalde/data/clinvar')

df = pd.read_csv(dir / 'variant_summary_MAVEN.tsv', sep='\t')

df = df[(df['Stars'] >= 1) & (df['ClinVar Variant Category'] != 'US')]
df['variant_id'] = df['ProteinChange'].apply(extract_variant_id)
df = df[df['variant_id'].notna()]
df['label'] = df['ClinVar Variant Category'].apply(
    lambda v: 1 if v == 'LP/P' else 0)

df['gene_symbol'] = df['GeneSymbol'].fillna('')

df[['gene_symbol', 'variant_id', 'label']].to_csv(
    dir / 'clinvar_labels.csv', index=False)