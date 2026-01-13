import re

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


def parse_hgvs_protein_string(s):
    match = re.match(r'^([A-Za-z]+)(\d+)([A-Za-z]+)$', s)
    if match:
        leading, number, trailing = match.groups()
        return leading, int(number), trailing
    else:
        return None


def hgvs_protein_to_variant_id(protein_change: str) -> str:
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

        ref_aa3, position, alt_aa3 = parse_hgvs_protein_string(mutation_part)
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
