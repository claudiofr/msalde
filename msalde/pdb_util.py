def extract_domain_like_regions(cif_dict, gap_threshold=5):
    """
    Extracts domain-like regions from an mmCIF file by merging
    helices and strands into continuous structured blocks.

    Parameters
    ----------
    cif_file : str
        Path to the mmCIF file.
    gap_threshold : int
        Maximum allowed gap (in residues) between structured segments
        before starting a new domain block.

    Returns
    -------
    list of (start, end)
        List of merged structured regions representing domain-like blocks.
    """

    # --- Extract helices ---
    helix_start = cif_dict.get("_struct_conf.beg_auth_seq_id", [])
    helix_end   = cif_dict.get("_struct_conf.end_auth_seq_id", [])

    helices = []
    for s, e in zip(helix_start, helix_end):
        try:
            helices.append((int(s), int(e)))
        except ValueError:
            pass

    # --- Extract strands ---
    sheet_start = cif_dict.get("_struct_sheet_range.beg_auth_seq_id", [])
    sheet_end   = cif_dict.get("_struct_sheet_range.end_auth_seq_id", [])

    strands = []
    for s, e in zip(sheet_start, sheet_end):
        try:
            strands.append((int(s), int(e)))
        except ValueError:
            pass

    # Combine all structured segments
    segments = helices + strands
    if not segments:
        return []

    # Sort by start residue
    segments.sort(key=lambda x: x[0])

    # Merge overlapping or close segments
    merged = []
    for start, end in segments:
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + gap_threshold:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])

    # Convert inner lists to tuples
    return [(s, e) for s, e in merged]


def build_residue_number_map(cif):
    """
    Build a mapping between mmCIF internal numbering (label_seq_id)
    and author numbering (auth_seq_id).
    """


    label_seq = cif["_pdbx_poly_seq_scheme.seq_id"]
    auth_seq  = cif["_pdbx_poly_seq_scheme.auth_seq_num"]
    aa        = cif["_pdbx_poly_seq_scheme.mon_id"]
    chain     = cif["_pdbx_poly_seq_scheme.asym_id"]

    # Build lookup dictionary
    mapping = {int(label_seq[i]): int(auth_seq[i])
               for i in range(len(label_seq))
               if auth_seq[i] != "?"}
    return mapping



def collapse_secondary_struct_code(code):
    if code in ("H", "G", "I"):
        return "H"   # helix
    if code in ("E", "B"):
        return "E"   # strand
    return "C"       # coil


def extract_dssp_from_cif(cif):

    chain  = cif["_dssp_struct_summary.label_asym_id"]
    resid  = cif["_dssp_struct_summary.label_seq_id"]
    aa     = cif["_dssp_struct_summary.label_comp_id"]
    ss     = cif["_dssp_struct_summary.secondary_structure"]
    acc    = cif["_dssp_struct_summary.accessibility"]

    dssp = []
    for i in range(len(chain)):
        dssp.append({
            "chain": chain[i],
            "residue_num": int(resid[i]),
            "aa": aa[i],
            "ss": ss[i],
            "accessibility": float(acc[i])
        })


    return dssp

