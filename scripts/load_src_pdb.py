import context
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from msalde.container import ALDEContainer


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

    mapping = []
    for i in range(len(label_seq)):
        try:
            mapping.append({
                "label_seq_id": int(label_seq[i]),
                "auth_seq_id": int(auth_seq[i]),
                "aa": aa[i],
                "chain": chain[i]
            })
        except ValueError:
            # Some entries may have missing or non-numeric auth_seq_id
            continue

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


cif_file = "/home/claudiof/data/al/pdb/2src.cif"
cif = MMCIF2Dict(cif_file)

regions = extract_domain_like_regions(cif, 1)
print("Domain-like structured regions:")
for i, (start, end) in enumerate(regions, 1):
    print(f"  Region {i}: {start}–{end}")

dssp_data = extract_dssp_from_cif(cif)
residue_number_map = build_residue_number_map(cif)
for row in dssp_data:
    row["ss_simple"] = collapse_secondary_struct_code(row["ss"])
    row["auth_residue_num"] = residue_number_map.get(
        row["residue_num"], None)

"""
	Region 1: ~84–145 → SH3
	Region 2: ~151–248 → SH2
	Region 3: ~267–520 → Kinase domain
	Region 4: ~521–533 → C terminal tail
"""
chainA = [r for r in dssp_data if r["chain"] == "A"]
chainA.sort(key=lambda r: r["residue_num"])

ss_track = [r["ss_simple"] for r in chainA]
residue_nums = [r["auth_residue_num"] for r in chainA]

domains = [
    ("SH3",   84, 145, "cornflowerblue"),
    ("SH2",  151, 248, "seagreen"),
    ("Kinase", 267, 520, "orange"),
    ("Tail", 521, 533, "gray")
]

def plotit(domains, residue_nums, ss_track):
    import matplotlib.pyplot as plt
    import numpy as np

    color_map = {"H": "red", "E": "gold", "C": "lightgray"}
    colors = [color_map[c] for c in ss_track]

    """
    plt.figure(figsize=(14, 1.5))
    plt.bar(residue_nums, [1]*len(residue_nums), color=colors, width=1.0)
    for name, start, end, color in domains:
        plt.barh(1.2, end-start, left=start, color=color, alpha=0.6)
        plt.text((start+end)/2, 1.25, name, ha="center", va="bottom")
    plt.yticks([])
    plt.xlabel("Residue number")
    plt.title("Secondary Structure Map for Src (2SRC)")
    plt.show()
    """

    plt.figure(figsize=(14, 2))
    # Draw domain bands with increased height and place labels inside the bands
    band_height = 0.5  # Height of the domain bands
    for (name, start, end, color) in domains:
        plt.axvspan(start, end, ymin=0.25, ymax=0.75, color=color, alpha=0.3)
        plt.text((start + end) / 2, band_height, name, ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw secondary structure bars
    plt.bar(residue_nums, [0.2]*len(residue_nums), color=colors, width=1.0, bottom=0.4)

    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Residue number")
    plt.title("Secondary Structure Map for Src (2SRC)")
    plt.tight_layout()
    plt.show()

def plotit1(domains, residue_nums, ss_track):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    color_map = {"H": "red", "E": "gold", "C": "lightgray"}
    colors = [color_map[c] for c in ss_track]

    plt.figure(figsize=(14, 2))

    # Draw secondary structure bars first (so they are below the bands)
    plt.bar(residue_nums, [0.2]*len(residue_nums), color=colors, width=1.0, bottom=0.1, zorder=1)

    # Draw domain bands with increased height and place labels inside the bands
    band_height = 0.7  # Height of the domain bands
    for (name, start, end, color) in domains:
        plt.axvspan(start, end, ymin=0.5, ymax=0.75, color=color, alpha=0.3, zorder=2)
        plt.text((start + end) / 2, band_height, name, ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)

    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Residue number")
    plt.title("Secondary Structure Map for Src (2SRC)")
    plt.tight_layout()

    # Create legend for secondary structure colors
    legend_patches = [
        mpatches.Patch(color=color_map["H"], label="Helix"),
        mpatches.Patch(color=color_map["E"], label="Strand"),
        mpatches.Patch(color=color_map["C"], label="Coil"),
    ]
    plt.legend(handles=legend_patches, loc="upper right", title="Secondary Structure")
    plt.show()
               
plotit1(domains, residue_nums, ss_track)

pass