from typing import Tuple
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from .pdb_util import (
    extract_domain_like_regions,
    extract_dssp_from_cif,
    build_residue_number_map,
    collapse_secondary_struct_code
)


class PdbRepository:
    PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{}.pdb"

    def __init__(self, config: dict):
        self._config = config.get("datasets", {})

    def get_pdb_url(self, pdb_id: str) -> str:
        """Get the download URL for a given PDB ID.

        Args:
            pdb_id (str): The PDB ID.

        Returns:
            str: The download URL for the PDB file.
        """
        return self.PDB_URL_TEMPLATE.format(pdb_id.upper()) 
    
    def download_pdb(self, pdb_id: str, save_path: str) -> None:
        """Download the PDB file for a given PDB ID.

        Args:
            pdb_id (str): The PDB ID.
            save_path (str): The path to save the downloaded PDB file.
        """
        import requests

        url = self.get_pdb_url(pdb_id)
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)

    def get_secondary_structure(self, pdb_id: str) -> Tuple[list, list, list]:
        """Retrieve the secondary structure annotation for a given PDB ID.

        Args:
            pdb_id (str): The PDB ID.

        Returns:
            str: A string representing the secondary structure (H, E, C).
        """

        config = self._config.get(pdb_id)
        pdb_file = config.get("pdb_path")
        domains_config = config.get("domains")
        domains = []
        for domain in domains_config:
            domains.append({
                "name": domain["name"],
                "start": domain["start"],
                "end": domain["end"],
                "color": domain["color"]
            })


        cif = MMCIF2Dict(pdb_file)

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

        return ss_track, residue_nums, domains

