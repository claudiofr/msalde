import logging
from pathlib import Path
import pandas as pd
from Bio import SeqIO


from .data_loader import VariantDataLoader, VariantDataLoaderFactory

logger = logging.getLogger(__name__)


class VariantDataFileLoader(VariantDataLoader):

    def __init__(self, config):
        super().__init__(config.column_names)
        self._input_path = Path(config.input_path)
        self._wild_type_id = config.wild_type_id
        if hasattr(config, "fasta_file"):
            self._fasta_file = config.fasta_file
        else:
            self._fasta_file = None

    def load_assay_data(self) -> pd.DataFrame:
        """
        Load assay results from CSV file.

        Args:
            input_path: Path to input file

        Returns:
            Tuple of (variants, results)
        """
        # Load CSV file
        df = pd.read_csv(self._input_path)

        # Check required columns
        column_names = self._column_name_mapping.values()
        for col in column_names:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain '{col}' column")
        id_col = self._column_name_mapping["id_col"]
        df = df[df[id_col] != self._wild_type_id]
        if self._fasta_file:
            df.set_index(id_col, drop=False, inplace=True)
            fasta_dict = {record.id: str(record.seq)
                          for record in SeqIO.parse(self._fasta_file, "fasta")}
            df["sequence"] = pd.Series(fasta_dict)
            df.reset_index(drop=True, inplace=True)

        return df


class VariantDataFileLoaderFactory(VariantDataLoaderFactory):

    def create_instance(self, config) -> VariantDataFileLoader:
        return VariantDataFileLoader(config)
