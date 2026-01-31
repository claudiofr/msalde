import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from Bio import SeqIO


from .data_loader import VariantDataLoader, VariantDataLoaderFactory

logger = logging.getLogger(__name__)


class VariantDataFileLoaderFastaOnly(VariantDataLoader):
    """
    Variant data loader for datasets where we only have a FASTA file.
    Assay results are not available. We would use this in the case
    where we only want to generate log likelihood ratios using the
    ESM2LogLikelihoodLearner learner.
    """
    def __init__(self, config):
        super().__init__(config.column_names)
        self._wild_type_id = config.wild_type_id
        self._fasta_file = config.fasta_file

    def load_assay_data(self) -> Tuple[pd.DataFrame, str, float]:
        """
        Load assay results from CSV file.

        Args:
            input_path: Path to input file

        Returns:
            Tuple of (variants, results)
        """
        # Load CSV file
        wt_sequence = None
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(self._fasta_file, "fasta")}
        df = pd.DataFrame.from_dict(fasta_dict, orient="index", columns=["sequence"])
        df.reset_index(drop=False, inplace=True)
        id_col = self._column_name_mapping.get("id_col", "variant")
        df.rename(columns={"index": id_col}, inplace=True)
        wt_sequence = fasta_dict.get(self._wild_type_id)
        df = df[df[id_col] != self._wild_type_id]

        return df, wt_sequence, None


class VariantDataFileLoaderFastaOnlyFactory(VariantDataLoaderFactory):

    def create_instance(self, config) -> VariantDataFileLoaderFastaOnly:
        return VariantDataFileLoaderFastaOnly(config)
