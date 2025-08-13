import logging
import pandas as pd
from typing import Tuple

from .model import Variant, AssayResult

logger = logging.getLogger(__name__)


class VariantDataLoader:

    def __init__(self, column_name_mapping: dict[str, str]):
        """
        Initialize the data loader with a mapping of column names.

        Args:
            column_name_mapping: Mapping of column names in the input file
        """
        self._column_name_mapping = column_name_mapping

    def load_assay_data(self) -> Tuple[pd.DataFrame, dict[str, str]]:
        pass

    def load(self) -> Tuple[list[Variant], list[AssayResult]]:
        """
        Load assay results from CSV file.

        Args:
            input_path: Path to input file

        Returns:
            Tuple of (variants, results)
        """
        assay_data_df = self.load_assay_data()
        id_col = self._column_name_mapping.get("id_col")
        name_col = self._column_name_mapping.get("name_col")
        sequence_col = self._column_name_mapping.get("sequence_col")
        score_col = self._column_name_mapping.get("score_col")
        uncertainty_col = self._column_name_mapping.get("uncertainty_col")
        variants = []
        results = []

        for i, row in assay_data_df.iterrows():
            variant_id = row[id_col] if id_col else i + 1
            variant_name = row[name_col] if name_col \
                else f"variant_{variant_id}"

            variant = Variant(
                id=variant_id,
                name=variant_name,
                sequence=row[sequence_col],
            )

            result = AssayResult(
                variant_id=variant_id,
                score=float(row[score_col]),
                uncertainty=float(row[uncertainty_col]) \
                if uncertainty_col and not pd.isna(
                            row[uncertainty_col]) else None,
                        )

            variants.append(variant)
            results.append(result)

        return variants, results


