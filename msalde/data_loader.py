import logging
from pathlib import Path
from pandas import pd
from typing import Tuple, Union

from .model import Variant, AssayResult

logger = logging.getLogger(__name__)


class VariantDataLoader:

    def __init__(self, input_path: str):
        self._input_path = Path(input_path)

    def load(self) -> Tuple[list[Variant], list[AssayResult]]:
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
        required_cols = ["sequence", "score"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain '{col}' column")

        # Get optional columns
        id_col = "id" if "id" in df.columns else None
        name_col = "name" if "name" in df.columns else None
        uncertainty_col = "uncertainty" if "uncertainty" in df.columns else None

        variants = []
        results = []

        for i, row in df.iterrows():
            variant_id = row[id_col] if id_col else i + 1
            variant_name = row[name_col] if name_col else f"variant_{variant_id}"

            variant = Variant(
                id=variant_id,
                name=variant_name,
                sequence=row["sequence"],
            )

            result = AssayResult(
                variant_id=variant_id,
                score=float(row["score"]),
                uncertainty=float(row[uncertainty_col]) if uncertainty_col and not pd.isna(
                    row[uncertainty_col]) else None,
            )

            variants.append(variant)
            results.append(result)

        return variants, results


