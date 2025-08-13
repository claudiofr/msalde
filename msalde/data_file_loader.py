import logging
from pathlib import Path
import pandas as pd

from .data_loader import VariantDataLoader

logger = logging.getLogger(__name__)


class VariantDataFileLoader(VariantDataLoader):

    def __init__(self, config):
        super().__init__(config.column_names)
        self._input_path = Path(config.input_path)

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

        return df
