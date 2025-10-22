import os
import pandas as pd


class ALDEExternalRepository:

    _clinvar_labels_df = None

    def __init__(self, config):
        self._config = config
        self._clinvar_dir = config.clinvar_dir

    def get_clinvar_labels_by_gene(self, gene_symbol: str) -> pd.DataFrame:

        if self._clinvar_labels_df is None:
            self._clinvar_labels_df = pd.read_csv(
                os.path.join(self._clinvar_dir, "clinvar_labels.csv"))
        labels_df = self._clinvar_labels_df[
            self._clinvar_labels_df['gene_symbol'] == gene_symbol]
        return labels_df
