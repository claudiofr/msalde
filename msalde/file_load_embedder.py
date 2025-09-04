import numpy as np
import pandas as pd
from .embedder import ProteinEmbedder, ProteinEmbedderFactory
from .model import Variant

class FileLoadEmbedder(ProteinEmbedder):
    """
    Placeholder for a specific PLM model implementation.
    This class should implement the methods required by the PLMModel interface.
    """

    def __init__(self, config):
        """
        Initialize the protein embedder.

        Args:
            model_name: Name of the ESM model to use
            device: Device to run the model on
            batch_size: Batch size for embedding
            use_pooling: Whether to use mean pooling
            cache_dir: Directory to cache models
            quantize: Whether to quantize the model to FP16/Int8
        """
        self._embeddings_file = config.embeddings_file

    def _embed_variants(self, variants: list[Variant]) -> list[np.ndarray]:
        """
        Embed protein sequences from a file.

        Args:
            variants: List of Variant objects

        Returns:
            List of embeddings for each variant
        """
        # Load embeddings from file
        embeddings = pd.read_csv(self._embeddings_file, index_col=0)
        embeddings = embeddings.loc[[variant.id for variant in variants]]
        if len(embeddings) != len(variants):
            raise ValueError("Number of embeddings does not match number of variants")
        return embeddings.to_numpy()


class FileLoadEmbedderFactory(ProteinEmbedderFactory):

    def create_instance(self, config) -> FileLoadEmbedder:
        return FileLoadEmbedder(config)