from typing import Optional
from .model import Variant
import numpy as np


class ProteinEmbedder:

    def _embed_variants(self, variants: list[Variant]) -> np.ndarray:
        pass

    def embed_variants(self, variants: list[Variant]) -> list[Variant]:
        embeddings = self._embed_variants(variants)
        vars = [Variant(id=var.id, name=var.name, sequence=var.sequence,
                        embedding=embedding) for var, embedding in
                zip(variants, embeddings)]
        return vars


class ProteinEmbedderFactory:

    # This method should be overridden by subclasses
    def create_instance(self, embedder_config, dataset_config) -> ProteinEmbedder:
        raise NotImplementedError("This method should be overridden by subclasses")
