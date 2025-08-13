from .model import Variant
import numpy as np


class ProteinEmbedder:
    def embed_sequences(self, sequences: list[str]) -> np.ndarray:
        pass

    def embed_variants(self, variants: list[Variant]) -> list[Variant]:
        embeddings = self.embed_sequences([variant.sequence for variant in
                                          variants])
        vars = [Variant(id=var.id, name=var.name, sequence=var.sequence,
                        embedding=embedding) for var, embedding in
                zip(variants, embeddings)]
        return vars
