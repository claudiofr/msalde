import context  # noqa: F401
import pytest
from msalde.embedding_extractor import EmbeddingExtractor


def test_extract_embeddings_for_dataset(
        embedding_extractor: EmbeddingExtractor):
    
    embedding_extractor.extract_by_dataset_name("cas12f2")
    pass

