import numpy as np

from .model import Variant
from .embedder import ProteinEmbedder, ProteinEmbedderFactory
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


class ESMEmbedder(ProteinEmbedder):
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
        self._model_name = config.model_name
        config = config.parameters
        self._batch_size = config.batch_size
        self._use_pooling = config.use_pooling

        # Set device
        self._device = (
            config.device if config.device else ("cuda" if torch.cuda.is_available()
                                                 else "cpu")
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, cache_dir=config.cache_dir)

        # Load model with quantization if requested
        if config.quantize:
            import bitsandbytes as bnb
            if self._device == "cuda":
                # Load in 8-bit precision
                self._model = AutoModel.from_pretrained(
                    self._model_name,
                    cache_dir=config.cache_dir,
                    load_in_8bit=True,
                    device_map="auto",
                )
            else:
                # Load in FP16 for CPU
                self._model = AutoModel.from_pretrained(
                    self._model_name,
                    cache_dir=config.cache_dir,
                    torch_dtype=torch.float16,
                ).to(self._device)
        else:
            # Load in full precision
            self._model = AutoModel.from_pretrained(
                self._model_name, cache_dir=config.cache_dir).to(
                self._device
            )

        # Get embedding dimension
        self._embedding_dim = self._model.config.hidden_size

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Embed a single protein sequence.

        Args:
            sequence: Protein sequence

        Returns:
            Embedding vector
        """
        # Tokenize sequence
        inputs = self._tokenizer(sequence, return_tensors="pt").to(
            self._device)

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get last hidden state
        embeddings = outputs.last_hidden_state

        # Apply pooling if requested
        if self._use_pooling:
            # Mean pooling (excluding special tokens)
            attention_mask = inputs["attention_mask"]
            embeddings = torch.sum(
                embeddings * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            # Use CLS token
            embeddings = embeddings[:, 0]

        return embeddings.cpu().numpy()[0]

    def embed_sequences(self, sequences: list[str]) -> list[np.ndarray]:
        """
        Embed a list of protein sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            List of embedding vectors
        """

        # Process sequences in batches
        all_embeddings = []

        for i in range(0, len(sequences), self._batch_size):
            batch = sequences[i:i + self._batch_size]

            inputs = self._tokenizer(batch, padding=True, return_tensors="pt").to(
                self._device
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get last hidden state
            embeddings = outputs.last_hidden_state

            # Apply pooling if requested
            if self._use_pooling:
                # Mean pooling (excluding special tokens)
                attention_mask = inputs["attention_mask"]
                embeddings = torch.sum(
                    embeddings * attention_mask.unsqueeze(-1), dim=1
                ) / torch.sum(attention_mask, dim=1, keepdim=True)
            else:
                # Use CLS token
                embeddings = embeddings[:, 0]

            # Convert to numpy and add to list
            batch_embeddings = embeddings.cpu().numpy()
            all_embeddings.extend(
                [batch_embeddings[j] for j in range(batch_embeddings.shape[0])]
            )

        return all_embeddings

    def embed_variants(self, variants: list[Variant]) -> list[np.ndarray]:
        """
        Embed a list of protein variants.

        Args:
            variants: List of Variant objects

        Returns:
            List of embedding vectors
        """
        sequences = [variant.sequence for variant in variants]
        embeddings = self.embed_sequences(sequences)
        return embeddings


class ESMEmbedderFactory(ProteinEmbedderFactory):

    def create_instance(self, embedder_config, dataset_config) -> ESMEmbedder:
        return ESMEmbedder(embedder_config)