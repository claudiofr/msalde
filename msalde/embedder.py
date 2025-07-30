"""Embedding utilities for the PLM Framework."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from plm_framework.datamodels import Variant

logger = logging.getLogger(__name__)


class ProteinEmbedder:
    """Protein embedder using ESM models."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        batch_size: int = 8,
        use_mean_pooling: bool = True,
        cache_dir: Optional[str] = None,
        quantize: bool = False,
    ):
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
        self._model_name = model_name
        self._batch_size = batch_size
        self._use_mean_pooling = use_mean_pooling

        # Set device
        self._device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)

        # Load model with quantization if requested
        if quantize:
            try:
                import bitsandbytes as bnb

                if self._device == "cuda":
                    # Load in 8-bit precision
                    self._model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        load_in_8bit=True,
                        device_map="auto",
                    )
                    logger.info(f"Loaded model {model_name} with 8-bit quantization")
                else:
                    # Load in FP16 for CPU
                    self._model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                    ).to(self._device)
                    logger.info(f"Loaded model {model_name} with FP16 quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, falling back to full precision"
                )
                self._model = AutoModel.from_pretrained(
                    model_name, cache_dir=cache_dir
                ).to(self._device)
        else:
            # Load in full precision
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(
                self._device
            )

        # Get embedding dimension
        self._embedding_dim = self.model.config.hidden_size

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Embed a single protein sequence.

        Args:
            sequence: Protein sequence

        Returns:
            Embedding vector
        """
        # Tokenize sequence
        inputs = self._tokenizer(sequence, return_tensors="pt").to(self._device)

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get last hidden state
        embeddings = outputs.last_hidden_state

        # Apply pooling if requested
        if self._use_mean_pooling:
            # Mean pooling (excluding special tokens)
            attention_mask = inputs["attention_mask"]
            embeddings = torch.sum(
                embeddings * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            # Use CLS token
            embeddings = embeddings[:, 0]

        return embeddings.cpu().numpy()[0]

    def embed_variants(self, variants: list[Variant]) -> \
            Tuple[list[int], np.ndarray]:
        """
        Embed a list of protein variants.

        Args:
            variants: List of variants to embed

        Returns:
            Tuple of (variant_ids, embeddings)
        """

        # Process variants in batches
        all_variant_ids = []
        all_embeddings = []

        for i in range(0, len(variants), self.batch_size):
            batch = variants[i : i + self.batch_size]

            # Extract sequences and IDs
            sequences = [v.sequence for v in batch]
            variant_ids = [v.id for v in batch]

            # Skip empty sequences
            valid_indices = [
                i for i, seq in enumerate(sequences) if seq and len(seq) > 0
            ]
            if not valid_indices:
                logger.warning(
                    f"Batch {i//self.batch_size} contains no valid sequences"
                )
                continue

            valid_sequences = [sequences[i] for i in valid_indices]
            valid_variant_ids = [variant_ids[i] for i in valid_indices]

            # Embed valid sequences
            try:
                batch_embeddings = self.embed_sequences(valid_sequences)
                all_variant_ids.extend(valid_variant_ids)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                logger.error(f"Sequences: {valid_sequences[:2]}...")
                import traceback

                logger.error(traceback.format_exc())

        # Check if we have any embeddings
        if not all_embeddings:
            logger.warning("No valid embeddings were generated")
            return [], np.array([])

        # Stack embeddings
        embeddings = np.vstack(all_embeddings)

        return all_variant_ids, embeddings

    def embed_sequences(self, sequences: List[str]) -> List[np.ndarray]:
        """
        Embed a list of protein sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            List of embedding vectors
        """
        # Check if sequences list is empty
        if not sequences:
            logger.warning("Empty sequences list provided to embed_sequences")
            return []

        # Filter out empty sequences
        valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]
        if not valid_sequences:
            logger.warning("No valid sequences to embed")
            return []

        # Process sequences in batches
        all_embeddings = []

        for i in range(0, len(valid_sequences), self.batch_size):
            batch = valid_sequences[i : i + self.batch_size]

            # Tokenize batch
            try:
                inputs = self.tokenizer(batch, padding=True, return_tensors="pt").to(
                    self.device
                )

                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Get last hidden state
                embeddings = outputs.last_hidden_state

                # Apply pooling if requested
                if self.use_pooling:
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

            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                logger.error(f"Sequences: {batch[:2]}...")
                import traceback

                logger.error(traceback.format_exc())

        return all_embeddings


def embed_sequences(
    variants: List[Variant],
    output_path: Union[str, Path],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    device: Optional[str] = None,
    use_pooling: bool = True,
    quantize: bool = False,
    append: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Embed protein sequences and save to file.

    Args:
        variants: List of variants to embed
        output_path: Path to save embeddings
        model_name: Name of the ESM model to use
        batch_size: Batch size for embedding
        device: Device to run the model on
        use_pooling: Whether to use mean pooling
        quantize: Whether to quantize the model
        append: Whether to append to existing embedding file
        cache_dir: Directory to cache models
    """
    # Ensure device is a string or None
    device_str = device if isinstance(device, str) or device is None else None

    embedder = ProteinEmbedder(
        model_name=model_name,
        device=device_str,
        batch_size=batch_size,
        use_pooling=use_pooling,
        quantize=quantize,
        cache_dir=cache_dir,
    )

    # Embed variants
    variant_ids, embeddings = embedder.embed_variants(variants)

    # Save embeddings
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    if append and output_path.exists():
        # Append to existing file
        with h5py.File(output_path, "a") as f:
            # Get existing data
            existing_ids = f["variant_ids"][:]
            existing_embeddings = f["embeddings"][:]

            # Check for duplicates
            new_ids = []
            new_embeddings = []
            for i, vid in enumerate(variant_ids):
                if vid not in existing_ids:
                    new_ids.append(vid)
                    new_embeddings.append(embeddings[i])

            if not new_ids:
                logger.info("No new variants to embed")
                return

            # Combine data
            combined_ids = np.concatenate([existing_ids, new_ids])
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])

            # Delete existing datasets
            del f["variant_ids"]
            del f["embeddings"]

            # Create new datasets
            f.create_dataset("variant_ids", data=combined_ids)
            f.create_dataset("embeddings", data=combined_embeddings)

            logger.info(
                f"Appended embeddings for {len(new_ids)} variants to {output_path}"
            )
    else:
        # Create new file
        with h5py.File(output_path, "w") as f:
            # Create datasets
            f.create_dataset("variant_ids", data=variant_ids)
            f.create_dataset("embeddings", data=embeddings)

            # Add metadata
            f.attrs["model_name"] = model_name
            f.attrs["embedding_dim"] = embedder.embedding_dim
            f.attrs["use_pooling"] = use_pooling

        logger.info(
            f"Saved embeddings for {len(variant_ids)} variants to {output_path}"
        )


def load_embeddings(path: Union[str, Path]) -> Tuple[List[int], np.ndarray]:
    """
    Load protein embeddings from a file.

    Args:
        path: Path to the embedding file

    Returns:
        Tuple of (variant_ids, embeddings)
    """
    with h5py.File(path, "r") as f:
        variant_ids = f["variant_ids"][:]
        embeddings = f["embeddings"][:]

        # Convert variant_ids to list of integers
        variant_ids = [int(id) for id in variant_ids]

        return variant_ids, embeddings
