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
        compression_config = config.get("compression", {})
        self._compression_method = compression_config.get("method", "none")
        self._compression_num_segments = compression_config.get("num_segments", 1)
        self._compression_overlap_fraction = compression_config.get(
            "overlap_fraction", 0.0)
        self._compression_pool_method = compression_config.get(
            "pool_method", "mean")

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

    def window_pool_embeddings(
        self,
        embeddings,        # (B, L, D)
        attention_masks,   # (B, L, 1)
        num_segments=8,
        overlap_frac=0.5,
        pool="mean"
    ):
        """
        Adaptive window pooling:
        - Uses each sequence's true length (from attention mask)
        - Computes window boundaries proportionally per sequence
        - Supports mean or max pooling
        - Returns (B, num_segments * D)
        """

        B, L, D = embeddings.shape

        # True lengths per sequence: (B,)
        true_lengths = attention_masks.squeeze(-1).sum(dim=1)

        # Compute window sizes per sequence: (B,)
        window_sizes = (true_lengths // num_segments).clamp(min=1)

        # Overlap and stride per sequence: (B,)
        overlaps = (window_sizes * overlap_frac).long()
        strides = (window_sizes - overlaps).clamp(min=1)

        pooled_segments = []

        for seg in range(num_segments):
            # Compute start and end indices for each sequence: (B,)
            starts = seg * strides
            ends = starts + window_sizes

            # Clamp to true lengths
            ends = torch.minimum(ends, true_lengths)

            # Build a mask of shape (B, L) indicating which positions fall in the window
            idxs = torch.arange(L, device=embeddings.device).unsqueeze(0)  # (1, L)
            starts_exp = starts.unsqueeze(1)  # (B, 1)
            ends_exp = ends.unsqueeze(1)      # (B, 1)

            # Window mask: True where start <= idx < end
            window_mask = (idxs >= starts_exp) & (idxs < ends_exp)  # (B, L)

            # Combine with attention mask to exclude padding
            window_mask = window_mask & (attention_masks.squeeze(-1).bool())  # (B, L)

            # Expand to (B, L, D)
            window_mask_exp = window_mask.unsqueeze(-1)

            if pool == "mean":
                masked = embeddings * window_mask_exp
                summed = masked.sum(dim=1)                     # (B, D)
                counts = window_mask_exp.sum(dim=1).clamp(min=1e-6)  # (B, 1)
                pooled_vec = summed / counts                   # (B, D)

            elif pool == "max":
                neg_inf = torch.finfo(embeddings.dtype).min
                masked = embeddings.masked_fill(~window_mask_exp, neg_inf)
                pooled_vec = masked.max(dim=1).values          # (B, D)

            else:
                raise ValueError("pool must be 'mean' or 'max'")

            pooled_segments.append(pooled_vec)

        # (B, num_segments, D)
        pooled = torch.stack(pooled_segments, dim=1)

        # Flatten to (B, num_segments * D)
        return pooled.reshape(B, num_segments * D)


    def window_pool_embeddings1(self, embeddings, attention_masks, 
                               num_segments=8, overlap_frac=0.5, pool="mean"):
        """
        Pool embeddings using a sliding window approach.
        Ignores attention_masks parameters for now, but can 
        be used to mask out padding tokens in the future.
        For now we assume that all sequences are of the 
        same length and that there is no padding.

        embeddings: Tensor of shape (N, L, d)
        attention_masks: ignored for now
        returns: Tensor of shape (N, num_segments * d)
        """
        N, L, d = embeddings.shape

        # Base window size: exactly L / num_segments
        window_size = max(1, L // num_segments)

        # Overlap in residues
        overlap = int(window_size * overlap_frac)

        # Effective stride
        stride = max(1, window_size - overlap)

        pooled = []

        for i in range(num_segments):
            start = i * stride
            end = start + window_size

            # Clamp to sequence boundaries
            if end > L:
                end = L
                start = max(0, end - window_size)

            # Slice each protein's window: (N, window_size, d)
            window = embeddings[:, start:end, :]

            if pool == "mean":
                pooled_vec = window.mean(dim=1)      # (N, d)
            elif pool == "max":
                pooled_vec = window.max(dim=1).values  # (N, d)
            else:
                raise ValueError("pool must be 'mean' or 'max'")

            pooled.append(pooled_vec)

        # pooled is a list of num_segments tensors, each (N, d)
        pooled = torch.stack(pooled, dim=1)  # (N, num_segments, d)

        # Flatten segments if you want a single vector per protein
        return pooled.reshape(N, num_segments * d)

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
        attention_masks = inputs["attention_mask"].unsqueeze(-1)  # Shape: (1, seq_len, 1)
        if self._compression_method == "window_pooling":
            # Window pooling
            embeddings = self.window_pool_embeddings(
                embeddings,
                attention_masks=attention_masks,
                num_segments=self._compression_num_segments,
                overlap_frac=self._compression_overlap_fraction,
                pool=self._compression_pool_method
            )
        elif self._compression_method == "mean_pooling":
            # Mean pooling (excluding special tokens)
            embeddings = torch.sum(
                embeddings * attention_masks, dim=1
            ) / torch.sum(attention_masks, dim=1, keepdim=True)
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

            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            # Apply pooling if requested
            if self._compression_method == "window_pooling":
                # Window pooling
                embeddings = self.window_pool_embeddings(
                    embeddings,
                    attention_masks=attention_mask,
                    num_segments=self._compression_num_segments,
                    overlap_frac=self._compression_overlap_fraction,
                    pool=self._compression_pool_method
                )
            elif self._compression_method == "mean_pooling":
                # Mean pooling (excluding special tokens)
                # Mask out padding tokens
                masked_embeddings = embeddings * attention_mask  # (B, L, D)

                # Sum over sequence dimension
                summed = masked_embeddings.sum(dim=1)            # (B, D)

                # Count real tokens per sequence
                counts = attention_mask.sum(dim=1)               # (B, 1)

                # Safe division
                embeddings = summed / counts                     # (B, D)
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