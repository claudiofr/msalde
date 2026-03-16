import numpy as np

from .model import Variant
from .embedder import ProteinEmbedder, ProteinEmbedderFactory
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.mixture import GaussianMixture


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSequenceCompressor(nn.Module):
    """
    Compresses per-residue embeddings (L, d) into a fixed-length vector
    using multi-kernel 1D CNN + global max pooling.
    """
    def __init__(self, embed_dim, num_filters=128, kernel_sizes=(3, 5, 7)):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2   # keep length same
            )
            for k in kernel_sizes
        ])
        
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x, mask=None):
        """
        x: (batch, L, d)
        mask: (batch, L) with 1 for valid residues, 0 for padding
        """
        # Convert to (batch, d, L) for Conv1d
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            h = conv(x)              # (batch, num_filters, L)
            h = F.relu(h)

            if mask is not None:
                # mask: (batch, L) → (batch, 1, L)
                m = mask.unsqueeze(1)
                h = h.masked_fill(m == 0, float('-inf'))

            # Global max pooling over sequence length
            pooled = torch.max(h, dim=2).values  # (batch, num_filters)
            conv_outputs.append(pooled)

        # Concatenate pooled outputs
        return torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))


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
        if self._compression_method == "window_pooling":
            self._compression_num_segments = compression_config.get("num_segments", 1)
            self._compression_overlap_fraction = compression_config.get(
                "overlap_fraction", 0.0)
            self._compression_pool_method = compression_config.get(
                "pool_method", "mean")
        elif self._compression_method == "cnn":
            self._cnn_compressor = CNNSequenceCompressor(
                embed_dim=self._embedding_dim,
                num_filters=compression_config.get("num_filters", 128),
                kernel_sizes=compression_config.get("kernel_sizes", [3, 5, 7])
            )
        elif self._compression_method == "fisher_vector":
            self._normalize = compression_config.get(
                "normalize", True)
            self._num_gaussian_mixture_components = compression_config.get(
                "num_gaussian_mixture_components", 64)
            self._pca_dim = compression_config.get("pca_dim", 64)
            self._random_state = compression_config.get(
                "random_state", 42)

    def _fit_gaussian_mixture_model(
        self,
        embeddings: np.ndarray,
        padding_mask: np.ndarray,
        n_components: int = 64,
        pca_dim: int | None = 64,
        random_state: int = 42,
    ) -> tuple[GaussianMixture, object | None]:
        """
        Fit a diagonal-covariance GMM on pooled residue embeddings.

        Args:
            embeddings:    (B, L, D) padded residue embedding array.
            padding_mask:  (B, L) bool array — True for real residues.
            n_components:  Number of Gaussian components (K).
            pca_dim:       Reduce to this many dimensions before GMM fitting.
                        Strongly recommended for high-D ESM2 embeddings.
                        Set to None to skip PCA.
            random_state:  RNG seed.

        Returns:
            gmm: Fitted GaussianMixture object.
            pca: Fitted PCA object, or None if pca_dim is None.
        """
        from sklearn.decomposition import PCA

        # Flatten to (N_real_residues, D) using the mask
        all_residues = embeddings[padding_mask]   # boolean index → (N, D)
        print(f"Fitting GMM on {all_residues.shape[0]:,} residue embeddings "
            f"of dimension {all_residues.shape[1]}")

        pca = None
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim, random_state=random_state)
            all_residues = pca.fit_transform(all_residues)
            print(f"PCA variance explained: "
                f"{pca.explained_variance_ratio_.sum():.3f}")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=500,
            random_state=random_state,
            verbose=1,
        )
        gmm.fit(all_residues)
        print(f"GMM fitted. Final lower bound: {gmm.lower_bound_:.4f}")

        return gmm, pca

    def _compute_fisher_vector(
        self,
        residues: np.ndarray,
        gmm: GaussianMixture,
        normalized: bool = True,
    ) -> np.ndarray:
        """
        Compute the Fisher Vector for a single protein (no padding).

        Args:
            residues:   (L, D) real residue embeddings (padding already removed).
            gmm:        Fitted GaussianMixture (covariance_type="diag").
            normalized: Apply power normalisation then L2 normalisation.

        Returns:
            fv: 1D Fisher Vector of length 2 * K * D.
        """
        L = residues.shape[0]
        K = gmm.n_components
        pi  = gmm.weights_       # (K,)
        mu  = gmm.means_         # (K, D)
        sig = gmm.covariances_   # (K, D)

        gamma = gmm.predict_proba(residues)   # (L, K)

        diff = residues[:, np.newaxis, :] - mu[np.newaxis, :, :]   # (L, K, D)
        g    = gamma[:, :, np.newaxis]                              # (L, K, 1)

        s1 = (g * diff / sig[np.newaxis]).sum(axis=0)                    # (K, D)
        s2 = (g * (diff ** 2 / sig[np.newaxis] - 1)).sum(axis=0)         # (K, D)

        sqrt_pi = np.sqrt(pi)[:, np.newaxis]   # (K, 1)
        s1 = s1 / (L * sqrt_pi)
        s2 = s2 / (L * sqrt_pi)

        fv = np.concatenate([s1.ravel(), s2.ravel()])   # (2*K*D,)

        if normalized:
            fv = np.sign(fv) * np.sqrt(np.abs(fv))      # power normalisation
            norm = np.linalg.norm(fv)
            if norm > 0:
                fv = fv / norm                           # L2 normalisation

        return fv

    def fisher_vector_encode_embeddings(
        self,
        embeddings,        # (B, L, D)
        attention_masks,   # (B, L, 1)
        normalize: bool = True,
        num_components: int = 64,
        pca_dim: int | None = 64,
        random_state: int = 42,
    ):
        """
        Adaptive window pooling:
        - Uses each sequence's true length (from attention mask)
        - Computes window boundaries proportionally per sequence
        - Supports mean or max pooling
        - Returns (B, num_segments * D)
        """

        B, L, D = embeddings.shape
        attention_masks = attention_masks.squeeze(-1).bool()  # (B, L)
        gmm, pca = self._fit_gaussian_mixture_model(
            embeddings,
            attention_masks,
            n_components=num_components,
            pca_dim=pca_dim,
            random_state=random_state
        )
        B = embeddings.shape[0]
        fisher_vectors = []

        for i in range(B):
            residues = embeddings[i, attention_masks[i], :]   # (L_i, D) — no padding
            if pca is not None:
                residues = pca.transform(residues)
            fv = self._compute_fisher_vector(residues, gmm,
                                             normalized=normalize)
            fisher_vectors.append(fv)
            if (i + 1) % 100 == 0:
                print(f"Encoded {i + 1}/{B} proteins")

        return torch.Tensor(np.vstack(fisher_vectors))   # (B, 2*K*D)

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
        elif self._compression_method == "cnn":
            # CNN-based compression
            embeddings = self._cnn_compressor(embeddings,
                                              mask=inputs["attention_mask"])
        elif self._compression_method == "fisher_vector":
            # Fisher vector compression
            embeddings = self.fisher_vector_encode_embeddings(
                embeddings,
                attention_masks=attention_masks,
                normalize=self._normalize,
                num_components=self._num_gaussian_mixture_components,
                pca_dim=self._pca_dim,
                random_state=self._random_state
            )
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
            elif self._compression_method == "cnn":
                # CNN-based compression
                embeddings = self._cnn_compressor(
                    embeddings, mask=inputs["attention_mask"])
            elif self._compression_method == "fisher_vector":
                # Fisher vector compression
                embeddings = self.fisher_vector_encode_embeddings(
                    embeddings,
                    attention_masks=attention_mask,
                    normalize=self._normalize,
                    num_components=self._num_gaussian_mixture_components,
                    pca_dim=self._pca_dim,
                    random_state=self._random_state
                )
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