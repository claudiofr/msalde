import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

from .model import ModelPrediction, Variant


class Learner:
    """Base class for learners."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        random_state: Optional[int] = None,
        scale_embeddings: bool = False,
    ):
        """
        Initialize the base learner.

        Args:
            embedding_dim: Dimension of input embeddings
            reduced_dim: Dimension to reduce embeddings to (None for no reduction)
            random_state: Random seed
        """
        self._input_dim = input_dim
        self._random_state = random_state

        # Initialize PCA for dimensionality reduction
        self._pca = None
        if self._input_dim is not None:
            self._pca = PCA(n_components=self._input_dim,
                            random_state=self._random_state)
        self._scaler = None
        if scale_embeddings:
            self._scaler = StandardScaler()

    def _fit_transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Reduced embeddings [n_samples, reduced_dim]
        """
        if self._pca is None:
            return embeddings

        # Adjust n_components to be no larger than the number of samples
        num_samples = embeddings.shape[0]
        if self._input_dim > num_samples - 1:
            return embeddings

        # Fit PCA
        return self._pca.fit_transform(embeddings)

    def _transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Reduced embeddings [n_samples, reduced_dim]
        """
        if self._pca is None or not hasattr(self._pca, 'components_'):
            return embeddings

        return self._pca.transform(embeddings)

    def _fit_scale_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Scale embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Scaled embeddings [n_samples, embedding_dim]
        """
        if self._scaler is None:
            return embeddings
        return self._scaler.fit_transform(embeddings)

    def _scale_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Scale embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Scaled embeddings [n_samples, embedding_dim]
        """
        if self._scaler is None:
            return embeddings
        return self._scaler.transform(embeddings)

    def fit_model(
        self,
        variants: list[Variant],
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def fit(
        self,
        variants: list[Variant],
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        self._max_train_score = np.max(scores)
        self.fit_model(variants, scores, uncertainties)

    def predict(
        self, variants: list[Variant],
    ) -> list[ModelPrediction]:
        """
        Predict scores and uncertainties.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Tuple of (predictions, uncertainties)
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Learner":
        """
        Load the model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model
        """
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    @property
    def max_train_score(self):
        return self._max_train_score


class LearnerFactory:

    _base_learner_params = ["input_dim", "random_state"]

    # This method should be overridden by subclasses
    def create_instance(self, **kwargs) -> Learner:
        raise NotImplementedError("This method should be overridden by subclasses")

    def extract_learner_params(self, **kwargs) -> dict:
        params = {param_name: kwargs.get(param_name) for param_name in kwargs
                  if param_name not in self._base_learner_params}
        return params

    def extract_base_learner_params(self, **kwargs) -> dict:
        params = {param_name: kwargs.get(param_name) for param_name in kwargs
                  if param_name in self._base_learner_params}
        return params

