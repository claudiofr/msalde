from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor

from .model import ModelPrediction, Variant

from .plm import PLMModel
from .learner import Learner, LearnerFactory


class RidgeLearner(Learner):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def __init__(
            self,
            input_dim: Optional[int] = None,
            random_state: Optional[int] = None,
            plm_model: Optional[str] = PLMModel,
            alpha: float = 1.0,
            num_estimators: int = 10,
            ):
        super().__init__(input_dim, random_state)
        self._plm_model = plm_model
        self._alpha = alpha
        self._num_estimators = num_estimators
        self._scaler = StandardScaler()
        self._model = BaggingRegressor(
            Ridge(alpha=alpha),
            n_estimators=num_estimators,
            random_state=random_state,
        )

    def fit(
        self,
        sequences: np.ndarray,
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the ridge regression model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        # Log input data stats
        logger.info(f"Fitting RidgeLearner with {len(scores)} samples")
        logger.info(
            f"Scores range: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")

        # Reduce dimensionality
        X = self._fit_transform_embeddings(embeddings)
        logger.info(f"Reduced embeddings shape: {X.shape}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X, scores)

        # Log model info
        if hasattr(self.model, 'estimators_'):
            logger.info(f"Fitted {len(self.model.estimators_)} estimators")

        self.is_fitted = True
        logger.info("Model fitting complete")

    def predict(
        self,
        variants: list[Variant],
        return_std: bool = False,
    ) -> list[ModelPrediction]:
        """
        Make predictions with the ridge regression model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            return_std: Whether to return standard deviations

        Returns:
            Predictions [n_samples] or tuple of (predictions, uncertainties)
        """

        # Reduce dimensionality
        X = self._transform_embeddings([v.embedding for var in variants])

        # Scale features
        X = self._scaler.transform(X)

        # Make predictions
        if not return_std:
            # Standard prediction
            predicted_scores = self._model.predict(X)
            return [ModelPrediction(variant_id=variant.id, score=score) for
                    variant, score in zip(variants, predicted_scores)]
        else:
            # For ensemble models, get predictions from all estimators
            predictions = np.zeros((X.shape[0], len(self._model.estimators_)))
            for i, estimator in enumerate(self._model.estimators_):
                predictions[:, i] = estimator.predict(X)

            # Calculate mean and std across estimators
            prediction_means = np.mean(predictions, axis=1)
            prediction_stds = np.std(predictions, axis=1)

            return [ModelPrediction(
                variant_id=variant.id, score=score, uncertaintity=std) for
                variant, score, std in zip(variants, prediction_means,
                prediction_stds)]


class RidgeLearnerFactory(LearnerFactory):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_learner(self, **kwargs) -> RidgeLearner:
        """
        Create a RidgeLearner instance with the given parameters.
        """

        return RidgeLearner(**kwargs)

