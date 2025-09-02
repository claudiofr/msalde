from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

from .model import ModelComponentPrediction, ModelPrediction, Variant

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
            alpha: float = 1.0,
            num_estimators: int = 10,
            ):
        super().__init__(input_dim, random_state)
        self._alpha = alpha
        self._num_estimators = num_estimators
        self._model = BaggingRegressor(
            Ridge(alpha=alpha),
            n_estimators=num_estimators,
            random_state=random_state,
        )

    def fit(
        self,
        variants: list[Variant],
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

        # Reduce dimensionality
        embeddings = np.asarray([v.embedding for v in variants])
        X = self._fit_transform_embeddings(embeddings)

        # Scale features
        X = self._fit_scale_embeddings(X)

        # Fit model
        self._model.fit(X, scores)

        # Log model info
        self.is_fitted = True

    def predict(
        self,
        variants: list[Variant],
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
        embeddings = np.asarray([v.embedding for v in variants])
        X = self._transform_embeddings(embeddings)

        # Scale features
        X = self._scale_embeddings(X)

        # Make predictions
        has_estimators = hasattr(self._model, "estimators_") and \
            len(self._model.estimators_) > 0
        if not has_estimators:
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
                variant_id=variant.id, score=score, uncertainty=std,
                component_predictions=[ModelComponentPrediction(score=pred)
                                       for pred in component_preds]
                ) for
                variant, score, std, component_preds in zip(
                    variants, prediction_means, prediction_stds, predictions)]


class RidgeLearnerFactory(LearnerFactory):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_instance(self, **kwargs) -> Learner:
        """
        Create a RidgeLearner instance with the given parameters.
        """

        return RidgeLearner(**kwargs)


class RandomForestLearner(Learner):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def __init__(
            self,
            input_dim: Optional[int] = None,
            random_state: Optional[int] = None,
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
    ):
        super().__init__(input_dim, random_state)
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._bootstrap = bootstrap
        self._oob_score = oob_score
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._warm_start = warm_start
        self._ccp_alpha = ccp_alpha
        self._max_samples = max_samples
        self._monotonic_cst = monotonic_cst


    def fit(
        self,
        variants: list[Variant],
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

        self._model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            criterion=self._criterion,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            min_weight_fraction_leaf=self._min_weight_fraction_leaf,
            max_features=self._max_features,
            max_leaf_nodes=self._max_leaf_nodes,
            min_impurity_decrease=self._min_impurity_decrease,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbose=self._verbose,
            warm_start=self._warm_start,
            ccp_alpha=self._ccp_alpha,
            max_samples=self._max_samples,
            monotonic_cst=self._monotonic_cst,
        )
        # Log input data stats

        # Reduce dimensionality
        embeddings = np.asarray([v.embedding for v in variants])
        X = self._fit_transform_embeddings(embeddings)

        # Scale features
        X = self._fit_scale_embeddings(X)

        # Fit model
        self._model.fit(X, scores)

        # Log model info
        self.is_fitted = True

        print(X.shape[0], "samples used to fit the model.")

    def predict(
        self,
        variants: list[Variant],
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
        embeddings = np.asarray([v.embedding for v in variants])
        X = self._transform_embeddings(embeddings)

        # Scale features
        X = self._scale_embeddings(X)

        print(X.shape[0], "samples used to make predictions.")

        # Make predictions
        has_estimators = hasattr(self._model, "estimators_") and \
            len(self._model.estimators_) > 0
        if not has_estimators:
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
                variant_id=variant.id, score=score, uncertainty=std,
                component_predictions=[ModelComponentPrediction(score=pred)
                                       for pred in component_preds]
                ) for
                variant, score, std, component_preds in zip(
                    variants, prediction_means, prediction_stds, predictions)]


class RandomForestLearnerFactory(LearnerFactory):
    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_instance(self, **kwargs) -> Learner:
        """
        Create a RidgeLearner instance with the given parameters.
        """

        return RandomForestLearner(**kwargs)


