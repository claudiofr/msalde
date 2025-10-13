import numpy as np
from .model import ModelPrediction, Variant
from .learner import Learner, LearnerFactory
from typing import Optional


class ESM2LogLikelihoodLearner(Learner):

    def fit_model(
        self,
        variants: list[Variant],
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def predict(
        self,
        variants: list[Variant],
    ) -> list[ModelPrediction]:
        return [ModelPrediction(
            variant_id=v.id, score=v.log_likelihood_ratio) for v in variants]


class ESM2LogLikelihoodLearnerFactory(LearnerFactory):

    """
    Factory class for creating RidgeLearner instances.
    This is a placeholder for the actual implementation.
    """
    def create_instance(self, **kwargs) -> Learner:
        """
        Create a RidgeLearner instance with the given parameters.
        """
        return ESM2LogLikelihoodLearner()
                                         
        
