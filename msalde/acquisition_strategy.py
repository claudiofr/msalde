from .model import ModelPrediction
from .strategy import AcquisitionStrategy, AcquisitionStrategyFactory
import numpy as np


class RandomStrategy(AcquisitionStrategy):
    """
    A strategy that selects random samples from the dataset.
    """

    def compute_scores(self,
                       variant_predictions: list[ModelPrediction]) -> \
            list[float]:
        """
        Selects random samples from the dataset.

        Returns:
            A list of randomly selected samples.
        """
        scores = np.random.rand(len(variant_predictions))
        return scores


class RandomStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return RandomStrategy()
