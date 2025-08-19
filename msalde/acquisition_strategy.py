from .model import AcquisitionScore, ModelPrediction
from .strategy import AcquisitionStrategy, AcquisitionStrategyFactory
import numpy as np
from typing import Optional


class RandomStrategy(AcquisitionStrategy):
    """
    A strategy that selects random samples from the dataset.
    """
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the random strategy.

        Args:
            random_state: Random seed for reproducibility
        """
        self._random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)    

    def compute_scores(self,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Selects random samples from the dataset.

        Returns:
            A list of randomly selected samples.
        """
        if self._random_state is not None:
            np.random.seed(self._random_state)

        scores = np.random.rand(len(variant_predictions))
        return [AcquisitionScore(variant_id=pred.variant_id, score=score)
                for pred, score in zip(variant_predictions, scores)]


class RandomStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return RandomStrategy()


class GreedyStrategy(AcquisitionStrategy):

    """
    A strategy that selects random samples from the dataset.
    """

    def compute_scores(self,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Acquisition score is prediction score.

        """
        return [AcquisitionScore(variant_id=pred.variant_id,
                                 score=pred.score) for pred in
                variant_predictions]


class GreedyStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return GreedyStrategy()


