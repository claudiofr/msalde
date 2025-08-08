from .model import ModelPrediction, AcquisitionScore


class AcquisitionStrategy:
    """
    Base class for acquisition strategies.
    """
    def __init__(self, name: str, parameters: dict):
        self._name = name
        self._parameters = parameters
    
    def compute_scores(self,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        pass

    def select_top_candidates(self, num_candidates: int,
                              variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Select the next data point to query based on the model and current data.
        """
        scores = self.compute_scores(variant_predictions)
        # Sort by score in descending order
        return scores.sort_values('SCORE', reverse=True)[:num_candidates]

