from .learner import Learner
from .model import AcquisitionScore, ModelPrediction
from .strategy import AcquisitionStrategy, AcquisitionStrategyFactory
from .al_util import cantor_pair
import numpy as np
from typing import Optional
from .esm_util import get_esm_model_and_alphabet


class ESM2LogLikelihoodStrategy(AcquisitionStrategy):
    """
    A strategy that selects random samples from the dataset.
    """
    def __init__(self, base_model, alphabet, wt_sequence: str):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._base_model = base_model.to(self._device)
        self._alphabet = alphabet
        self._wt_sequence = wt_sequence

   # Identify mutation positions
    def _find_mutation(self, wt_sequence, mutant_sequence):
        for i, (wt_residue, mutant_residue) in enumerate(
                zip(wt_sequence, mutant_sequence)):
            if wt_residue != mutant_residue:
                return i, wt_residue, mutant_residue
        return None, None, None
    
    def compute_scores(self,
                       fitted_learner: Learner,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Selects random samples from the dataset.

        Returns:
            A list of randomly selected samples.
        """
        self._base_model.eval()
        data = [(v.id, v.sequence) for v in variants]
        batch_converter = self._alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        # Forward pass
        with torch.no_grad():
            logits = self._base_model(
                batch_tokens, repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Score each mutant
        predictions = []
        for i, variant in enumerate(variants):
            pos, wt_residue, mutant_residue = self._find_mutation(
                self._wt_sequence, variant.sequence)
            if pos is None:
                llr = 0
            else:
                mut_log_prob = log_probs[i, pos + 1, alphabet.get_idx(mutant_residue)].item()
                wt_log_prob = log_probs[i, pos + 1, alphabet.get_idx(wt_residue)].item()
                # log likelihood ratio
                llr = mut_log_prob - wt_log_prob
            predictions.append(ModelPrediction=variant.id,
                            score=llr)


        if self._random_state is not None:
            np.random.seed(self._random_state)

        scores = np.random.rand(len(variant_predictions))
        return [AcquisitionScore(variant_id=pred.variant_id, score=score)
                for pred, score in zip(variant_predictions, scores)]


class RandomStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return RandomStrategy(**kwargs)


class GreedyStrategy(AcquisitionStrategy):

    """
    A strategy that selects random samples from the dataset.
    """

    def compute_scores(self,
                       fitted_learner: Learner,
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


class UCBStrategy(AcquisitionStrategy):

    """
    A strategy that selects random samples from the dataset.
    """

    def __init__(self, exploration_weight: float = 1.0):
        self._exploration_weight = exploration_weight

    def compute_scores(self,
                       fitted_learner: Learner,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Acquisition score is prediction score.

        """
        return [AcquisitionScore(
            variant_id=pred.variant_id,
            score=pred.score + (
                self._exploration_weight * pred.uncertainty))
                for pred in variant_predictions]


class UCBStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return UCBStrategy(**kwargs)


class VarianceStrategy(AcquisitionStrategy):

    """
    A strategy that selects random samples from the dataset.
    """

    def compute_scores(self,
                       fitted_learner: Learner,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Acquisition score is prediction score.

        """
        return [AcquisitionScore(
            variant_id=pred.variant_id,
            score=pred.uncertainty) for pred in variant_predictions]


class VarianceStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return VarianceStrategy(**kwargs)


class ExpectedImprovementStrategy(AcquisitionStrategy):

    """
    A strategy that selects random samples from the dataset.
    """

    def compute_scores(self,
                       fitted_learner: Learner,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Acquisition score is prediction score.

        """
        def compute_ei(mu, sigma, best_so_far):
            from scipy.stats import norm

            if sigma is None or sigma == 0.0:
                return 0.0

            Z = (mu - best_so_far) / sigma
            ei = (mu - best_so_far) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        return [AcquisitionScore(
            variant_id=pred.variant_id,
            score=compute_ei(pred.score, pred.uncertainty,
                             fitted_learner.max_train_score))
                for pred in variant_predictions]


class ExpectedImprovementStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return ExpectedImprovementStrategy(**kwargs)


class ThompsonSamplingStrategy(AcquisitionStrategy):
    """
    A strategy that selects random samples from the dataset.
    """
    def __init__(self, random_state1: Optional[int] = None,
                 random_state2: Optional[int] = None):
        """
        Initialize the random strategy.

        Args:
            random_state: Random seed for reproducibility
        """
        self._random_state = cantor_pair(random_state1, random_state2)

    def compute_scores(self,
                       fitted_learner: Learner,
                       variant_predictions: list[ModelPrediction]) -> \
            list[AcquisitionScore]:
        """
        Randomly pick one of the component models and use their predictions as
        the acquisition scores.
        These component models would typically be the estimators in an
        ensemble model.
        We ignore the predictions from the main ensemble model.
        """
        num_components = len(
            variant_predictions[0].component_predictions)
        if self._random_state is not None:
            np.random.seed(self._random_state)
        random_component_idx = np.random.randint(num_components)
        return [AcquisitionScore(
            variant_id=pred.variant_id,
            score=pred.component_predictions[random_component_idx].score)
                for pred in variant_predictions]


class ThompsonSamplingStrategyFactory(AcquisitionStrategyFactory):

    def create_instance(self, **kwargs) -> AcquisitionStrategy:
        return ThompsonSamplingStrategy(**kwargs)




