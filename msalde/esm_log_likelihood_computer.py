from typing import Optional
from .model import Variant
from .log_likelihood_computer import (
    LogLikelihoodComputer,
    LogLikelihoodComputerFactory
)
from .esm_util import get_esm_model_and_alphabet
import torch


class ESM2LogLikelihoodComputer(LogLikelihoodComputer):
    """ Base class for computing log-likelihoods of protein sequences.
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

    def compute_log_likelihoods(self, variants: list[Variant]
                                ) -> list[Variant]:
        """ Compute log-likelihoods for a list of protein variants.
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
            variant.log_likelihood = llr
        return variants


class ESM2LogLikelihoodComputerFactory(LogLikelihoodComputerFactory):

    # This method should be overridden by subclasses
    def create_instance(self, **kwargs
                        ) -> LogLikelihoodComputer:
        base_model_name = kwargs.pop("base_model_name")
        base_model, alphabet = get_esm_model_and_alphabet(
            base_model_name)
        wt_sequence = kwargs.pop("wt_sequence")
        return ESM2LogLikelihoodComputer(base_model, alphabet,
                                        wt_sequence)