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
    def __init__(self, base_model, alphabet, wt_sequence: str, batch_size: int):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._base_model = base_model.to(self._device)
        self._alphabet = alphabet
        self._wt_sequence = wt_sequence
        self._batch_size = batch_size
        self._batch_size = batch_size

    # Identify mutation positions
    def _find_mutation(self, wt_sequence, mutant_sequence):
        for i, (wt_residue, mutant_residue) in enumerate(
                zip(wt_sequence, mutant_sequence)):
            if wt_residue != mutant_residue:
                return i, wt_residue, mutant_residue
        return None, None, None

    def compute_log_likelihoods(self, variants: list[Variant]
                                ) -> list[Variant]:
        """ Compute log-likelihoods for a list of protein variants in batches. """
        self._base_model.eval()
        data = [(v.id, v.sequence) for v in variants]
        batch_converter = self._alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        num_variants = len(variants)
        llr_list = [0.0] * num_variants

        with torch.no_grad():
            for start in range(0, num_variants, self._batch_size):
                end = min(start + self._batch_size, num_variants)
                batch_slice = slice(start, end)
                batch_tokens_subset = batch_tokens[batch_slice]
                logits = self._base_model(
                    batch_tokens_subset, repr_layers=[], return_contacts=False
                )["logits"]
                log_probs = torch.log_softmax(logits, dim=-1)

                for i, variant_idx in enumerate(range(start, end)):
                    variant = variants[variant_idx]
                    pos, wt_residue, mutant_residue = self._find_mutation(
                        self._wt_sequence, variant.sequence)
                    if pos is None:
                        llr = 0
                    else:
                        mut_log_prob = log_probs[
                            i, pos + 1,
                            self._alphabet.get_idx(mutant_residue)
                        ].item()
                        wt_log_prob = log_probs[
                            i, pos + 1,
                            self._alphabet.get_idx(wt_residue)
                        ].item()
                        llr = mut_log_prob - wt_log_prob
                    llr_list[variant_idx] = llr

        for variant, llr in zip(variants, llr_list):
            variant.log_likelihood_ratio = llr
        return variants


def compute_log_likelihoods1(self, variants: list[Variant]
                                ) -> list[Variant]:
        """ Compute log-likelihoods for a list of protein variants in batches. """
        self._base_model.eval()
        data = [(v.id, v.sequence) for v in variants]
        batch_converter = self._alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        num_variants = len(variants)
        llr_list = [0.0] * num_variants

        with torch.no_grad():
            for start in range(0, num_variants, self._batch_size):
                end = min(start + self._batch_size, num_variants)
                batch_slice = slice(start, end)
                batch_tokens_subset = batch_tokens[batch_slice]
                logits = self._base_model(
                    batch_tokens_subset, repr_layers=[], return_contacts=False
                )["logits"]
                log_probs = torch.log_softmax(logits, dim=-1)

                for i, variant_idx in enumerate(range(start, end)):
                    variant = variants[variant_idx]
                    pos, wt_residue, mutant_residue = self._find_mutation(
                        self._wt_sequence, variant.sequence)
                    if pos is None:
                        llr = 0
                    else:
                        mut_log_prob = log_probs[
                            i, pos + 1,
                            self._alphabet.get_idx(mutant_residue)
                        ].item()
                        wt_log_prob = log_probs[
                            i, pos + 1,
                            self._alphabet.get_idx(wt_residue)
                        ].item()
                        llr = mut_log_prob - wt_log_prob
                    llr_list[variant_idx] = llr

        for variant, llr in zip(variants, llr_list):
            variant.log_likelihood_ratio = llr
        return variants


def compute_log_likelihoods1(self, variants: list[Variant]
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
                mut_log_prob = log_probs[
                    i, pos + 1,
                    self._alphabet.get_idx(mutant_residue)].item()
                wt_log_prob = log_probs[
                    i, pos + 1,
                    self._alphabet.get_idx(wt_residue)].item()
                # log likelihood ratio
                llr = mut_log_prob - wt_log_prob
            variant.log_likelihood_ratio = llr
        return variants


class ESM2LogLikelihoodComputerFactory(LogLikelihoodComputerFactory):

    # This method should be overridden by subclasses
    def create_instance(self, **kwargs
                        ) -> LogLikelihoodComputer:
        config = kwargs.pop("config")
        base_model_name = config["parameters"]["base_model_name"]
        base_model, alphabet = get_esm_model_and_alphabet(
            base_model_name)
        wt_sequence = kwargs.pop("wt_sequence")
        batch_size = config["parameters"]["batch_size"]
        return ESM2LogLikelihoodComputer(base_model, alphabet,
                                         wt_sequence, batch_size, batch_size)
