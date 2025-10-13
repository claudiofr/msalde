from typing import Optional
from .model import Variant


class LogLikelihoodComputer:
    """ Base class for computing log-likelihoods of protein sequences.
    """

    def compute_log_likelihoods(self, variants: list[Variant]
                                ) -> list[Variant]:
        """ Compute log-likelihoods for a list of protein variants.
        """
        pass


class LogLikelihoodComputerFactory:

    # This method should be overridden by subclasses
    def create_instance(self, **kwargs
                        ) -> LogLikelihoodComputer:
        raise NotImplementedError("This method should be overridden by subclasses")
