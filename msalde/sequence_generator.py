from .model import Variant


class ProteinSequenceGenerator:

    def generate_sequence(self, variant: Variant) -> str:
        pass


class ProteinSequenceGeneratorFactory:

    def create_instance(self, **kwargs) -> ProteinSequenceGenerator:
        raise NotImplementedError("This method should be overridden by subclasses")


class SingleAASubstitutionSequenceGenerator(ProteinSequenceGenerator):

    def __init__(self, wt_fasta_file: str, substitution_col: str):
        self._wt_fasta_file = wt_fasta_file
        self._substitution_col = substitution_col

    def generate_sequence(self, variant: Variant) -> str:
        pass



class SingleAASubstitutionSequenceGeneratorFactory(ProteinSequenceGenerator):
    def create_instance(self, **kwargs) -> ProteinSequenceGenerator:
        return SingleAASubstitutionSequenceGenerator(**kwargs)
