from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class SubRunParameters:
    learner_type: str
    learner_name: str
    learner_parameters: str
    learner_uses_embedder: bool
    learner_uses_random_seed: bool
    learner_uses_wt_sequence: bool
    first_round_acquisition_strategy_type: str
    first_round_acquisition_strategy_name: str
    first_round_acquisition_strategy_parameters: str
    first_round_acquisition_strategy_uses_random_seed: bool
    first_round_acquisition_strategy_uses_sub_run_context: bool
    acquisition_strategy_type: str
    acquisition_strategy_name: str
    acquisition_strategy_parameters: str
    acquisition_strategy_uses_random_seed: bool
    learner:  Optional[object] = None
    first_round_acquisition_strategy: Optional[object] = None
    acquisition_strategy: Optional[object] = None


@dataclass
class Variant:
    """Protein sequence variant."""

    id: Union[int, str]
    name: str
    sequence: str
    aa_substitution: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    log_likelihood_ratio: Optional[float] = None


@dataclass
class AssayResult:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float
    uncertainty: Optional[float] = None
    assay_id: str = None


@dataclass
class ModelComponentPrediction:
    """Experimental assay result."""

    score: float
    uncertainty: Optional[float] = None


@dataclass
class ModelPrediction:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float
    uncertainty: Optional[float] = None
    component_predictions: Optional[list[ModelComponentPrediction]] = None


@dataclass
class AcquisitionScore:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float


@dataclass
class SelectedVariant:
    """Experimental assay result."""

    variant: Variant
    prediction: Optional[ModelPrediction] = None
    acquisition_score: Optional[AcquisitionScore] = None
    top_prediction: bool = False
    top_acquisition_score: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model run."""

    train_rmse: float
    train_r2: float
    train_spearman: float
    validation_rmse: float
    validation_r2: float
    validation_spearman: float
    test_rmse: float
    test_r2: float
    test_spearman: float
    spearman: float
    top_n_mean: Optional[float] = None


@dataclass
class NFoldCVParams:
    """List of list of variant ids for each fold"""

    fold_variant_ids: list[list[Union[int, str]]]

    def num_train_variants(self, fold_num: int) -> int:
        return sum(len(fold_variant_ids) for i, fold_variant_ids in enumerate(self.fold_variant_ids)
                   if i != fold_num)


@dataclass
class SubRunContext:
    """Right now only relevant for n_fold_cv"""

    n_fold_cv: NFoldCVParams
