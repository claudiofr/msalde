from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class SubRunParameters:
    learner_name: str
    learner_parameters: str
    acquisition_strategy_name: str
    acquisition_strategy_parameters: str
    learner: object
    acquisition_strategy: object


@dataclass
class Variant:
    """Protein sequence variant."""

    id: Union[int, str]
    name: str
    sequence: str
    embedding: Optional[np.ndarray] = None


@dataclass
class AssayResult:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float
    uncertainty: Optional[float] = None
    assay_id: str = None


@dataclass
class ModelPrediction:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float
    uncertainty: Optional[float] = None


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

    rmse: float
    r2: float
    spearman: float
    top_n_mean = float
