from dataclasses import Field, dataclass
from typing import Optional, Union

from .learner import Learner
from .strategy import AcquisitionStrategy


@dataclass
class SubRunParameters:
    learner_name: str
    learner_parameters: str
    acquisition_strategy_name: str
    acquisition_strategy_parameters: str
    learner: Learner
    acquisition_strategy: AcquisitionStrategy


@dataclass
class Variant:
    """Protein sequence variant."""

    id: Union[int, str]
    name: str
    sequence: str
    embedding: Optional[np.ndarray]


@dataclass
class AssayResult:
    """Experimental assay result."""

    variant_id: Union[int, str]
    score: float
    uncertainty: Optional[float]
    assay_id: str


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


class ProposedVariant(BaseModel):
    """Variant proposed by the acquisition function."""

    variant: Variant = Field(..., description="The proposed variant")
    acquisition_score: float = Field(..., description="Acquisition score")
    acquisition_type: str = Field(...,
                                  description="Type of acquisition function used")
    predicted_score: Optional[float] = Field(
        None, description="Predicted score")
    predicted_uncertainty: Optional[float] = Field(
        None, description="Predicted uncertainty")


class Round(BaseModel):
    """Active learning round."""

    id: int = Field(..., description="Round number")
    name: str = Field(..., description="Round name")
    description: Optional[str] = Field(None, description="Round description")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp")
    proposed_variants: List[ProposedVariant] = Field(
        default_factory=list, description="Proposed variants")
    assay_results: List[AssayResult] = Field(
        default_factory=list, description="Assay results")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics")
