from omegaconf import OmegaConf

from .variant_ref_loader import VariantRefLoader

from .external_repository import ALDEExternalRepository

from .query_repository import ALDEQueryRepository

from .learner import Learner
from .strategy import AcquisitionStrategy

from .acquisition_strategy import (
    GreedyStrategyFactory,
    RandomStrategyFactory,
    UCBStrategyFactory,
    ThompsonSamplingStrategyFactory,
    ExpectedImprovementStrategyFactory,
    VarianceStrategyFactory
)
from .esm_embedder import ESMEmbedderFactory
from .file_load_embedder import FileLoadEmbedderFactory

from .simulator import DESimulator
from .active_learner import (
    RidgeLearnerFactory,
    RandomForestLearnerFactory,
)
from .esm_learner import (
    ESM2HingeForestLearnerFactory,
    ESM2RandomForestLearnerFactory,
    ESM2MLPLearnerFactory
)
from .esm_ll_learner import (
    ESM2LogLikelihoodLearnerFactory   
)
from .esm_log_likelihood_computer import (
    ESM2LogLikelihoodComputerFactory)

from .data_file_loader import VariantDataFileLoaderFactory

from .repository import (
    ALDERepository,
    RepoSessionContext,
)
from .plotter import ALDEPlotter
from .dataset_repository import DatasetRepository
from .var_repository import VariantRepository
from .var_repository import RepoSessionContext as VariantRepoSessionContext


import yaml


class ALDEContainer:
    """
    Class to simulate a Dependency Injection container.
    It could be reimplemented in the future if we decide to use
    a proper one. The interface, however, would remain the same.
    """
    _learner_factories = {
            "RidgeRegression": RidgeLearnerFactory(),
            "RandomForestRegression": RandomForestLearnerFactory(),
            "ESM2RandomForestRegression": ESM2RandomForestLearnerFactory(),
            "ESM2MLPRegression": ESM2MLPLearnerFactory(),
            "ESM2HingeForestRegression": ESM2HingeForestLearnerFactory(),
            "ESM2LogLikelihood": ESM2LogLikelihoodLearnerFactory()
    }
    _acquisition_strategy_factories = {
            "Random": RandomStrategyFactory(),
            "Greedy": GreedyStrategyFactory(),
            "UCB": UCBStrategyFactory(),
            "ThompsonSampling": ThompsonSamplingStrategyFactory(),
            "ExpectedImprovement": ExpectedImprovementStrategyFactory(),
            "Variance": VarianceStrategyFactory(),
        }
    _data_loader_factories = {
        "file_loader": VariantDataFileLoaderFactory(),
    }
    _protein_embedder_factories = {
        "file_loader": FileLoadEmbedderFactory(),
        "esm": ESMEmbedderFactory(),
    }
    _log_likelihood_computer_factories = {
        "ESM2LLRComputer": ESM2LogLikelihoodComputerFactory(),
    }

    def __init__(self, config_file: str = "./config/msalde.yaml"):
        """
        Parameters
        ----------
        app_root : str
            Directory where app config file is location.
            Path of config file:
            <value of app_root>/config/config.yaml
        """
        # Load YAML file
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        sub_run_config_file = config.sub_runs.config_file
        with open(sub_run_config_file, "r") as f:
            sub_run_config_dict = yaml.safe_load(f)
        sub_run_config = OmegaConf.create(sub_run_config_dict)

        repo_session_context = RepoSessionContext(
            config.db.url)
        self._repository = ALDERepository(repo_session_context)
        self._query_repository = ALDEQueryRepository(
            repo_session_context)
        self._dataset_repository = DatasetRepository(
            config.datasets,
            self._data_loader_factories,
            self._repository,
            self._query_repository
        )
        self._simulator = DESimulator(
            repository=self._repository,
            dataset_repository=self._dataset_repository,
            protein_embedder_factories=self._protein_embedder_factories,
            learner_factories=self._learner_factories,
            acquisition_strategy_factories=
            self._acquisition_strategy_factories,
            log_likelihood_computer_factories=
            self._log_likelihood_computer_factories,
            run_config=config,
            sub_run_config=sub_run_config
        )
        self._external_repository = ALDEExternalRepository(
            config.external_repo)
        self._plotter = ALDEPlotter(config)
        variant_repo_session_context = VariantRepoSessionContext(
            config.variant_ref.db.url)
        self._variant_repository = VariantRepository(
            variant_repo_session_context
        )
        self._variant_ref_loader = VariantRefLoader(
            config.variant_ref.datasets
        )

    @property
    def simulator(self):
        return self._simulator

    @property
    def query_repository(self):
        return self._query_repository

    @property
    def external_repository(self):
        return self._external_repository

    @property
    def plotter(self):
        return self._plotter

    @property
    def dataset_repository(self):
        return self._dataset_repository

    @property
    def variant_ref_loader(self):
        return self._variant_ref_loader
    
    @property
    def variant_repository(self):
        return self._variant_repository