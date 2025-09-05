from omegaconf import OmegaConf

from .learner import Learner
from .strategy import AcquisitionStrategy

from .acquisition_strategy import (
    GreedyStrategyFactory, 
    RandomStrategyFactory, 
    UncertaintyStrategyFactory, 
    UCBStrategyFactory,
    DiversityStrategyFactory,
    QBCStrategyFactory, 
    EIStrategyFactory,
    TSStrategyFactory,

)
from .esm_embedder import ESMEmbedder

from .simulator import DESimulator
from .active_learner import RidgeLearnerFactory
from .data_file_loader import VariantDataFileLoader
from .repository import (
    ALDERepository,
    RepoSessionContext,
)
import yaml


class ALDEContainer:
    """
    Class to simulate a Dependency Injection container.
    It could be reimplemented in the future if we decide to use
    a proper one. The interface, however, would remain the same.
    """
    _learner_factories = {
            "RidgeRegression": RidgeLearnerFactory(),
    }
    _acquisition_strategy_factories = {
            "Random": RandomStrategyFactory(),
            "Greedy": GreedyStrategyFactory(),
            "Uncertainty": UncertaintyStrategyFactory(),
            "UCB": UCBStrategyFactory(),
            "Diversity": DiversityStrategyFactory(),
            "QBC": QBCStrategyFactory(),
            "EI": EIStrategyFactory(),
            "TS": TSStrategyFactory(),
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
        self._data_loader = VariantDataFileLoader(config.data_loader)
        self._embedder = ESMEmbedder(config.embedder)
        self._simulator = DESimulator(
            repository=self._repository,
            data_loader=self._data_loader,
            embedder=self._embedder,
            learner_factories=self._learner_factories,
            acquisition_strategy_factories=
            self._acquisition_strategy_factories,
            sub_run_defs=sub_run_config.sub_runs
        )

    @property
    def simulator(self):
        return self._simulator
    
    @property
    def repository(self):
        return self._repository


