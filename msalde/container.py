from omegaconf import OmegaConf

from .learner import Learner
from .strategy import AcquisitionStrategy

from .acquisition_strategy import (
    GreedyStrategyFactory,
    RandomStrategyFactory
)
from .esm_embedder import ESMEmbedderFactory
from .file_load_embedder import FileLoadEmbedderFactory

from .simulator import DESimulator
from .active_learner import (
    RidgeLearnerFactory,
    RandomForestLearnerFactory,
)
from .data_file_loader import VariantDataFileLoaderFactory

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
            "RandomForestRegression": RandomForestLearnerFactory(),
    }
    _acquisition_strategy_factories = {
            "Random": RandomStrategyFactory(),
            "Greedy": GreedyStrategyFactory(),
        }
    _data_loader_factories = {
        "file_loader": VariantDataFileLoaderFactory(),
    }
    _protein_embedder_factories = {
        "file_loader": FileLoadEmbedderFactory(),
        "esm": ESMEmbedderFactory(),
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
        self._simulator = DESimulator(
            repository=self._repository,
            data_loader_factories=self._data_loader_factories,
            protein_embedder_factories=self._protein_embedder_factories,
            learner_factories=self._learner_factories,
            acquisition_strategy_factories=
            self._acquisition_strategy_factories,
            run_config=config,
            sub_run_config=sub_run_config
        )

    @property
    def simulator(self):
        return self._simulator

