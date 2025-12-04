from .model import (Variant, AssayResult)
from .data_loader import VariantDataLoader
from .repository import ALDERepository
from .query_repository import ALDEQueryRepository
from .dbmodel import Dataset
from typing import Tuple


class DatasetRepository:

    def __init__(self, config, data_loader_factories,
                 repository: ALDERepository,
                 query_repository: ALDEQueryRepository):
        self._config = config
        self._data_loader_factories = data_loader_factories
        self._repository = repository
        self._query_repository = query_repository
    
    def get_data_loader_type(self, dataset_name: str) -> str:
        config = self._config[dataset_name]
        if "data_loader_type" not in config:
            raise ValueError(f"Data loader type not specified for dataset {dataset_name}")
        return config.data_loader_type

    def _get_data_loader(self, dataset_name: str) -> VariantDataLoader:
        data_loader_type = self.get_data_loader_type(dataset_name)
        if data_loader_type not in self._data_loader_factories:
            raise ValueError(f"Unknown data loader type: {data_loader_type}")
        config = self._config[dataset_name]
        factory = self._data_loader_factories[data_loader_type]
        data_loader = factory.create_instance(config)
        return data_loader

    def load_dataset(self, dataset_name: str) -> Tuple[
            list[Variant], list[AssayResult], str, float]:
        data_loader = self._get_data_loader(dataset_name)
        variants, results, wt_sequence, wt_assay_score = data_loader.load()
        self._repository.upsert_dataset(dataset_name, wt_sequence,
                                        wt_assay_score)
        return variants, results, wt_sequence, wt_assay_score

    def get_dataset_info(self, dataset_name: str) -> Dataset:
        dataset = self._query_repository.get_dataset_by_name(dataset_name)
        if dataset is None:
            _, _, wt_sequence, wt_assay_score = self.load_dataset(dataset_name)
            dataset = self._query_repository.get_dataset_by_name(dataset_name)
        return dataset



    