import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from itertools import chain

from datetime import datetime
from typing import Tuple, Iterable

from .dbmodel import ALDERound, ALDERun, ALDESubRun, ALDESimulation

from .embedder import ProteinEmbedder

from .strategy import AcquisitionStrategy, AcquisitionStrategyFactory

from .learner import Learner, LearnerFactory

from .repository import ALDERepository

from .data_loader import VariantDataLoader
from .model import (
    AcquisitionScore, AssayResult, PerformanceMetrics, SelectedVariant,
    SubRunParameters, Variant,
    ModelPrediction
)


class DESimulator:
    def __init__(
        self,
        data_loader: VariantDataLoader,
        repository: ALDERepository,
        embedder: ProteinEmbedder,
        learner_factories: dict[str, LearnerFactory],
        acquisition_strategy_factories: dict[str, AcquisitionStrategyFactory],
        sub_run_defs,
    ):
        self._data_loader = data_loader
        self._repository = repository
        self._embedder = embedder
        self._learner_factories = learner_factories
        self._acquisition_strategy_factories = acquisition_strategy_factories
        self._sub_run_defs = sub_run_defs

    def _load_assay_data(self) -> Tuple[list[Variant], list[AssayResult]]:
        """
        Loads assay data from the data loader.
        Input file is listed in the config file.

        Returns:
            Tuple of (variants, assay_results)
        """
        return self._data_loader.load()

    def _split_assay_data(
        self,
        assay_variants: list[Variant],
        assay_results: list[AssayResult],
        test_fraction: float,
        random_seed: int,
    ) -> Tuple[
        list[Variant], list[AssayResult], list[Variant], list[AssayResult]
    ]:
        """
        Splits assay data into simulation and test sets.

        Returns:
            simulation_variants, simulation_assay_results, test_variants, test_assay_results
        """
        import random

        # Ensure reproducibility
        rng = random.Random(random_seed)
        indices = list(range(len(assay_variants)))
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_fraction))
        sim_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        simulation_variants = [assay_variants[i] for i in sim_indices]
        simulation_assay_results = [assay_results[i] for i in sim_indices]
        test_variants = [assay_variants[i] for i in test_indices]
        test_assay_results = [assay_results[i] for i in test_indices]

        return (
            simulation_variants,
            simulation_assay_results,
            test_variants,
            test_assay_results,
        )

    def _get_max_assay_score(self, assay_results: list[AssayResult]) -> float:
        """
        Returns the maximum assay score from a list of AssayResult objects.
        """
        return max(result.score for result in assay_results)

    def _create_run(
        self,
        name: str,
        descrip: str,
        num_rounds: int,
        num_variants: int,
        num_selected_variants_first_round: int,
        num_top_acquistion_score_variants_per_round: int,
        num_top_prediction_score_variants_per_round: int,
        batch_size: int,
        test_fraction: float,
        random_seed: int,
        max_assay_score: float
    ) -> ALDERun:
        """
        Creates a new run record in the repository and returns its run_id.
        """
        run = self._repository.add_run(
            name=name,
            descrip=descrip,
            num_rounds=num_rounds,
            num_variants=num_variants,
            num_selected_variants_first_round=
            num_selected_variants_first_round,
            num_top_acquistion_score_variants_per_round=
            num_top_acquistion_score_variants_per_round,
            num_top_prediction_score_variants_per_round=
            num_top_prediction_score_variants_per_round,
            batch_size=batch_size,
            test_fraction=test_fraction,
            random_seed=random_seed,
            max_assay_score=max_assay_score,
            start_ts=datetime.now(),
        )
        return run

    def _create_sub_run(self, run_id: int, sub_run_params: SubRunParameters) \
            -> ALDESubRun:
        """
        Creates a new sub-run record in the repository and returns its sub_run_id.
        """
        learner_params = json.dumps(sub_run_params.learner_parameters) \
            if sub_run_params.learner_parameters and \
            len(sub_run_params.learner_parameters) > 0 \
            else None
        first_round_strategy_params = json.dumps(
            sub_run_params.first_round_acquisition_strategy_parameters) \
            if sub_run_params.first_round_acquisition_strategy_parameters and \
            len(sub_run_params.first_round_acquisition_strategy_parameters) > 0 \
            else None
        strategy_params = json.dumps(
            sub_run_params.acquisition_strategy_parameters) \
            if sub_run_params.acquisition_strategy_parameters and \
            len(sub_run_params.acquisition_strategy_parameters) > 0 \
            else None
        sub_run = self._repository.add_sub_run(
            run_id=run_id,
            learner_name=sub_run_params.learner_name,
            learner_parameters=learner_params,
            first_round_acquisition_strategy_name=
            sub_run_params.first_round_acquisition_strategy_name,
            first_round_acquisition_strategy_parameters=first_round_strategy_params,
            acquisition_strategy_name=sub_run_params.acquisition_strategy_name,
            acquisition_strategy_parameters=strategy_params,
            start_ts=datetime.now(),
        )
        return sub_run

    def _create_round(self, simulation_id: int, round_num: int) -> ALDERound:
        """
        Creates a new round record in the repository and returns its round_id.
        """
        from datetime import datetime
        round = self._repository.add_round(
            simulation_id=simulation_id,
            round_num=round_num,
            start_ts=datetime.now(),
        )
        return round

    def _end_round(self, round_id: int,
                   performance_metrics: PerformanceMetrics,
                   best_variant: AssayResult):
        """
        Ends a round by updating its performance metrics in the repository.
        """
        self._repository.end_round(round_id, performance_metrics,
                                   best_variant.variant_id, datetime.now())

    def _end_sub_run(self, sub_run_id: int):
        """
        Ends a sub-run by updating its end timestamp in the repository.
        """
        self._repository.end_sub_run(sub_run_id, datetime.now())

    def _end_run(self, run_id: int):
        """
        Ends a run by updating its end timestamp in the repository.
        """
        self._repository.end_run(run_id, datetime.now())

    def _random_sample_variants(self, num_to_select: int,
                                variants: list[Variant]) -> list[Variant]:
        """
        Randomly samples a specified number of variants from the given list.
        """
        import random
        if num_to_select > len(variants):
            raise ValueError("Requested more variants than available.")
        return random.sample(variants,
                             num_to_select)

    def _get_assay_results_for_variants(
        self, variants: list[Variant], assay_results: list[AssayResult]
    ) -> list[AssayResult]:
        """
        Returns a list of AssayResult objects corresponding to the given variants.
        Matching the order of the variants in variants.
        """
        assay_results = [assay_result for variant in variants
                         for assay_result in assay_results if
                         assay_result.variant_id == variant.id]
        return assay_results

    def _convert_assay_results_to_acquisition_scores(
        self, assay_results: list[AssayResult]
    ) -> list[AcquisitionScore]:
        """
        Converts a list of AssayResult objects to acquisition scores.
        By default, uses the assay score as the acquisition score.
        """
        return [AcquisitionScore(variant_id=result.variant_id, score=result.score)
                for result in assay_results]

    def _make_predictions(self, learner: Learner, variants: list[Variant]
                          ) -> list[ModelPrediction]:
        """
        Uses the provided learner to make predictions for the given variants.

        Args:
            learner: The trained learner/model with a predict method.
            variants: List of Variant objects to predict on.

        Returns:
            List of predicted values (floats), one for each variant.
        """
        # Assume each Variant has a method or property to get its features as a numpy array

        return learner.predict(variants)

    def _embed_variants(self, variants: list[Variant]) -> list[Variant]:
        """
        Uses the PLM model to embed the given variants and returns the updated list.
        Assumes self._embedder has an 'embed' method that takes a list of Variant and returns a list of Variant with embeddings.
        """
        if self._embedder is None:
            return variants
        return self._embedder.embed_variants(variants)
        
    def _select_top_variants(
        self,
        acquisition_strategy: AcquisitionStrategy,
        variants: list[Variant],
        predictions: list[ModelPrediction],
        num_top_acquisition_variants: int,
        num_top_prediction_variants: int,

    ) -> list[SelectedVariant]:
        """
        Computes acquisition scores for the given variants using the specified acquisition strategy.

        Args:
            variants: List of Variant objects.
            model_predictions: List of model prediction scores for each variant.
            acquisition_strategy: An object or function with a compute_scores method.

        Returns:
            List of acquisition scores (floats), one for each variant.
        """
        acquisition_scores = acquisition_strategy.compute_scores(predictions)
        top_acquisition_scores = sorted(
            acquisition_scores, key=lambda x: x.score, reverse=True
        )[:num_top_acquisition_variants]
        top_predictions = sorted(predictions, key=lambda x: x.score,
                                    reverse=True
                                    )[:num_top_prediction_variants]

        top_variant_ids = [acq_score.variant_id for acq_score
                            in top_acquisition_scores] + \
                            [pred.variant_id for pred in top_predictions]
        top_variants_dict = {variant.id: SelectedVariant(variant=variant)
                             for variant in variants
                             if variant.id in top_variant_ids}
        for acq_score in top_acquisition_scores:
            top_variants_dict[acq_score.variant_id].acquisition_score = \
                acq_score
            top_variants_dict[acq_score.variant_id
                              ].top_acquisition_score = True
        for pred in top_predictions:
            top_variants_dict[pred.variant_id].prediction = pred
            top_variants_dict[pred.variant_id].top_prediction = True
        return list(top_variants_dict.values())

    def _select_top_variants1(
        self,
        variants: list[Variant],
        acquisition_scores: list[AcquisitionScore],
        num_to_select: int
    ) -> Tuple[list[Variant], list[AcquisitionScore]]:
        """
        Selects the top variants based on acquisition scores.

        Args:
            acquisition_scores: List of AcquisitionScore objects.
            num_to_select: Number of top variants to select.

        Returns:
            Tuple of (selected_variants, selected_acquisition_scores)
        """
        # Sort acquisition_scores by score in descending order
        sorted_scores = sorted(acquisition_scores, key=lambda x: x.score,
                               reverse=True)
        selected_scores = sorted_scores[:num_to_select]
        # Get the corresponding variants in the same sorted order
        # as selected_scores
        selected_variants = [variant for score in selected_scores
                             for variant in variants if variant.id
                             == score.variant_id]
        return selected_variants, selected_scores

    def _get_best_variant(self, assay_results: list[AssayResult]) -> AssayResult:
        """
        Returns the AssayResult with the highest score from the list.
        If the list is empty, returns None.
        """
        return max(assay_results, key=lambda x: x.score)

    def _save_proposed_variants(
        self,
        round_id: int,
        variants: list[SelectedVariant],
        assay_results: list[AssayResult],
    ):
        """
        Saves the proposed variants and their associated data for a given
        round.
        This method should persist the information to the repository or
        database.
        """
        for variant, assay_result in zip(variants, assay_results):
            self._repository.add_round_acquired_variant(
                round_id=round_id,
                variant_id=variant.variant.id,
                variant_name=variant.variant.name,
                assay_score=assay_result.score,
                acquisition_score=variant.acquisition_score.score
                if variant.acquisition_score else None,
                prediction_score=variant.prediction.score
                if variant.prediction else None,
                top_acquisition_score=variant.top_acquisition_score,
                top_prediction_score=variant.top_prediction,
                insert_ts=datetime.now(),
            )

    def _subtract_variant_lists(
        self,
        original_variants: list[Variant],
        variants_to_remove: list[Variant]
    ) -> list[Variant]:
        """
        Removes variants from the original list based on variant IDs.

        Args:
            original_variants: List of variants to filter from
            variants_to_remove: List of variants to remove

        Returns:
            List of remaining variants
        """
        remove_ids = {variant.id for variant in variants_to_remove}
        return [variant for variant in original_variants
                if variant.id not in remove_ids]

    def _fit_model(self, learner: Learner, train_variants: list[Variant],
                   train_assay_results: list[AssayResult]):
        """
        Fits the model using the provided training data.
        This method should call the appropriate method on the learner to fit
        the model.
        """
        train_scores = [result.score for result in train_assay_results]
        learner.fit(train_variants, train_scores)

    def _compute_performance_metrics(
        self,
        train_predictions: list[ModelPrediction],
        train_assay_results: list[AssayResult],
        validation_predictions: list[ModelPrediction],
        validation_assay_results: list[AssayResult],
        test_predictions: list[ModelPrediction],
        test_assay_results: list[AssayResult],
        num_predictions_for_top_n_mean: int = 10
    ) -> PerformanceMetrics:
        """
        Computes performance metrics comparing predictions to actual test results.
        
        Args:
            predictions: List of model predictions
            test_assay_results: List of actual test results
        
        Returns:
            Dictionary containing performance metrics
        """

        def _compute_metrics(
                predictions: Iterable[ModelPrediction],
                assay_results: Iterable[AssayResult]) -> Tuple[float, float,
                                                               float]:
            if predictions is None:
                return None, None, None
            y_pred = [pred.score for pred in predictions]
            y_true = [result.score for result in assay_results]

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)

            return rmse, r2, spearman_corr
        
        train_rmse, train_r2, train_spearman_corr = _compute_metrics(
            train_predictions, train_assay_results
        )
        val_rmse, val_r2, val_spearman_corr = _compute_metrics(
            validation_predictions, validation_assay_results
        )
        test_rmse, test_r2, test_spearman_corr = _compute_metrics(
            test_predictions, test_assay_results
        )
        train_val_predictions = []
        train_val_assay_results = []
        if train_predictions is not None:
            train_val_predictions.extend(train_predictions)
            train_val_assay_results.extend(train_assay_results)
        train_val_predictions.extend(validation_predictions)
        train_val_assay_results.extend(validation_assay_results)
        _, _, spearman_corr = _compute_metrics(
            train_val_predictions, train_val_assay_results)

        # data validation logic
        if train_predictions is None:
            trp = []
            tra = []
        else:
            trp = [v.variant_id for v in train_predictions]
            tra = [v.variant_id for v in train_assay_results]
        vp = [v.variant_id for v in validation_predictions]
        va = [v.variant_id for v in validation_assay_results]
        if test_predictions is None:
            tp = []
            ta = []
        else:
            tp = [v.variant_id for v in test_predictions]
            ta = [v.variant_id for v in test_assay_results]
        tvp = [v.variant_id for v in train_val_predictions]
        tva = [v.variant_id for v in train_val_assay_results]
        test_val = (trp == tra and vp == va and tp == ta and tvp == tva) and \
                (len(trp) + len(vp)) == len(list(tvp)) and \
                not any(v in trp for v in vp) and \
                trp + vp == tvp and \
                not any(v in tvp for v in tp)
        lt = len(trp)
        lv = len(vp)
        assert test_val, "compute performance metrics failed validation"
        assert (lt + lv) == len(tvp), "compute performance metrics failed length check"

        return PerformanceMetrics(
            train_rmse=train_rmse,
            train_r2=train_r2,
            train_spearman=train_spearman_corr,
            validation_rmse=val_rmse,
            validation_r2=val_r2,
            validation_spearman=val_spearman_corr,
            test_rmse=test_rmse,
            test_r2=test_r2,
            test_spearman=test_spearman_corr,
            spearman=spearman_corr
        )

    def run_simulations(
        self,
        name: str,
        descrip: str = None,
        num_simulations: int = 1,
        num_rounds: int = 5,
        num_selected_variants_first_round: int = 10,
        num_top_acquistion_score_variants_per_round: int = 10,
        num_top_prediction_score_variants_per_round: int = 10,
        num_predictions_for_top_n_mean: int = 10,
        test_fraction: float = 0.2,
        random_seed: int = 42,
    ):

        assay_variants, assay_results = self._load_assay_data()
        assay_variants = self._embed_variants(assay_variants)
        (
            simulation_variants,
            simulation_assay_results,
            test_variants,
            test_assay_results,
        ) = self._split_assay_data(
            assay_variants, assay_results, test_fraction, random_seed
        )
        max_assay_score = self._get_max_assay_score(simulation_assay_results)
        run = self._create_run(
            name,
            descrip,
            num_rounds,
            len(assay_variants),
            num_selected_variants_first_round,
            num_top_acquistion_score_variants_per_round,
            num_top_prediction_score_variants_per_round,
            0,
            test_fraction,
            random_seed,
            max_assay_score
        )
        sub_runs = self._compile_sub_runs()

        for sub_run_params in sub_runs:
            sub_run = self._create_sub_run(run.id, sub_run_params)
            for sumulation_num in range(1, num_simulations + 1):
                self._init_learners_and_strategies(sub_run_params,
                                                   sumulation_num)
                self._run_simulation(
                    sub_run.id,
                    sub_run_params,
                    sumulation_num,
                    simulation_variants,
                    simulation_assay_results,
                    test_variants,
                    test_assay_results,
                    num_rounds,
                    num_selected_variants_first_round,
                    num_top_acquistion_score_variants_per_round,
                    num_top_prediction_score_variants_per_round,
                    num_predictions_for_top_n_mean,
                )

            self._end_sub_run(sub_run.id)
        self._end_run(run.id)

    def _run_simulation(
        self,
        sub_run_id: int,
        sub_run_params: SubRunParameters,
        simulation_num,
        simulation_variants,
        simulation_assay_results,
        test_variants,
        test_assay_results,
        num_rounds,
        num_selected_variants_first_round,
        num_top_acquistion_score_variants_per_round,
        num_top_prediction_score_variants_per_round,
        num_predictions_for_top_n_mean,
    ):
        """
        Runs a single simulation with the given parameters.
        """
        remaining_variants = simulation_variants.copy()
        train_variants = []
        train_assay_results = []
        simulation = self._create_simulation(
            sub_run_id, simulation_num)
        for round_num in range(1, num_rounds + 1):
            if len(remaining_variants) == 0:
                raise Exception(
                    "No more variants to select. " + "Could not complete round"
                )
            round = self._create_round(simulation.id, round_num)
            if round_num == 1:
                train_predictions = None
                remaining_predictions = [ModelPrediction(
                    variant_id=variant.id, score=0.0)
                    for variant in remaining_variants]
                if len(test_variants) > 0:
                    test_predictions = [ModelPrediction(
                        variant_id=variant.id, score=0.0)
                        for variant in test_variants]
                else:
                    test_predictions = None
                current_round_variant_data = \
                    self._select_top_variants(
                        sub_run_params.first_round_acquisition_strategy,
                        remaining_variants,
                        remaining_predictions,
                        num_selected_variants_first_round,
                        0,
                    )
            else:
                self._fit_model(sub_run_params.learner, train_variants,
                                train_assay_results)
                train_predictions = self._make_predictions(
                    sub_run_params.learner, train_variants
                    )
                remaining_predictions = self._make_predictions(
                    sub_run_params.learner, remaining_variants
                )
                if len(test_variants) > 0:
                    test_predictions = self._make_predictions(
                        sub_run_params.learner, test_variants)
                else:
                    test_predictions = None
                current_round_variant_data = \
                    self._select_top_variants(
                        sub_run_params.acquisition_strategy,
                        remaining_variants,
                        remaining_predictions,
                        num_top_acquistion_score_variants_per_round,
                        num_top_prediction_score_variants_per_round,
                    )
            current_round_variants = \
                [variant.variant for variant in
                    current_round_variant_data]
            current_round_assay_results = \
                self._get_assay_results_for_variants(
                    current_round_variants, simulation_assay_results
                )
            best_variant = self._get_best_variant(
                current_round_assay_results)
            remaining_variants_results = self._get_assay_results_for_variants(
                remaining_variants, simulation_assay_results
            )
            top_variants, top_variant_predictions, top_variant_results = \
                self._get_top_predicted_variants(
                    train_predictions, train_assay_results,
                    remaining_predictions,
                    remaining_variants_results,
                    simulation_variants,
                    num_predictions_for_top_n_mean
                )
            performance_metrics = self._compute_performance_metrics(
                train_predictions, train_assay_results,
                remaining_predictions,
                remaining_variants_results,
                test_predictions,
                test_assay_results,
                num_predictions_for_top_n_mean
            )
            train_variants.extend(current_round_variants)
            train_assay_results.extend(current_round_assay_results)
            remaining_variants = self._subtract_variant_lists(
                remaining_variants, current_round_variants
            )
            self._save_proposed_variants(
                round.id,
                current_round_variant_data,
                current_round_assay_results,
            )
            self._save_top_variants(round.id, top_variants,
                                    top_variant_predictions,
                                    top_variant_results)
            self._end_round(round.id, performance_metrics,
                            best_variant)
        self._end_simulation(simulation.id)

    def _compile_sub_runs(self) -> list[SubRunParameters]:
        """
        """
        sub_runs = []
        for simulation in self._sub_run_defs:
            for acquisition_strategy_config in \
                    simulation.acquisition_strategies:
                sub_run_params = SubRunParameters(
                    learner_name=simulation.learner.name,
                    learner_parameters=dict(simulation.learner.parameters),
                    learner_uses_embedder=simulation.learner.uses_embedder,
                    learner_uses_random_seed=
                    simulation.learner.uses_random_seed,
                    first_round_acquisition_strategy_name=
                    simulation.first_round_acquisition_strategy.name,
                    first_round_acquisition_strategy_parameters=
                    dict(simulation.first_round_acquisition_strategy.parameters),
                    first_round_acquisition_strategy_uses_random_seed=
                    simulation.first_round_acquisition_strategy.uses_random_seed,
                    acquisition_strategy_name=acquisition_strategy_config.name,
                    acquisition_strategy_parameters=
                    dict(acquisition_strategy_config.parameters),
                    acquisition_strategy_uses_random_seed=
                    acquisition_strategy_config.uses_random_seed,
                )
                sub_runs.append(sub_run_params)

        return sub_runs

    def _init_learners_and_strategies(self,
                                      sub_run_params: list[SubRunParameters],
                                      random_seed: int):
        """
        Initialize learners and acquisition strategies with proper random seeds.
        
        Args:
            sub_run_params: SubRunParameters object containing configuration
            random_seed: Random seed for reproducibility
        """
        # Initialize learner
        learner_factory = self._learner_factories.get(
            sub_run_params.learner_name)
        learner_params = sub_run_params.learner_parameters.copy()

        if sub_run_params.learner_uses_random_seed:
            learner_params['random_state'] = random_seed

        if sub_run_params.learner_uses_embedder:
            learner_params["embedder"] = self._embedder
        learner = learner_factory.create_instance(**learner_params)

        # Initialize first round acquisition strategy
        first_round_factory = self._acquisition_strategy_factories.get(
            sub_run_params.first_round_acquisition_strategy_name
        )
        first_round_params = \
            sub_run_params.first_round_acquisition_strategy_parameters.copy()

        if sub_run_params.first_round_acquisition_strategy_uses_random_seed:
            first_round_params['random_state'] = random_seed

        first_round_strategy = first_round_factory.create_instance(
            **first_round_params)

        # Initialize acquisition strategy
        strategy_factory = self._acquisition_strategy_factories.get(
            sub_run_params.acquisition_strategy_name
        )
        strategy_params = sub_run_params.acquisition_strategy_parameters.copy()

        if sub_run_params.acquisition_strategy_uses_random_seed:
            strategy_params['random_state'] = random_seed

        acquisition_strategy = strategy_factory.create_instance(
            **strategy_params)

        # Update sub_run_params with initialized objects
        sub_run_params.learner = learner
        sub_run_params.first_round_acquisition_strategy = first_round_strategy
        sub_run_params.acquisition_strategy = acquisition_strategy

    def _initialize(self):
        self._load_simulation_data()
        self.current_time = self.initial_time
        self.events = []
        self.entities = []
        self.resources = []
        self.statistics = {}
        self.status = "stopped"
        self.log = []
        self.random_seed = None

    def _create_simulation(self, sub_run_id: int, simulation_num: int
                           ) -> ALDESimulation:
        """
        Creates a new simulation record in the repository and returns it.
        """
        simulation = self._repository.add_simulation(
            sub_run_id=sub_run_id,
            simulation_num=simulation_num,
            start_ts=datetime.now(),
        )
        return simulation

    def _end_simulation(self, simulation_id: int):
        """
        Creates a new simulation record in the repository and returns it.
        """
        self._repository.end_simulation(
            id=simulation_id,
            end_ts=datetime.now(),
        )

    def _get_top_predicted_variants(
        self,
        train_predictions: list[ModelPrediction],
        train_assay_results: list[AssayResult],
        remaining_variants_predictions: list[ModelPrediction],
        remaining_variants_results: list[AssayResult],
        all_variants: list[Variant],
        num_top_proposed_variants: int
    ) -> Tuple[list[Variant], list[ModelPrediction], list[AssayResult]]:
        """
        Get the top predicted variants from combined train and remaining variants.
        
        Args:
            train_predictions: Predictions for training variants
            train_assay_results: Actual results for training variants
            remaining_variants_predictions: Predictions for remaining variants
            remaining_variants_results: Actual results for remaining variants
            num_top_proposed_variants: Number of top variants to return
        
        Returns:
            Tuple of (top_predictions, top_assay_results)
        """
        # Combine all predictions
        if train_predictions:
            all_predictions = train_predictions.copy()
            all_assay_results = train_assay_results.copy()
        else:
            all_predictions = []
            all_assay_results = []
        
        if remaining_variants_predictions:
            all_predictions.extend(remaining_variants_predictions)
            all_assay_results.extend(remaining_variants_results)
        
        # Sort by prediction score in descending order
        sorted_predictions = sorted(all_predictions, key=lambda x: x.score, reverse=True)
        top_predictions = sorted_predictions[:num_top_proposed_variants]
        
        # Get corresponding assay results
        top_variant_ids = {pred.variant_id for pred in top_predictions}
        top_assay_results = [result for result in all_assay_results
                             if result.variant_id in top_variant_ids]
        top_variants = [variant for variant in all_variants
                        if variant.id in top_variant_ids]
        top_predictions = sorted(top_predictions, key=lambda p: p.variant_id)
        top_assay_results = sorted(top_assay_results,
                                   key=lambda r: r.variant_id)
        top_variants = sorted(top_variants, key=lambda v: v.id)

        return top_variants, top_predictions, top_assay_results

    def _save_top_variants(self, round_id: int,
                           top_variants: list[Variant],
                           top_variant_predictions: list[ModelPrediction],
                           top_assay_results: list[AssayResult]):
        """
        Saves the top predicted variants for a given round.
        
        Args:
            round_id: ID of the current round
            top_variant_predictions: List of top variant predictions
            top_variant_results: List of corresponding assay results
        """
    
        for variant, prediction, assay_result in zip(
                top_variants, top_variant_predictions, top_assay_results):

            self._repository.add_round_top_variant(
                round_id=round_id,
                variant_id=variant.id,
                variant_name=variant.name,
                assay_score=assay_result.score,
                prediction_score=prediction.score,
                insert_ts=datetime.now(),
            )
