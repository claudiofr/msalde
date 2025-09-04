import context  # noqa: F401
import pytest
from msalde.simulator import DESimulator


def test_run_simulations(
        de_simulator: DESimulator):
    de_simulator.run_simulations(
        config_id="c1",
        name="test_run",
        num_simulations=2,
        num_rounds=5,
        num_selected_variants_first_round=16,
        num_top_acquistion_score_variants_per_round=16,
        num_top_prediction_score_variants_per_round=16,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42)
    pass


def test_run_simulations1(
        de_simulator1: DESimulator):
    de_simulator1.run_simulations(
        name="test_run",
        num_simulations=2,
        num_rounds=2,
        num_selected_variants_first_round=2,
        num_top_acquistion_score_variants_per_round=2,
        num_top_prediction_score_variants_per_round=2,
        num_predictions_for_top_n_mean=2,
        test_fraction=0.2,
        random_seed=42)
    pass



