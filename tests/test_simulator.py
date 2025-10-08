import context  # noqa: F401
import pytest
from msalde.simulator import DESimulator


def test_run_simulations(
        de_simulator: DESimulator):

    de_simulator.run_simulations(
        #config_id="c1",
        config_id="c2_1",
        name="test_run",
        # num_simulations=3,
        num_simulations=2,
        # num_rounds=10,
        num_rounds=3,
        num_selected_variants_first_round=16,
        num_top_acquistion_score_variants_per_round=16,
        num_top_prediction_score_variants_per_round=16,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42,
        dataset_name="cas12f2")
    pass


def test_run_simulations1(
        de_simulator: DESimulator):
    de_simulator.run_simulations(
        #config_id="c1",
        config_id="c3",
        name="test_run",
        # num_simulations=3,
        num_simulations=5,
        # num_rounds=10,
        num_rounds=2,
        num_selected_variants_first_round=6200,
        num_top_acquistion_score_variants_per_round=1,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42)
    pass



def test_run_simulations2(
        de_simulator: DESimulator):
    de_simulator.run_simulations(
        #config_id="c1",
        config_id="c3_1",
        name="test_run",
        # num_simulations=3,
        num_simulations=2,
        # num_rounds=10,
        num_rounds=3,
        num_selected_variants_first_round=16,
        num_top_acquistion_score_variants_per_round=16,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42,
        dataset_name="cas12f2",
        save_last_round_predictions=True,)
    pass
