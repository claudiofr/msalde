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


def run_simulation_mc(simulator, configid, dataset,
                      num_rounds, num_simulations,
                      num_selected_variants_first_round,
                      num_top_acquisition_score_variants_per_round):
    simulator.run_simulations(
        config_id=configid,
        name="test_run",
        # num_simulations=3, # 5,
        num_simulations=num_simulations,
        num_rounds=num_rounds,
        # num_rounds=1,
        num_selected_variants_first_round=num_selected_variants_first_round,
        num_top_acquistion_score_variants_per_round=
        num_top_acquisition_score_variants_per_round,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42,
        dataset_name=dataset,
        save_last_round_predictions=True,)


def test_run_simulations_llr(
        de_simulator: DESimulator):
    run_simulation_mc(de_simulator, "c10", "cas12f2", #  "brenan", # "cas12f2",
                      num_rounds=2,
                      num_simulations=1,
                      num_selected_variants_first_round=1,
                      num_top_acquisition_score_variants_per_round=100)
    pass


def test_run_simulations_al(
        de_simulator: DESimulator):
    run_simulation_mc(de_simulator, "c3_1", "cas12f2",
                    num_rounds=5,
                    num_simulations=5,
                    num_selected_variants_first_round=5,
                    num_top_acquisition_score_variants_per_round=5)
    pass


def test_run_simulations_all(
        de_simulator: DESimulator):
    run_simulation_mc(de_simulator, "c3_2", "cas12f2",
                          num_rounds=2,
                          num_simulations=5,
                          num_selected_variants_first_round=15000,
                          num_top_acquisition_score_variants_per_round=5)
    pass


def test_run_simulations_hf(
        de_simulator: DESimulator):
    run_simulation_mc(de_simulator, "c9_2", "cas12f2",
                    num_rounds=5,
                    num_simulations=5,
                    num_selected_variants_first_round=5,
                    num_top_acquisition_score_variants_per_round=5)
    pass
