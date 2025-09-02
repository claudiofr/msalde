from msalde.simulator import DESimulator
from msalde.container import ALDEContainer
import cProfile


def test_run_simulations():
    container = ALDEContainer(config_file="./config/msalde.yaml")
    de_simulator: DESimulator = container.simulator

    de_simulator.run_simulations(
        name="test_run",
        num_simulations=5,
        num_rounds=10,
        num_selected_variants_first_round=16,
        num_top_acquistion_score_variants_per_round=16,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42)
    pass


cProfile.run('test_run_simulations()')
