import context  # noqa: F401
import pytest
from msalde.simulator import DESimulator


def test_compute_metrics_col_name_map_include_ve_sources(
        de_simulator: DESimulator):
    de_simulator.run_simulations(
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


