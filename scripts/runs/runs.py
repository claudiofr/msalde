import context  # noqa: F401 E402
from msalde.container import ALDEContainer
import argparse


def get_alde_container():
    return ALDEContainer("./config/msaldem.yaml")


def create_parser():
    parser = argparse.ArgumentParser(description="Run experiments with different combinations of grid search variables.")
    parser.add_argument("--config_id", type=str, help="Config id")
    return parser


def run_simulation1(simulator, configid):
    simulator.run_simulations(
        config_id=configid,
        name="test_run",
        num_simulations=3,
        # num_simulations=1,
        num_rounds=100,
        # num_rounds=1,
        num_selected_variants_first_round=16,
        num_top_acquistion_score_variants_per_round=32,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42)


def main():
    parser = create_parser()
    args = parser.parse_args()
    configid = args.config_id
    simulator = get_alde_container().simulator
    run_simulation1(simulator, configid)


if __name__ == "__main__":
    main()
