import context  # noqa: F401 E402
from msalde.container import ALDEContainer
import argparse


def get_alde_container():
    return ALDEContainer("./config/msaldem.yaml")


def create_parser():
    parser = argparse.ArgumentParser(description="Run experiments with different combinations of grid search variables.")
    parser.add_argument("--config_id", type=str, help="Config id")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--num_vars", type=int, help="Num vars first round")
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

def run_simulation_mc(simulator, configid, dataset, num_vars):
    simulator.run_simulations(
        config_id=configid,
        name="test_run",
        num_simulations=5,
        # num_simulations=1,
        num_rounds=2,
        # num_rounds=1,
        num_selected_variants_first_round=num_vars,
        num_top_acquistion_score_variants_per_round=1,
        num_top_prediction_score_variants_per_round=0,
        num_predictions_for_top_n_mean=16,
        test_fraction=0, # 0.2,
        #test_fraction=0.2,
        random_seed=42,
        dataset_name=dataset)


def main():
    parser = create_parser()
    args = parser.parse_args()
    configid = args.config_id
    dataset = args.dataset
    num_vars = round(args.num_vars * 0.8)
    simulator = get_alde_container().simulator
    # run_simulation1(simulator, configid)
    print("dataset", dataset)
    run_simulation_mc(simulator, configid, dataset, num_vars)


if __name__ == "__main__":
    main()
