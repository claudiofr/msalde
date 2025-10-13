import context  # noqa: F401 E402
from msalde.container import ALDEContainer
import argparse
import pandas as pd

datasets = [
    "ADRB2",
    "AICDA",
    "BRCA1",
    "BRCA2",
    "CALM1",
    "CAR11",
    "CASP3",
    "CASP7",
    "CBS",
    "GDIA",
    "GRB2",
    "HEM3",
    "HMDH",
    "HXK4",
    "KCNE1",
    "KCNH2",
    "MET",
    "MK01",
    "MSH2",
    "MTHR",
    "NPC1",
    "OTC",
    "P53",
    "PAI1",
    "PPARG",
    "PPM1D",
    "PTEN",
    "RAF1",
    "RASH",
    "S22A1",
    "SC6A4",
    "SCN5A",
    "SERC",
    "SHOC2",
    "SRC",
    "SUMO1",
    "SYUA",
    "TADBP",
    "TPK1",
    "TPOR",
    "UBC9",
    "VKOR1",
    "brenan",
    "cas12f",
    "cov2_S",
    "doud",
    "giacomelli",
    "haddox",
    "jones",
    "kelsic",
    "lee",
    "markin",
    "stiffler",
    "zikv_E"]

label_dir = "/sc/arion/work/fratac01/data/al/dms"


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


def main():
    # parser = create_parser()
    # args = parser.parse_args()
    # configid = args.config_id
    # dataset = args.dataset
    simulator = get_alde_container().simulator
    # run_simulation1(simulator, configid)
    # datasets=["gb1", "aav", "sars2", "calmodulin"]
    datasets_ = datasets[:1]
    for dataset in datasets_:
        df = pd.read_csv(f"{label_dir}/{dataset}_labels.csv")
        if df.shape[0] < 1000:
            print(f"Skipping {dataset} with {df.shape[0]} variants")
            continue
        print(f"Running {dataset} with {df.shape[0]} variants")
        run_simulation_mc(simulator, "c3_1", dataset,
                          num_rounds=5,
                          num_simulations=5,
                          num_selected_variants_first_round=16,
                          num_top_acquisition_score_variants_per_round=100)
        run_simulation_mc(simulator, "c3_2", dataset,
                          num_rounds=2,
                          num_simulations=5,
                          num_selected_variants_first_round=15000,
                          num_top_acquisition_score_variants_per_round=100)
        run_simulation_mc(simulator, "c10", dataset,
                          num_rounds=2,
                          num_simulations=1,
                          num_selected_variants_first_round=1,
                          num_top_acquisition_score_variants_per_round=100)


if __name__ == "__main__":
    main()
