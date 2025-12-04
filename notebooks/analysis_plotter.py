
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from msalde.container import ALDEContainer

DATASETS = [
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



aminos = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q',
          'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']

def get_amino_acid_index(aa: str) -> int:
    return aminos.index(aa)

def prep_results_for_protein_landscape(results, standardize_scores = False,
                                       num_bins=50):
    if standardize_scores:
        results['assay_score'] = (
            (results['assay_score'] - results['assay_score'].mean()) /
            results['assay_score'].std())
        results['prediction_score'] = (
            (results['prediction_score'] - results['prediction_score'].mean()) /
            results['prediction_score'].std())
    results['squared_error'] = (results['assay_score'] -
                                results['prediction_score']) ** 2
    results['mutation_position'] = results['variant_id'].apply(
        lambda vid: int(vid[1:-1]) - 1)
    results['mutation_aa_index'] = results['variant_id'].apply(
        lambda vid: get_amino_acid_index(vid[-1]))
    results['mutation_position_aa_index'] = round((
        results['mutation_position'] * len(aminos) +
        results['mutation_aa_index']) / len(aminos), 2)
    results = results.sort_values(by=['mutation_position_aa_index'])
    max_position = results['mutation_position'].max()
    bins = np.arange(0, max_position + 1, max(1, max_position // num_bins))
    labels = [(bins[i] + bins[i+1]) // 2 for i in range(len(bins)-1)]
    results["position_bin"] = pd.cut(results["mutation_position"],
                                     bins=bins, labels=labels)
    results = results.groupby("position_bin").agg(
        # assay_score=("assay_score", "max"),
        assay_score=("assay_score", "mean"),
        # prediction_score=("prediction_score", "max"),
        prediction_score=("prediction_score", "mean"),
        # squared_error=("squared_error", "max"),
        squared_error=("squared_error", "mean"),
        count=("variant_id", "count")
    ).reset_index()
    return results


def show_protein_landscape_2d(
        repo, external_repo, plotter, axes, config_id, dataset,
        run_name, title, standardize_scores: bool = False) -> int:
    results = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id, dataset_name=dataset, run_name=run_name)
    if len(results) == 0:
        return 0
    results = prep_results_for_protein_landscape(results, standardize_scores)
    y_value_lists = [results["prediction_score"], results["assay_score"]]
    line_labels = ["Prediction", "Assay"]
    y_value_lists = [results["squared_error"], results["assay_score"]]
    line_labels = ["Prediction Error", "Assay"]
    title = f"{dataset}/{title}"
    plotter.plot_2d_landscape_by_position_aa(axes, results["position_bin"],
                                             y_value_lists, line_labels,
                                             results["count"],
                                             "Variant counts", title)
                                             # results["squared_error"], title)
    return len(results)

def show_protein_landscape_3d(
        repo, external_repo, plotter, axes, config_id, dataset,
        run_name, title) -> int:
    results = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id, dataset_name=dataset, run_name=run_name)
    if len(results) == 0:
        return 0
    results = prep_results_for_protein_landscape(results)
    plotter.plot_3d_landscape_by_position_aa(axes, results["mutation_position"],
                                             results["mutation_aa_index"],
                                             results["squared_error"], title)
    return len(results)


def show_plots(show_plot_func, datasets, projection='rectilinear'):
    container = ALDEContainer("./config/msaldem.yaml")
    # container = ALDEContainer("./config/msalde.yaml")
    repo = container.query_repository
    external_repo = container.external_repository
    plotter = container.plotter

    datasets_ = datasets # [:5]

    num_rows = len(datasets_)

    fig = plt.figure()
    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
                             subplot_kw={'projection': projection})
    axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    ind = 0
    for dataset in datasets_:
        if not show_plot_func(repo, external_repo, plotter, axes[ind], 
                                     "c10", dataset, "ESM2_LLR",
                                     "LogLikelihood", standardize_scores=True):
            continue
        ind += 1
        show_plot_func(repo, external_repo, plotter, axes[ind],
                       "c3_1", dataset, "RF_AL",
                                     "RandomForest AL")
        ind += 1
        show_plot_func(repo, external_repo, plotter, axes[ind],
                       "c3_2", dataset, "RFTRAIN_ALL",
                                     "RandomForest Train20%")
        ind += 1

    for j in range(ind, len(axes)):
        fig.delaxes(axes[j])

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()
