
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    results = results.groupby("position_bin", observed=True).agg(
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
        plotter, results, axes_top, axes_bottom,
        dataset,
        title):
    title = f"{dataset}/{title}"
    plotter.plot_2d_landscape_by_position_aa(axes_top, axes_bottom,
                                             results["position_bin"],
                                             results["assay_score"],
                                             # results["squared_error"],
                                             # "Squared Error",
                                             results["prediction_score"],
                                             "Prediction",
                                             results["count"],
                                             "Variant counts", title)

def show_protein_landscape_2d_old(
        repo, external_repo, plotter, axes, results,
        config_id, dataset,
        run_name, title, standardize_scores: bool = False) -> int:
    y_value_lists = [results["assay_score"], results["prediction_score"]]
    line_labels = ["Prediction", "Assay"]
    y_value_lists = [results["assay_score"], results["squared_error"]]
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
    results = prep_results_for_protein_landscape(results, 200)
    plotter.plot_3d_landscape_by_position_aa(axes, results["mutation_position"],
                                             results["mutation_aa_index"],
                                             results["squared_error"], title)
    return len(results)


def show_plots_old(show_plot_func, datasets, projection='rectilinear'):
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


def create_grid_row(grid, fig, row_ind, col_ind):
    # Create an inner grid with 2 rows (vertical subplots) inside each outer cell
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=grid[row_ind, col_ind], hspace=0.1
    )
    
    # Top subplot
    axes_top = fig.add_subplot(inner_grid[0])
    axes_top.grid(False)
    
    # Bottom subplot
    axes_bottom = fig.add_subplot(inner_grid[1])
    axes_bottom.grid(False)
    return axes_top, axes_bottom    


def show_protein_landscape_plots(show_plot_func, datasets, projection='rectilinear'):
    container = ALDEContainer("./config/msaldem.yaml")
    # container = ALDEContainer("./config/msalde.yaml")
    repo = container.query_repository
    external_repo = container.external_repository
    plotter = container.plotter

    datasets_ = datasets
    # datasets_ = datasets[:5]

    config_ids = ["c10", "c3_1", "c3_2"]
    run_names = ["ESM2_LLR", "RF_AL", "RFTRAIN_ALL"]
    titles = ["LogLikelihood", "RandomForest AL", "RandomForest Train20%"]
    standardize_scores = [True, False, False]
    num_models = len(config_ids)

    num_rows = len(datasets_ * num_models)

    fig = plt.figure(figsize=(20, 10*num_rows))
    fig.patch.set_facecolor('white')
    grid = gridspec.GridSpec(num_rows, 1, figure=fig, hspace=.2) # wspace=0.4, hspace=0.4)

    #fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
    #                         subplot_kw={'projection': projection})
    #axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    col_ind = 0
    row_ind = 0
    for dataset in datasets_:
        # increment_row_id = True
        for model_ind in range(num_models):
            config_id = config_ids[model_ind]
            run_name = run_names[model_ind]
            title = titles[model_ind]
            standardize_scores_flag = standardize_scores[model_ind]
            results = repo.get_last_round_scores_by_config_dataset_run(
                config_id=config_id, dataset_name=dataset, run_name=run_name)
            if len(results) == 0:
                # increment_row_id = False
                break
            results = prep_results_for_protein_landscape(
                results, standardize_scores_flag, 400)
            axes_top, axes_bottom = create_grid_row(grid, fig, row_ind,
                                                    col_ind)

            show_plot_func(plotter, results, axes_top, 
                           axes_bottom, dataset, title)
            row_ind += 1

        # if increment_row_id:
        #     row_ind += 1

    # for j in range(ind, len(axes)):
    #    fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def show_mean_activity_by_round_comparison_plots(datasets,
                                        llr_config_id,
                                        llr_run_name,
                                        config_ids, run_names,
                                        labels=None):

    container = ALDEContainer("./config/msaldem.yaml")
    repo = container.query_repository
    plotter = container.plotter

    datasets_ = datasets
    # datasets_ = ["cas12f2"]

    num_rows = len(datasets_) // 3 + int(len(datasets_) % 3 > 0)

    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*num_rows))
    axes = axes.flatten()
    ind = 0

    for dataset in datasets_:
        llr_results = repo.get_mean_activity_of_top_variants_by_round(
            config_id=llr_config_id, dataset_name=dataset,
            run_name=llr_run_name)
        if len(llr_results) == 0:
            continue
        llr_results = llr_results[llr_results["round_num"] == 2]
        llr_top_mean_activity = llr_results["mean_score"].values[0]
        results_list = []
        results_labels = []
        for i in range(len(config_ids)):
            config_id = config_ids[i]
            run_name = run_names[i]
            label = labels[i] if labels is not None else ""
            results = repo.get_mean_activity_of_top_variants_by_round(
                config_id=config_id, dataset_name=dataset, run_name=run_name)
            if len(results) == 0:
                continue
            for strategy in results['strategy_name'].unique():
                strategy_results = results[
                    results['strategy_name'] == strategy]
                results_list.append(strategy_results)
                if label == "":
                    results_labels.append(strategy)
                else:
                    results_labels.append(f"{label}/{strategy}")
        #  show_plot_func(axes[ind], results_list, results_labels,
        #               llr_top_mean_activity, dataset)
        plotter.plot_mean_activity_by_round(
           axes[ind], results_list, results_labels,
           llr_top_mean_activity, dataset)
        ind += 1

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Mean Assay Activity of Top 16 Predicted Variants by Round (100 acquired variants/round)', fontsize=20, y=1.0)
    plt.tight_layout()
    plt.show()

