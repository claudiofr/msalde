
import gc
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from msalde.container import ALDEContainer
from msalde.ml_util import calculate_optimal_youden_index
from msalde.plotter import ALDEPlotter
from msalde.variant_util import variant_id_to_position

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

def prep_results_for_protein_landscape_old(results, standardize_scores = False,
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
    min_position = results['mutation_position'].min()
    num_positions = max_position - min_position + 1
    if len(results) < num_bins:
        results["position_bin"] = results['mutation_position_aa_index']
    else:
        bins = np.arange(min_position-1, max_position + 1,
                     max(1, num_positions // num_bins))
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

def prep_results_for_protein_landscape(results, standardize_scores=False):
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
    # Not sure I need this column anymore but keeping for now
    results['mutation_position_aa_index'] = round((
        results['mutation_position'] * len(aminos) +
        results['mutation_aa_index']) / len(aminos), 2)
    results = results.groupby("mutation_position", observed=True).agg(
        # assay_score=("assay_score", "max"),
        assay_score=("assay_score", "mean"),
        # prediction_score=("prediction_score", "max"),
        prediction_score=("prediction_score", "mean"),
        # squared_error=("squared_error", "max"),
        squared_error=("squared_error", "mean"),
        count=("variant_id", "count")
    ).reset_index()
    return results.sort_values(by=['mutation_position'])

def show_protein_landscape_2d(
        plotter, results, axes_top, axes_middle, axes_bottom,
        dataset,
        ss_track: list, residue_nums: list,
        domains: list,
        title):
    title = f"{dataset}/{title}"

    plotter.plot_2d_landscape_by_position_aa(axes_top,
                                             axes_middle, axes_bottom,
                                             results["mutation_position"],
                                             results["assay_score"],
                                             # results["squared_error"],
                                             # "Squared Error",
                                             results["prediction_score"],
                                             "Prediction",
                                             results["count"],
                                             "Variant counts",
                                             ss_track,
                                             residue_nums,
                                             domains,
                                             title)


def compute_gof_lof_by_quantile(results: pd.DataFrame, class_label_column: str,
                                lof_quantile: float = 0.5,
                                gof_quantile: float = 0.5
                                ) -> pd.DataFrame:
    results = results.copy()
    gof_threshold = results['assay_score'].quantile(gof_quantile)
    lof_threshold = results['assay_score'].quantile(lof_quantile)
    results[class_label_column] = np.where(
        results["assay_score"] > gof_threshold, 1,
        np.where(results["assay_score"] <= lof_threshold, 0, np.nan)
    )
    return_val = results[results[class_label_column].notna()]
    return return_val


def get_compute_gof_lof_by_quantile_func(lof_quantile: float = 0.5,
                                         gof_quantile: float = 0.5
                                         ) -> callable:
    return lambda results, class_label_name: compute_gof_lof_by_quantile(
        results, class_label_name, lof_quantile=lof_quantile, gof_quantile=gof_quantile)


def compute_gof_lof_by_activity_threshold(
        results: pd.DataFrame, class_label_column: str,
        lof_threshold: float = 0.5,
        gof_threshold: float = 0.5
        ) -> pd.DataFrame:
    results = results.copy()
    results[class_label_column] = np.where(
        results["assay_score"] > gof_threshold, 1,
        np.where(results["assay_score"] < lof_threshold, 0, np.nan)
    )
    return results[results[class_label_column].notna()]


def compute_gof_lof_by_z_score_q_value(
        results: pd.DataFrame, class_label_column: str,
        log2fc_threshold: float = 0.3,
        q_threshold: float = 0.05
        ) -> pd.DataFrame:
    baseline = results["assay_score"].median()
    mad = (results["assay_score"] - results["assay_score"].mean()).abs().mean()
    baseline_sigma = (
        mad * 1.4826   # convert MAD to SD
    )

    results["z_score"] = (
        (results["assay_score"] - baseline) /
        baseline_sigma
    )
    results["p_value_lof"] = norm.cdf(results["z_score"])
    results["p_value_gof"] = 1 - results["p_value_lof"]
    results = results[results["p_value_lof"].notna()]

    results["q_value_gof"] = multipletests(
        results["p_value_gof"],
        method="fdr_bh"
    )[1]
    results["q_value_lof"] = multipletests(
        results["p_value_lof"],
        method="fdr_bh"
    )[1]
    results["log2fc_baseline"] = np.log2(
        results["assay_score"] / baseline
    )

    # putative_gof: 1 = GOF, 0 = LOF, nan = neither
    results[class_label_column] = np.where(
        (results["log2fc_baseline"] >= log2fc_threshold) &
        (results["q_value_gof"] < q_threshold),
        1,
        np.where(
            (results["log2fc_baseline"] <= -log2fc_threshold) &
            (results["q_value_lof"] < q_threshold),
            0,
            np.nan
        )
    )
    return results[results[class_label_column].notna()]



def show_last_round_mse_by_domain(
        repo, plotter: ALDEPlotter, axes, config_id,
        run_name,
        assay_source,
        title,
        domains: list,
        ) -> int:
    results = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id,
        dataset_name=assay_source,
        run_name=run_name)
    if len(results) == 0:
        return 0
    results['position'] = results['variant_id'].apply(
        variant_id_to_position)
    
    # mse is mean signed error
    gc.collect(1)
    # Add class label column to results

    mse_rows = []
    for domain in domains:
        domain_results = results[results["position"].between(
            domain["start"], domain["end"])]
        if len(domain_results) == 0:
            continue
        mse = np.mean(domain_results['prediction_score'] -
                      domain_results['assay_score'])
        mse_rows.append(
            {
            "domain": domain,
            "metric": mse
            })

    mse_results = pd.DataFrame(mse_rows)
    plotter.plot_metric_by_domain(axes, mse_results,
                                        "Mean Signed Error",
                                 title=title)
    return len(results)


def show_last_round_auc_by_domain(
        repo, var_repo, plotter: ALDEPlotter, axes, config_id,
        run_name,
        assay_source, protein_symbol,
        plot_info_list,
        class_label_column, title,
        domains: list,
        ) -> int:
    results_all = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id,
        dataset_name=assay_source,
        run_name=run_name)
    if len(results_all) == 0:
        return 0
    results_all['position'] = results_all['variant_id'].apply(
        variant_id_to_position)
    
    auc_results_list = []
    for plot_info in plot_info_list:
        gc.collect(1)
        results = results_all.copy()
        # Add class label column to results
        if plot_info.get('compute_class_label_func', None) is not None:
            assay_df = plot_info['compute_class_label_func'](
                results[["variant_id", "assay_score"]].drop_duplicates(),
                class_label_column
            )
        else:
            assay_df = var_repo.get_variant_assay(
                assay_source=assay_source,
                protein_symbol=protein_symbol,
                assay_type=plot_info["assay_type"],
                assay_subtype=plot_info.get("assay_subtype", None),
            )
            assay_df = assay_df[~assay_df[class_label_column].isna()]
            if len(assay_df) == 0:
                return 0
        results[class_label_column] = results["variant_id"].map(
            assay_df.set_index("variant_id")[class_label_column])
        results = results[~results[class_label_column].isna()]
        if len(results) == 0:
            return 0
        # num_positive = results[class_label_column].sum()
        # num_negative = len(results) - num_positive

        auc_rows = []
        for domain in domains:
            domain_results = results[results["position"].between(
                domain["start"], domain["end"])]
            if len(domain_results) == 0:
                continue
            optimal_youden_index, fpr, tpr, _ = calculate_optimal_youden_index(
                    domain_results[class_label_column].values,
                    domain_results['prediction_score'].values)
            auc = roc_auc_score(domain_results[class_label_column].values,
                                domain_results['prediction_score'].values)
            num_positive = domain_results[class_label_column].sum()
            num_negative = len(domain_results) - num_positive
            auc_rows.append(
                {"metric": auc,
                "optimal_youden_index": optimal_youden_index,
                "domain": domain,
                "num_positive": num_positive,
                "num_negative": num_negative})

        auc_results = pd.DataFrame(auc_rows)
        auc_results_list.append({"label": plot_info['label'],
                                 "results": auc_results})
    plotter.plot_metric_by_domain_multi(axes, auc_results_list,
                                        "AUC",
                                 title=f"{assay_source}/{run_name}")
    return len(results)



def show_roc_auc_by_round(
        repo, var_repo, plotter: ALDEPlotter, axes, config_id,
        run_name,
        assay_source, protein_symbol,
        plot_info_list,
        class_label_column, title,
        llr_config_id=None,
        llr_run_name=None) -> int:
    llr_results_all = repo.get_variant_scores_by_simulation_round(
        config_id=llr_config_id,
        dataset_name=assay_source,
        run_name=llr_run_name)
    if len(llr_results_all) == 0:
        return 0
    llr_results_all = llr_results_all[llr_results_all["round_num"] == 2]
    
    results_all = repo.get_variant_scores_by_simulation_round(
        config_id=config_id,
        dataset_name=assay_source,
        run_name=run_name)
    if len(results_all) == 0:
        return 0

    auc_results_list = []
    for plot_info in plot_info_list:
        gc.collect(1)
        results = results_all.copy()
        llr_results = llr_results_all.copy()
        # Add class label column to results
        if plot_info.get('compute_class_label_func', None) is not None:
            assay_df = plot_info['compute_class_label_func'](
                results[["variant_id", "assay_score"]].drop_duplicates(),
                class_label_column
            )
        else:
            assay_df = var_repo.get_variant_assay(
                assay_source=assay_source,
                protein_symbol=protein_symbol,
                assay_type=plot_info["assay_type"],
                assay_subtype=plot_info.get("assay_subtype", None),
            )
            assay_df = assay_df[~assay_df[class_label_column].isna()]
            if len(assay_df) == 0:
                return 0
        results[class_label_column] = results["variant_id"].map(
            assay_df.set_index("variant_id")[class_label_column])
        results = results[~results[class_label_column].isna()]
        if len(results) == 0:
            return 0
        llr_results[class_label_column] = llr_results["variant_id"].map(
            assay_df.set_index("variant_id")[class_label_column]
        )
        llr_results = llr_results[~llr_results[class_label_column].isna()]
        if len(llr_results) == 0:
            return 0
        llr_auc = roc_auc_score(llr_results[class_label_column].values,
                            llr_results['prediction_score'].values)
        results_sim1_round1 = results[(results["simulation_num"] == 1) &
                                    (results["round_num"] == 1)]
        num_positive = results_sim1_round1[class_label_column].sum()
        num_negative = len(results_sim1_round1) - num_positive

        """
        for (round_num, simulation_id), group_results in results.groupby(
                ['round_num', 'simulation_id'], sort=True):
            optimal_youden_index, fpr, tpr, _ = calculate_optimal_youden_index(
                    group_results[class_label_column].values,
                    group_results['prediction_score'].values)
            auc = roc_auc_score(group_results[class_label_column].values,
                                group_results['prediction_score'].values)
            auc_rows.append(
                {"round_num": round_num,
                "simulation_id": simulation_id,
                "auc": auc,
                "optimal_youden_index": optimal_youden_index,
            })
        """

        round_results = results.groupby(["variant_id", "round_num"]).agg(
            mean_prediction=("prediction_score", "mean"),
            label=(class_label_column, "max")
        ).reset_index()
        auc_rows = []
        for round_num, group_results in round_results.groupby("round_num"):
            optimal_youden_index, fpr, tpr, _ = calculate_optimal_youden_index(
                    group_results['label'].values,
                    group_results['mean_prediction'].values)
            auc = roc_auc_score(group_results['label'].values,
                                group_results['mean_prediction'].values)
            auc_rows.append(
                {"round_num": round_num,
                "auc": auc,
                "optimal_youden_index": optimal_youden_index,
                })

        auc_results = pd.DataFrame(auc_rows)
        auc_results_list.append({"label": plot_info['label'],
                                 "auc_results": auc_results,
                                 "llr_auc": llr_auc,
                                 "num_positive": num_positive,
                                 "num_negative": num_negative})
    plotter.plot_roc_auc_by_round_multi(axes, auc_results_list,
                                 title=f"{assay_source}/{run_name}")
    return len(results)


def show_roc_prediction_assay_curve(
        repo, var_repo, plotter: ALDEPlotter, axes, config_id,
        run_name,
        assay_source, protein_symbol,
        assay_type, assay_subtype,
        class_label_column, title,
        compute_class_label_func=None) -> int:
    results = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id,
        dataset_name=assay_source,
        run_name=run_name)
    if len(results) == 0:
        return 0
    if compute_class_label_func is not None:
        results = compute_class_label_func(results, class_label_column)
    else:
        assay_df = var_repo.get_variant_assay(
            assay_source=assay_source,
            protein_symbol=protein_symbol,
            assay_type=assay_type,
            assay_subtype=assay_subtype,
        )
        assay_df = assay_df[~assay_df[class_label_column].isna()]
        if len(assay_df) == 0:
            return 0
        results = results.merge(assay_df[['variant_id', class_label_column]],
                                on='variant_id', how='inner')
    if len(results) == 0:
        return 0
    optimal_youden_index, fpr, tpr, _ = calculate_optimal_youden_index(
            results[class_label_column].values,
            results['prediction_score'].values)

    auc = roc_auc_score(results[class_label_column].values,
                        results['prediction_score'].values)
    num_positive = results[class_label_column].sum()
    num_negative = len(results) - num_positive
    plotter.plot_roc_curve(axes, fpr, tpr, auc, 
                           fpr[optimal_youden_index],
                           title=f"{assay_source}/{run_name} (AUC={auc:.3f})",
                           num_positive=num_positive,
                           num_negative=num_negative)
    return len(results)



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
    # Create an inner grid with 3 rows (vertical subplots) inside each outer cell
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=grid[row_ind, col_ind], hspace=0.2,
        height_ratios=[1, 4, 4],
    )
    
    # Top subplot
    axes_top = fig.add_subplot(inner_grid[0])
    axes_top.grid(False)
    
    # Bottom subplot
    axes_middle = fig.add_subplot(inner_grid[1])
    axes_middle.grid(False)
    axes_bottom = fig.add_subplot(inner_grid[2])
    axes_bottom.grid(False)
    return axes_top, axes_middle, axes_bottom    


def show_protein_landscape_plots(show_plot_func, datasets, projection='rectilinear'):
    container = ALDEContainer("./config/msaldem.yaml")
    # container = ALDEContainer("./config/msalde.yaml")
    repo = container.query_repository
    pdb_repo = container.pdb_repository
    plotter = container.plotter

    datasets_ = datasets
    # datasets_ = datasets[:5]

    config_ids = ["c10", "c3_1", "c3_2"]
    run_names = ["ESM2_LLR", "RF_AL", "RFTRAIN_ALL"]
    run_names = ["ESM2_LLR_ALL_PRED", "RF_AL_ALL_PRED", "RFTRAIN_ALL_ALL_PRED"]
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
                results, standardize_scores_flag)
            secondary_structure_info = pdb_repo.get_secondary_structure(
                dataset)
            axes_top, axes_middle, axes_bottom = create_grid_row(
                grid, fig, row_ind, col_ind)

            show_plot_func(plotter, results, axes_top, 
                           axes_middle, axes_bottom, dataset, 
                           secondary_structure_info[0],
                           secondary_structure_info[1],
                           secondary_structure_info[2],
                           title)
            row_ind += 1

        # if increment_row_id:
        #     row_ind += 1

    # for j in range(ind, len(axes)):
    #    fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def show_roc_prediction_assay_plots(show_plot_func, datasets, assay_types,
                                    compute_class_label_funcs):
    container = ALDEContainer("./config/msaldem.yaml")
    # container = ALDEContainer("./config/msalde.yaml")
    repo = container.query_repository
    var_repo = container.variant_repository
    plotter = container.plotter

    datasets_ = datasets
    # datasets_ = datasets[:5]

    config_ids = ["c10", "c3_1", "c3_2"]
    run_names = ["ESM2_LLR", "RF_AL", "RFTRAIN_ALL"]
    run_names = ["ESM2_LLR_ALL_PRED", "RF_AL_ALL_PRED", "RFTRAIN_ALL_ALL_PRED"]
    titles = ["LogLikelihood", "RandomForest AL", "RandomForest Train20%"]
    standardize_scores = [True, False, False]
    num_models = len(config_ids)

    num_cols = 3
    num_rows_per_dataset = num_models// num_cols + int(num_models % num_cols > 0)
    num_rows = len(datasets_) * num_rows_per_dataset

    fig = plt.figure(figsize=(20, 5*num_rows))
    fig.patch.set_facecolor('white')
    grid = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=.2) # wspace=0.4, hspace=0.4)

    #fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
    #                         subplot_kw={'projection': projection})
    #axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    row_ind = 0
    col_ind = 0
    for i, dataset in enumerate(datasets_):
        # increment_row_id = True
        for model_ind in range(num_models):
            config_id = config_ids[model_ind]
            run_name = run_names[model_ind]
            title = titles[model_ind]
            standardize_scores_flag = standardize_scores[model_ind]
            compute_class_label_func = compute_class_label_funcs[i]
            # Add a subplot to the grid at (row_ind, col_ind)
            axes = fig.add_subplot(grid[row_ind, col_ind])
            results_count = show_plot_func(
                repo, var_repo, plotter, axes, config_id,
                run_name,
                dataset, dataset,
                assay_types[i], None,
                class_label_column="class_label",
                title=f"{dataset}/{title}",
                compute_class_label_func=compute_class_label_func)
            if results_count == 0:
                break
            col_ind += 1
            if col_ind >= num_cols:
                col_ind = 0
                row_ind += 1

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


def show_auc_by_round_plots(show_plot_func, dataset_plot_info,
                                    llr_config_id,
                                    llr_run_name):

    container = ALDEContainer("./config/msaldem.yaml")
    repo = container.query_repository
    plotter = container.plotter
    var_repo = container.variant_repository

    config_ids = ["c3_1"]
    run_names = ["RF_AL_ALL_PRED"]
    titles = ["RandomForest AL"]
    standardize_scores = [False]
    num_models = len(config_ids)

    num_cols = 3
    num_rows_per_dataset = num_models// num_cols + int(num_models % num_cols > 0)
    num_rows = len(dataset_plot_info) * num_rows_per_dataset

    fig = plt.figure(figsize=(20, 5*num_rows))
    fig.patch.set_facecolor('white')
    grid = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=.4) # wspace=0.4, hspace=0.4)

    #fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
    #                         subplot_kw={'projection': projection})
    #axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    row_ind = 0
    col_ind = 0
    for i, plot_info in enumerate(dataset_plot_info):
        # increment_row_id = True
        dataset = plot_info['dataset']
        for model_ind in range(num_models):
            config_id = config_ids[model_ind]
            run_name = run_names[model_ind]
            title = titles[model_ind]
            standardize_scores_flag = standardize_scores[model_ind]
            # Add a subplot to the grid at (row_ind, col_ind)
            axes = fig.add_subplot(grid[row_ind, col_ind])
            results_count = show_plot_func(
                repo, var_repo, plotter, axes, config_id,
                run_name,
                dataset, dataset,
                plot_info['plots'],
                class_label_column="class_label",
                title=f"{dataset}/{title}",
                llr_config_id=llr_config_id,
                llr_run_name=llr_run_name)
            if results_count == 0:
                break
            col_ind += 1
            if col_ind >= num_cols:
                col_ind = 0
                row_ind += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def show_metric_by_domain_plots_multi(show_plot_func, dataset_plot_info):

    container = ALDEContainer("./config/msaldem.yaml")
    repo = container.query_repository
    plotter = container.plotter
    var_repo = container.variant_repository
    pdb_repo = container.pdb_repository

    config_ids = ["c3_1"]
    run_names = ["RF_AL_ALL_PRED"]
    titles = ["RandomForest AL"]
    standardize_scores = [False]
    num_models = len(config_ids)

    num_cols = 1
    num_rows_per_dataset = num_models// num_cols + int(num_models % num_cols > 0)
    num_rows = len(dataset_plot_info) * num_rows_per_dataset

    fig = plt.figure(figsize=(20, 5*num_rows))
    fig.patch.set_facecolor('white')
    grid = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=.8) # wspace=0.4, hspace=0.4)

    #fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
    #                         subplot_kw={'projection': projection})
    #axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    row_ind = 0
    col_ind = 0
    for i, plot_info in enumerate(dataset_plot_info):
        # increment_row_id = True
        dataset = plot_info['dataset']
        _, _, domains = pdb_repo.get_secondary_structure(dataset)
        for model_ind in range(num_models):
            config_id = config_ids[model_ind]
            run_name = run_names[model_ind]
            title = titles[model_ind]
            standardize_scores_flag = standardize_scores[model_ind]
            # Add a subplot to the grid at (row_ind, col_ind)
            axes = fig.add_subplot(grid[row_ind, col_ind])
            results_count = show_plot_func(
                repo, var_repo, plotter, axes, config_id,
                run_name,
                dataset, dataset,
                plot_info['plots'],
                class_label_column="class_label",
                title=f"{dataset}/{title}",
                domains=domains)
            if results_count == 0:
                break
            col_ind += 1
            if col_ind >= num_cols:
                col_ind = 0
                row_ind += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def show_metric_by_domain_plots(show_plot_func, datasets):

    container = ALDEContainer("./config/msaldem.yaml")
    repo = container.query_repository
    plotter = container.plotter
    var_repo = container.variant_repository
    pdb_repo = container.pdb_repository

    config_ids = ["c3_1"]
    run_names = ["RF_AL_ALL_PRED"]
    titles = ["RandomForest AL"]
    standardize_scores = [False]
    num_models = len(config_ids)

    num_cols = 1
    num_rows_per_dataset = num_models// num_cols + int(num_models % num_cols > 0)
    num_rows = len(datasets) * num_rows_per_dataset

    fig = plt.figure(figsize=(20, 5*num_rows))
    fig.patch.set_facecolor('white')
    grid = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=.8) # wspace=0.4, hspace=0.4)

    #fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6*len(datasets_)),
    #                         subplot_kw={'projection': projection})
    #axes = axes.flatten()
    plt.style.use('seaborn-v0_8')
    row_ind = 0
    col_ind = 0
    for i, dataset in enumerate(datasets):
        # increment_row_id = True
        _, _, domains = pdb_repo.get_secondary_structure(dataset)
        for model_ind in range(num_models):
            config_id = config_ids[model_ind]
            run_name = run_names[model_ind]
            title = titles[model_ind]
            standardize_scores_flag = standardize_scores[model_ind]
            # Add a subplot to the grid at (row_ind, col_ind)
            axes = fig.add_subplot(grid[row_ind, col_ind])
            results_count = show_plot_func(
                repo, plotter, axes, config_id,
                run_name,
                dataset,
                title=f"{dataset}/{title}",
                domains=domains)
            if results_count == 0:
                break
            col_ind += 1
            if col_ind >= num_cols:
                col_ind = 0
                row_ind += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()

