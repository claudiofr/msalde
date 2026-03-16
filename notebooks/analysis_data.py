import context  # noqa: F401
from notebooks.analysis_plotter import (
    compute_gof_lof_by_activity_threshold,
    get_compute_gof_lof_by_quantile_func
)

compute_class_label_by_activity_threshold_func = \
    lambda results, class_label_name: compute_gof_lof_by_activity_threshold(
        results, class_label_name, gof_threshold=1.18, lof_threshold=0.66
    )

compute_gof_top_10_pct_func = \
    get_compute_gof_lof_by_quantile_func(lof_quantile=0.9, gof_quantile=0.9)
compute_gof_top_20_pct_func = \
    get_compute_gof_lof_by_quantile_func(lof_quantile=0.8, gof_quantile=0.8)
compute_lof_top_10_pct_func = \
    get_compute_gof_lof_by_quantile_func(lof_quantile=0.1, gof_quantile=0.1)
compute_lof_top_20_pct_func = \
    get_compute_gof_lof_by_quantile_func(lof_quantile=0.2, gof_quantile=0.2)

additional_plots_info = [
    {
        "assay_type": None,
        "compute_class_label_func": compute_gof_top_10_pct_func,
        "label": "GOF_10%",
        "config_id": "c3_1",
        "run_name": "RF_AL_ALL_PRED",
        "standardize_scores": False
    },
    {
        "assay_type": None,
        "compute_class_label_func": compute_gof_top_20_pct_func,
        "label": "GOF_20%",
        "config_id": "c3_1",
        "run_name": "RF_AL_ALL_PRED",
        "standardize_scores": False
    },
    {
        "assay_type": None,
        "compute_class_label_func": compute_lof_top_10_pct_func,
        "label": "LOF_10%",
        "config_id": "c3_1",
        "run_name": "RF_AL_ALL_PRED",
        "standardize_scores": False
    },
    {
        "assay_type": None,
        "compute_class_label_func": compute_lof_top_20_pct_func,
        "label": "LOF_20%",
        "config_id": "c3_1",
        "run_name": "RF_AL_ALL_PRED",
        "standardize_scores": False
    }
]


gof_dataset_al_llr_plot_info = [
    {
        "dataset": "MC4R",
        "plots": [
            {
                "assay_type": "Gs",
                "compute_class_label_func": None,
                "label": "RF AL",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
            {
                "assay_type": "Gs",
                "compute_class_label_func": None,
                "label": "LLR",
                "config_id": "c10",
                "run_name": "ESM2_LLR_ALL_PRED",
                "standardize_scores": True
            },
        ]
    },
    {
        "dataset": "HXK4",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": compute_class_label_by_activity_threshold_func,
                "label": "RF AL",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
            {
                "assay_type": None,
                "compute_class_label_func": compute_class_label_by_activity_threshold_func,
                "label": "LLR",
                "config_id": "c10",
                "run_name": "ESM2_LLR_ALL_PRED",
                "standardize_scores": True
            },
        ]
    },
    {
        "dataset": "PTEN",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "RF AL",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "LLR",
                "config_id": "c10",
                "run_name": "ESM2_LLR_ALL_PRED",
                "standardize_scores": True
            }
        ]
    },
    {
        "dataset": "SRC",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "RF AL",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "LLR",
                "config_id": "c10",
                "run_name": "ESM2_LLR_ALL_PRED",
                "standardize_scores": True
            }
        ]
    }
]

gof_dataset_plot_info = [
    {
        "dataset": "MC4R",
        "plots": [
            {
                "assay_type": "Gs",
                "compute_class_label_func": None,
                "label": "GOF/LOF",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
        ]
    },
    {
        "dataset": "HXK4",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": compute_class_label_by_activity_threshold_func,
                "label": "GOF/LOF",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            },
        ]
    },
    {
        "dataset": "PTEN",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "GOF/LOF",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            }
        ]
    },
    {
        "dataset": "SRC",
        "plots": [
            {
                "assay_type": None,
                "compute_class_label_func": None,
                "label": "GOF/LOF",
                "config_id": "c3_1",
                "run_name": "RF_AL_ALL_PRED",
                "standardize_scores": False
            }
        ]
    }
]

for dpi in gof_dataset_plot_info:
    dpi["plots"].extend(additional_plots_info)

gof_datasets = [dpi["dataset"] for dpi in gof_dataset_plot_info]