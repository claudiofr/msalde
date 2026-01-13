import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from msalde.var_dbmodel import VariantAssay
from msalde.variant_util import hgvs_protein_to_variant_id


class VariantRefLoader:
    def __init__(self, config: dict):
        self._config = config

    def _get_dataset_paths(self, dataset_name: str, require_wt: bool = False) -> tuple[str, str]:
        dataset_config =  self._config.get(dataset_name, None)
        if dataset_config is None:
            raise ValueError(f"{dataset_name} dataset configuration is missing.")
        dataset_path = dataset_config.get("input_path", None)
        if dataset_path is None:
            raise ValueError(f"input_path is missing in {dataset_name} dataset configuration.")
        wt_path = dataset_config.get("wt_path", None)
        if wt_path is None and require_wt:
            raise ValueError(f"wt_path is missing in {dataset_name} dataset configuration.")
        return dataset_path, wt_path

    def load_src_dataset(self) -> pd.DataFrame:
        dataset_name = "src"
        assay_source = "SRC"
        dataset_path, wt_path = self._get_dataset_paths(dataset_name)
        # Add logic to load the dataset from the specified dataset_path
        variant_df = pd.read_excel(dataset_path, skiprows=2)

        # Process the DataFrame as needed
        variant_df = variant_df[~variant_df['Variant'].str.contains("=")]
        variant_df['variant_id'] = variant_df['Variant'].apply(
            hgvs_protein_to_variant_id)
        variant_df = variant_df[variant_df['variant_id'].notna()]
        
        variant_df = variant_df.assign(
            assay_source=assay_source,
            protein_symbol=assay_source,
            assay_score=variant_df["Activity Score"],
            class_label=np.where(
                variant_df["Classification"] == "gain of function", 1,
                np.where(
                    variant_df["Classification"] == "loss of function",
                    0, np.nan
                )
            )
        )
        variant_df = variant_df[variant_df['class_label'].notna()]
        variant_assay = VariantAssay(
            assay_source=assay_source,
            protein_symbol=assay_source,
            gene_symbol=assay_source,
            description="class_label=1 if class = gain of function, 0 if class = lof"
        )
        return variant_assay, variant_df
        return variant_df

    def load_pten_dataset(self) -> pd.DataFrame:
        dataset_name = "pten"
        dataset_path, wt_path = self._get_dataset_paths(dataset_name)
        # Add logic to load the dataset from the specified dataset_path
        variant_df = pd.read_csv(dataset_path)

        # Process the DataFrame as needed
        variant_df = variant_df[~variant_df['hgvs_pro'].str.contains("=")]
        variant_df['variant_id'] = variant_df['hgvs_pro'].apply(
            hgvs_protein_to_variant_id)
        variant_df.loc[variant_df["hgvs_pro"] == "_wt", "variant_id"] = "WT"
        variant_df = variant_df[variant_df['variant_id'].notna()]
        
        variant_df = variant_df.assign(
            assay_source="PTEN",
            protein_symbol="PTEN",
            assay_score=variant_df["score"],
            class_label=np.where(
                (variant_df["abundance_class"] == 4) &
                (variant_df["score"] > 1.1), 1,
                np.where(
                    variant_df["abundance_class"] == 1,
                    0, np.nan
                )
            )
        )
        variant_df = variant_df[variant_df['class_label'].notna()]
        variant_assay = VariantAssay(
            assay_source="PTEN",
            protein_symbol="PTEN",
            gene_symbol="PTEN",
            description="class_label=1 if abundance_class == 4 and assay_score > 1.1, 0 if abundance_class < 3"
        )
        return variant_assay, variant_df
        return variant_df

    def load_mc4r_dataset(self, pathway: str,
                          compound: str = None, dose: str = None) -> pd.DataFrame:
        dataset_config =  self._config.get("mc4r", None)
        if dataset_config is None:
            raise ValueError("mc4r dataset configuration is missing.")
        dataset_path = dataset_config.get("input_path", None)
        if dataset_path is None:
            raise ValueError("input_path is missing in mc4r dataset configuration.")
        wt_path = dataset_config.get("wt_path", None)
        if wt_path is None:
            raise ValueError("wt_path is missing in mc4r dataset configuration.")
        # Add logic to load the dataset from the specified dataset_path
        variant_df = pd.read_csv(dataset_path, sep="\t")
        variant_df = variant_df[(variant_df['aa'] != '*')
                            & (variant_df['pathway'] == pathway)
                            & ((variant_df['compound'].isna() & (compound is None))
                               | ((variant_df['compound'] == compound) &
                                  (variant_df['dose'] == dose)))]
        wt_df = pd.read_csv(wt_path, sep='\t')

        variant_df = variant_df.merge(
            wt_df,
            left_on='pos',
            right_on='Pos',
            suffixes=('', '_wt')
        )

        variant_df['variant_id'] = (
            variant_df['WT_AA_Short'] + variant_df['pos'].astype(str) +
            variant_df['aa']
        )
        variant_df.rename(columns={'statistic': 'z_score'}, inplace=True)

        wt_sequence = ''.join(wt_df.sort_values('Pos')['WT_AA_Short'].tolist())

        variant_df['sequence'] = variant_df.apply(
            lambda row: wt_sequence[:row['pos'] - 1] +
            row['aa'] +
            wt_sequence[row['pos']:],
            axis=1
        )

        wt_row = pd.DataFrame({"variant_id": ["WT"], "z_score": [0],
                            "sequence": [wt_sequence],
                            "compound": [compound],
                            "dose": [dose],
                            "pathway": [pathway],
                            "pos": [1],
                            "WT_AA_Short": [wt_sequence[0]],
                            "aa": [wt_sequence[0]]})
        
        variant_df = pd.concat([variant_df, wt_row], ignore_index=True)
        variant_df[["assay_score","alt_assay_score1"]] = variant_df[["z_score", "p.adj"]]
        variant_df['class_label'] = variant_df.apply(
            lambda row: row["assay_score"] > 0 if row["alt_assay_score1"] < 0.01 else None,
            axis=1
        )
        variant_df['alt_class_label1'] = variant_df.apply(
            lambda row: row["assay_score"] > 0 if row["alt_assay_score1"] < 0.05 else None,
            axis=1
        )
        variant_df = variant_df.assign(
            assay_source="MC4R",
            protein_symbol="MC4R",
            assay_type=variant_df["pathway"],
            assay_subtype=np.where(
                variant_df["compound"].isna(),
                variant_df["compound"],
                variant_df["compound"] + "/" + variant_df["dose"]
            ),
            position=variant_df["pos"],
            ref_aa=variant_df["WT_AA_Short"],
            var_aa=variant_df["aa"],
            alt_assay_score2=None,
            alt_class_label2=None,
        )
        variant_assay = VariantAssay(
            assay_source="MC4R",
            protein_symbol="MC4R",
            gene_symbol="MC4R",
            description="assay_score=z score, alt_assay_score1=q value, class_label/q value<.01, alt_class_label1/q value<.05"
        )
        return variant_assay, variant_df

    def load_adrb2_dataset_ntile(self, condition: float) -> pd.DataFrame:
        dataset_config =  self._config.get("adrb2", None)
        if dataset_config is None:
            raise ValueError("adrb2 dataset configuration is missing.")
        dataset_path = dataset_config.get("input_path", None)
        if dataset_path is None:
            raise ValueError("input_path is missing in adrb2 dataset configuration.")
        wt_path = dataset_config.get("wt_path", None)
        if wt_path is None:
            raise ValueError("wt_path is missing in adrb2 dataset configuration.")
        # Add logic to load the dataset from the specified dataset_path
        variant_df = pd.read_excel(dataset_path)
        variant_df = variant_df[variant_df["Condition"] == condition]

        TOP_PERCENTILE = 80
        BOTTOM_PERCENTILE = 20

        top_percentile_cutoff = np.percentile(
            variant_df["Norm"],
            TOP_PERCENTILE
        )
        bottom_percentile_cutoff = np.percentile(
            variant_df["Norm"],
            BOTTOM_PERCENTILE
        )

        variant_df["putative_gof"] = np.where(
            (variant_df["Norm"] >= top_percentile_cutoff),
            1,
            np.where(
                (variant_df["Norm"] <= bottom_percentile_cutoff),
                0,
                np.nan
            )
        )

        with open(wt_path, "r") as f:
            data = f.readlines()
        wt_sequence = data[1].strip()

        variant_df['variant_id'] = variant_df.apply(
            lambda row: wt_sequence[row['Pos'] - 2] + str(row['Pos']) + row['AA'],
            axis=1
        )

        variant_df = variant_df.assign(
            assay_source="ADRB2",
            protein_symbol="ADRB2",
            assay_type=str(condition),
            assay_subtype=str(TOP_PERCENTILE) + "_" + str(BOTTOM_PERCENTILE),
            position=variant_df["Pos"],
            ref_aa=variant_df["variant_id"].str[0],
            var_aa=variant_df["AA"],
            assay_score=variant_df["Norm"],
            alt_assay_score1=np.nan,
            alt_assay_score2=np.nan,
            class_label=variant_df["putative_gof"],
            alt_class_label1=np.nan,
            alt_class_label2=np.nan
        )
        variant_assay = VariantAssay(
            assay_source="ADRB2",
            protein_symbol="ADRB2",
            gene_symbol="ADRB2",
            description="class_label: gof=1,lof=0,vus=null, computed by top ntile gof," +
            "bottom ntile lof"
        )
        return variant_assay, variant_df

    def load_adrb2_dataset_q_value(self, condition: float) -> pd.DataFrame:
        dataset_config =  self._config.get("adrb2", None)
        if dataset_config is None:
            raise ValueError("adrb2 dataset configuration is missing.")
        dataset_path = dataset_config.get("input_path", None)
        if dataset_path is None:
            raise ValueError("input_path is missing in adrb2 dataset configuration.")
        wt_path = dataset_config.get("wt_path", None)
        if wt_path is None:
            raise ValueError("wt_path is missing in adrb2 dataset configuration.")
        # Add logic to load the dataset from the specified dataset_path
        variant_df = pd.read_excel(dataset_path)
        variant_df = variant_df[variant_df["Condition"] == condition]

        # Compute GOF and LOF based on z-scores. We approximate WT activity
        # using the center of the distribution of activities. The
        # center is characterized by the median and variability using MAD.
        # We define GOF and LOF using z-scores compared to this baseline.
        # GOF and LOF variants have activities that are significantly higher
        # or lower, respectively than the typical variant.

        baseline = variant_df["Norm"].median()
        mad = (variant_df["Norm"] - variant_df["Norm"].mean()).abs().mean()
        baseline_sigma = (
            mad * 1.4826   # convert MAD to SD
        )

        variant_df["z_score"] = (
            (variant_df["Norm"] - baseline) /
            np.sqrt(variant_df["Uncert"]**2 + baseline_sigma**2)
        )
        variant_df["p_value_lof"] = norm.cdf(variant_df["z_score"])
        variant_df["p_value_gof"] = 1 - variant_df["p_value_lof"]
        variant_df = variant_df[variant_df["p_value_lof"].notna()]

        variant_df["q_value_gof"] = multipletests(
            variant_df["p_value_gof"],
            method="fdr_bh"
        )[1]
        variant_df["q_value_lof"] = multipletests(
            variant_df["p_value_lof"],
            method="fdr_bh"
        )[1]
        variant_df["log2fc_baseline"] = np.log2(
            variant_df["Norm"] / baseline
        )
        LOG2FC_THRESHOLD = 0.3
        Q_THRESHOLD = 0.05
        MIN_BARCODES = 0 # 20

        # putative_gof: 1 = GOF, 0 = LOF, nan = neither
        variant_df["putative_gof"] = np.where(
            (variant_df["log2fc_baseline"] >= LOG2FC_THRESHOLD) &
            (variant_df["q_value_gof"] < Q_THRESHOLD) &
            (variant_df["N"] >= MIN_BARCODES),
            1,
            np.where(
                (variant_df["log2fc_baseline"] <= -LOG2FC_THRESHOLD) &
                (variant_df["q_value_lof"] < Q_THRESHOLD) &
                (variant_df["N"] >= MIN_BARCODES),
                0,
                np.nan
            )
        )

        with open(wt_path, "r") as f:
            data = f.readlines()
        wt_sequence = data[1].strip()

        variant_df['variant_id'] = variant_df.apply(
            lambda row: wt_sequence[row['Pos'] - 2] + str(row['Pos']) + row['AA'],
            axis=1
        )

        variant_df = variant_df.assign(
            assay_source="ADRB2",
            protein_symbol="ADRB2",
            assay_type=str(condition),
            assay_subtype=str(Q_THRESHOLD),
            position=variant_df["Pos"],
            ref_aa=variant_df["variant_id"].str[0],
            var_aa=variant_df["AA"],
            assay_score=variant_df["Norm"],
            alt_assay_score1=np.nan,
            alt_assay_score2=np.nan,
            class_label=variant_df["putative_gof"],
            alt_class_label1=np.nan,
            alt_class_label2=np.nan
        )
        variant_assay = VariantAssay(
            assay_source="ADRB2",
            protein_symbol="ADRB2",
            gene_symbol="ADRB2",
            description="class_label: gof=1,lof=0,vus=null, computed by computing z value," +
            "q value, fc, with fc >= .3 or fc <=.3; assay_subtype= q threshold"
        )
        return variant_assay, variant_df
    


