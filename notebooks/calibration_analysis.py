import context  # noqa: F401
import pandas as pd

def get_clinvar_labels_for_dataset(repo, external_repo, dataset_name):
    gene_symbol = repo.get_gene_symbol_for_dataset(dataset_name)
    clinvar_labels = external_repo.get_clinvar_labels_by_gene(gene_symbol)
    return clinvar_labels


def get_dataset_clinvar_results(repo, external_repo, config_id, dataset,
                                run_name) -> pd.DataFrame:
    results = repo.get_last_round_scores_by_config_dataset_run(
        config_id=config_id, dataset_name=dataset, run_name=run_name)
    if len(results) == 0:
        return results, 0, 0
    clinvar_labels = get_clinvar_labels_for_dataset(repo, external_repo, dataset)
    clinvar_results = results.merge(clinvar_labels, on="variant_id",
                                    how="inner")
    if len(clinvar_results) == 0:
        return clinvar_results, 0, 0
    clinvar_results['prediction_score'] = clinvar_results['prediction_score'] * -1
    num_positive = (clinvar_results['label'] == 1).sum()
    num_negative = (clinvar_results['label'] == 0).sum()
    return clinvar_results, num_positive, num_negative

