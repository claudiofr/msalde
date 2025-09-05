import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    """Plot results using data fetched from the repository."""

    def __init__(self, config, query_repository):
        self._config = config
        self._query_repo = query_repository

    def plot_results(self, simulation_id: int, metrics: list[str] = ["roc", "pr"]):
        """Main entry point to plot results for a given simulation."""

        if "roc" in metrics:
            df_roc = self._query_repo.get_roc_metrics(simulation_id)
            if df_roc.empty:
                raise ValueError(f"No ROC results found for simulation_id={simulation_id}")
            self._plot_roc(df_roc)

        if "pr" in metrics:
            df_pr = self._query_repo.get_pr_metrics(simulation_id)
            if df_pr.empty:
                raise ValueError(f"No PR results found for simulation_id={simulation_id}")
            self._plot_pr(df_pr)

    def _plot_roc(self, df: pd.DataFrame):
        """Example ROC plotting method."""
        plt.figure(figsize=(8, 6))
        plt.plot(df["fpr"], df["tpr"], label="ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(self._config.roc_title)
        plt.legend()
        plt.show()

    def _plot_pr(self, df: pd.DataFrame):
        """Example Precision-Recall plotting method."""
        plt.figure(figsize=(8, 6))
        plt.plot(df["recall"], df["precision"], label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(self._config.pr_title)
        plt.legend()
        plt.show()
