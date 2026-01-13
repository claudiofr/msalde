from itertools import chain
import random
import matplotlib.colors as mcolors


class ALDEPlotter:
    """Plot results of an analysis"""

    def __init__(self, config):
        self._config = config

    def plot_roc_curve(self, axes, fpr, tpr, auc,
                       optimal_youden_fpr, title,
                       num_positive, num_negative):
        label = f'AUC={str(round(auc, 2))}'
        axes.plot(fpr, tpr,
                  label=label)
        axes.axvline(optimal_youden_fpr,
                     color='red', linestyle='--', label='Youden Index')
        axes.set_xlabel('False Positive Rate', fontsize=14)
        axes.set_ylabel('True Positive Rate', fontsize=14)
        # plt.tick_params(axis='both', labelsize=14)
        axes.set_title(f'{title} (+ {str(num_positive)}/- {str(num_negative)})',
                       fontsize=14)
        axes.legend(fontsize=14)

    def plot_pr_curve(self, axes, recall, precision, auc, title,
                      num_positive, num_negative):
        label = f'AUC={str(round(auc, 2))}'
        axes.plot(recall, precision,
                  label=label)
        axes.set_xlabel('Recall', fontsize=14)
        axes.set_ylabel('Precision', fontsize=14)
        # plt.tick_params(axis='both', labelsize=14)
        axes.set_title(f'{title} (+{str(num_positive)} / -{str(num_negative)})',
                       fontsize=14)
        axes.legend(fontsize=14)

    def plot_2d_landscape_by_position_aa(self, axes_top, axes_bottom, position,
                                         assay_scores, prediction_scores,
                                         prediction_label,
                                         counts,
                                         count_label: str,
                                         title):

        line_styles = ['-', '--', '-.', ':']
        colors = [(0.118, 0.565, 1.000, 0.7), (0.235, 0.702, 0.443, 0.7), 'orange', 'purple', 'cyan', 'magenta']
        axes_top.plot(position, assay_scores, linestyle='-', color=colors[0], label='Assay Score')
        axes_top.set_ylabel('Mean Score')
        axes_top.plot(position, prediction_scores, linestyle='--', color=colors[1],
                      label=prediction_label)
        axes_top.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores > prediction_scores), color='lightcoral', alpha=0.5)
        axes_top.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores < prediction_scores), color='yellow', alpha=0.5)
        axes_top.legend(loc='lower right', bbox_to_anchor=(1.1, 1.2))
        axes_bottom.plot(position, counts, linestyle='-', color=colors[2],
                         label=count_label)
        axes_bottom.set_ylabel(count_label)
        axes_bottom.set_xlabel('Residue Position')
        axes_top.set_title(f"{title}")
        axes_top.patch.set_facecolor('white')
        axes_bottom.patch.set_facecolor('white')
        for spine in chain(axes_top.spines.values(), axes_bottom.spines.values()):
            spine.set_visible(True)        # make sure they are visible
            spine.set_color('black')       # set border color
            spine.set_linewidth(1.0)       # set thickness

    def plot_2d_landscape_by_position_aa_old1(self, axes_top, axes_bottom, position,
                                         assay_scores, prediction_scores,
                                         prediction_label,
                                         counts,
                                         count_label: str,
                                         title):

        line_styles = ['-', '--', '-.', ':']
        colors = [(0.118, 0.565, 1.000, 0.7), (0.235, 0.702, 0.443, 0.7), 'orange', 'purple', 'cyan', 'magenta']
        axes_top.plot(position, assay_scores, linestyle='-', color=colors[0], label='Assay Score')
        axes_top.set_ylabel('Assay Score')
        axes_top2 = axes_top.twinx()
        axes_top2.plot(position, prediction_scores, linestyle='--', color=colors[1],
                       label=prediction_label)
        axes_top2.set_ylabel(prediction_label)
        axex_top.fill_between(position, assay_scores, prediction_scores, color='lightblue', alpha=0.5)
        axes_top.legend(loc='lower right', bbox_to_anchor=(1.1, 1.2))
        axes_top2.legend(loc='lower right', bbox_to_anchor=(1.1, 1.1))
        axes_bottom.plot(position, counts, linestyle='-', color=colors[2],
                         label=count_label)
        axes_bottom.set_ylabel(count_label)
        axes_bottom.set_xlabel('Sequence Space (Position AA Bin)')
        axes_top.set_title(f"{title}")

    def plot_2d_landscape_by_position_aa_old(self, axes, position,
                                         y_value_lists, line_labels,
                                         counts,
                                         count_label: str,
                                         title):
        line_styles = ['-', '--', '-.', ':']
        colors = [(0.118, 0.565, 1.000, 0.7), (0.235, 0.702, 0.443, 0.7), 'orange', 'purple', 'cyan', 'magenta']
        if len(y_value_lists) == 0:
            axes.plot(position, y_value_lists[0], marker='o', linestyle='-')
            axes.set_ylabel(line_labels[0])
        else:
            for i, y_values_label in enumerate(zip(y_value_lists, line_labels)):
                axes.plot(position, y_values_label[0], marker='o', linestyle=line_styles[i],
                          label=y_values_label[1], color=colors[i])
            axes.set_ylabel("Measure Value")
        # counts on secondary y-axis
        if counts is not None:
            ax2 = axes.twinx()
            ax2.plot(position, counts, marker='x', linestyle=':', color='lightgrey',
                     label=count_label)
            ax2.set_ylabel(count_label)
            axes.legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            axes.legend()
        # Add labels
        axes.set_xlabel('Sequence Space (Position Bin)')
        axes.set_title(f"{title}")


    def plot_3d_protein_landscape_by_position_aa(
            self, axes, position, aa_index, squared_error, title):

        axes.scatter(position, aa_index, squared_error, alpha=0.7)
        axes.set_xlabel('Amino Acid Position')
        axes.set_ylabel('Amino Acid')
        axes.set_zlabel('Squared Error')
        axes.set_title(f"{title}")


    def plot_mean_activity_by_round(self, axes, results_df_list, labels, llr_top_mean_activity, title):

        # Get all named colors from Matplotlib
        all_colors = list(mcolors.CSS4_COLORS.keys())

        # Pick 10 random ones
        random.seed(42)
        colors = random.sample(all_colors, 7)
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta', 'brown']
        for i, results_df in enumerate(results_df_list):

            rounds = results_df["round_num"].astype(int)
            axes.errorbar(rounds, results_df["mean_score"],
                        yerr=results_df["stddev"], fmt='-o', capsize=5, label=labels[i],
                        color=colors[i])
        axes.set_title(f'{title}', fontsize=16)
        axes.set_ylabel('Mean Activity', fontsize=14)
        axes.set_xlabel('Round', fontsize=14)
        axes.set_xticks(rounds)
        # Add horizontal dashed line for LLR
        axes.axhline(llr_top_mean_activity, color='black', linestyle='--', label='Log Likelihood Ratio')
        axes.legend(fontsize=14, loc='upper left', framealpha=0.0)
        # plt.colorbar(scatter, ax=axes, label="Label")


