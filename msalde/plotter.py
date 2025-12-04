
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

    def plot_2d_landscape_by_position_aa(self, axes, position,
                                         y_value_lists: list,
                                         line_labels: list,
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

