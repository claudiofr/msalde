from itertools import chain
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as plt
from matplotlib.pyplot import axes
from sqlalchemy import label


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

    def plot_2d_landscape_by_position_aa(self, axes_top, axes_middle,
                                         axes_bottom, position,
                                         assay_scores, prediction_scores,
                                         prediction_label,
                                         counts,
                                         count_label: str,
                                         ss_track: list, residue_nums: list,
                                         domains: list,
                                         title):

        line_styles = ['-', '--', '-.', ':']
        min_position = min(position)
        max_position = max(position)
        axes_top.set_xlim(min_position, max_position)
        axes_middle.set_xlim(min_position, max_position)
        axes_bottom.set_xlim(min_position, max_position)

        color_map = {"H": "red", "E": "gold", "C": "lightgray", "?": "white"}
        colors = [color_map[c] for c in ss_track]

        # Draw secondary structure bars first (so they are below the bands)
        axes_top.bar(residue_nums, [0.2]*len(residue_nums), color=colors, width=1.0, bottom=0.1, zorder=1)

        # Draw domain bands with increased height and place labels inside the bands
        band_height = 0.7  # Height of the domain bands
        for domain in domains:
            start = domain["start"]
            end = domain["end"]
            name = domain["name"]
            color = domain["color"]
            if end < min_position or start > max_position:
                continue
            if start < min_position:
                start = min_position
            if end > max_position:
                end = max_position
            axes_top.axvspan(start, end, ymin=0.5, ymax=0.75, color=color, alpha=0.3, zorder=2)
            axes_top.text((start + end) / 2, band_height, name, ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)

        axes_top.set_ylim(0, 1)
        axes_top.set_yticks([])
        # axes_top.set_xlabel("Residue number")
        axes_top.set_title(title)

        # Create legend for secondary structure colors
        legend_patches = [
            mpatches.Patch(color=color_map["H"], label="Helix"),
            mpatches.Patch(color=color_map["E"], label="Strand"),
            mpatches.Patch(color=color_map["C"], label="Coil"),
        ]
        axes_top.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.1, 1.0)) #, title="Secondary Structure")

        colors = [(0.118, 0.565, 1.000, 0.7), (0.235, 0.702, 0.443, 0.7), 'orange', 'purple', 'cyan', 'magenta']
        axes_middle.plot(position, assay_scores, linestyle='-', color=colors[0], label='Assay Score')
        axes_middle.set_ylabel('Mean Score')
        axes_middle.plot(position, prediction_scores, linestyle='--', color=colors[1],
                      label=prediction_label)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores > prediction_scores), color='lightcoral', alpha=0.5)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores < prediction_scores), color='yellow', alpha=0.5)
        axes_middle.legend(loc='lower right', bbox_to_anchor=(1.1, 0.8))
        axes_bottom.plot(position, counts, linestyle='-', color=colors[2],
                         label=count_label)
        axes_bottom.set_ylabel(count_label)
        axes_bottom.set_xlabel('Residue Position')
        axes_middle.patch.set_facecolor('white')
        axes_bottom.patch.set_facecolor('white')
        for spine in chain(axes_middle.spines.values(), axes_bottom.spines.values()):
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

    def plot_roc_auc_by_round(self, axes, results, llr_auc,
                       title,
                       num_positive, num_negative):
    
        rounds = results["round_num"].astype(int)
        axes.plot(results["round_num"], results["auc"],
                  marker='o', linestyle='-')
        """
        axes.errorbar(results["round_num"], results["auc_mean"],
                        yerr=results["auc_std"], fmt='-o', capsize=5)
        """
        axes.set_title(f'{title} (+{num_positive} / -{num_negative})', fontsize=16)
        axes.set_ylabel('AUC', fontsize=14)
        axes.set_xlabel('Round', fontsize=14)
        axes.set_xticks(rounds)
        # Add horizontal dashed line for LLR
        axes.axhline(llr_auc, color='black', linestyle='--', label='Log Likelihood Ratio')
        axes.legend(fontsize=14, loc='upper left', framealpha=0.0)
        # plt.colorbar(scatter, ax=axes, label="Label")

    def plot_roc_auc_by_round_multi(self, axes, results_list,
                       title):

        colors = plt.cm.tab10.colors   # 10 distinct colors
        if len(results_list) > len(colors):
            raise ValueError("Too many result sets to plot; increase color palette.")

        for i, results in enumerate(results_list):
            aucs = results["auc_results"]
            label = results["label"]
            llr_auc = results["llr_auc"]
            num_positive = results["num_positive"]
            num_negative = results["num_negative"]
            rounds = aucs["round_num"].astype(int)
            max_rounds = rounds.max()
            axes.set_xlim(right=max_rounds+1)
            axes.plot(rounds, aucs["auc"],
                      marker='o', linestyle='-',
                      label=f'{label} (+{num_positive}/-{num_negative})',
                      color=colors[i]
            )
            # Add horizontal dashed line for LLR

            # Short horizontal dashed line at y = 5, spanning x = 2 to x = 6
            axes.hlines(
                y=llr_auc,
                xmin=max_rounds + 0.2,
                xmax=max_rounds + 1.0,
                colors=colors[i],
                linestyles='dashed',
                linewidth=2
            )
            """
            axes.errorbar(results["round_num"], results["auc_mean"],
                            yerr=results["auc_std"], fmt='-o', capsize=5)
            """
        axes.set_title(f'{title}', fontsize=16)
        axes.set_ylabel('AUC', fontsize=12)
        axes.set_xlabel('Round', fontsize=12)
        axes.set_xticks(rounds)

        # Add a right-side label manually
        axes.text(
            1.02, 0.5, "LLR AUC",          # x, y in axes coordinates
            transform=axes.transAxes,
            rotation=90,
            va='center',
            ha='left'
        )

        #axes.axhline(llr_auc, color='black', linestyle='--', label='Log Likelihood Ratio')
        axes.legend(fontsize=11, loc='upper left', framealpha=0.0)
        # plt.colorbar(scatter, ax=axes, label="Label")


    def plot_metric_by_domain_multi(self, axes, results_list,
                                    metric_name: str,
                       title):

        #plt.rcParams.update({
        #    "text.usetex": True,
        #})
        colors = plt.cm.tab10.colors   # 10 distinct colors
        if len(results_list) > len(colors):
            raise ValueError("Too many result sets to plot; increase color palette.")

        # We will build custom x-tick labels with domain names and +/- counts
        # in xtick_label_info_list. Each element in the list corresponds to a 
        # tuple with the first element being the domain name and the second element
        # being a list of (num_positive, num_negative) tuples for each result set in
        # results_list. So the structure will be:
        # [ (domain_name, [ (num_pos_1, num_neg_1), (num_pos_2, num_neg_2), ... ] ), ... ]
        # So for each domain along x axis, we will have the name and a 
        # list of counts for each plot represented in results_list.

        xtick_label_info_list = None
        for i, result_dict in enumerate(results_list):
            results = result_dict["results"]
            label = result_dict["label"]
            #num_positive = results["num_positive"]
            #num_negative = results["num_negative"]
            if not xtick_label_info_list:
                domain_names = results["domain"].apply(lambda d: d["name"])
                xtick_label_info_list = [(domain_name, []) for domain_name in domain_names]
            # Update the counts for each domain
            for domain_ind, result in results.iterrows():
                xtick_label_info_list[domain_ind][1].append((result["num_positive"], result["num_negative"]))
            axes.plot(domain_names,
                      results["metric"],
                      marker='o', linestyle='-',
                      label=f'{label}',
                      color=colors[i]
            )

        axes.set_title(f'{title}', fontsize=16)
        axes.set_ylabel(metric_name, fontsize=12)
        # axes.set_xlabel('Domain', fontsize=12)

        # For each domain, set custom x-tick labels with domain name and +/- counts
        # for each plot in results_list. We will place the domain name at y=-0.1
        # and the counts below that, with some vertical spacing.
        axes.set_xticks(range(len(results)))
        axes.set_xticklabels([])  # Clear default labels
        positions = axes.get_xticks() 
        for xpos, label_info in zip(positions, xtick_label_info_list):
            axes.text(xpos, -0.1, label_info[0], ha="center", va="top", transform=axes.get_xaxis_transform() )
            for i, (variant_counts, color) in enumerate(zip(label_info[1], colors[:len(results_list)])):
                text = f"+{variant_counts[0]}/-{variant_counts[1]}"
                axes.text(xpos, -0.2 - 0.08*i, text, ha="center", va="top", color=color, transform=axes.get_xaxis_transform() )

        axes.legend(fontsize=11, loc='upper left', framealpha=0.0,
                    bbox_to_anchor=(1.02, 1))
        # plt.colorbar(scatter, ax=axes, label="Label")


    def plot_metric_by_domain(self, axes, results,
                                    metric_name: str,
                       title):

        axes.plot(results["domain"].apply(lambda d: d["name"]),
                          results["metric"],
                          marker='o', linestyle='-',
            )

        axes.set_title(f'{title}', fontsize=16)
        axes.set_ylabel(metric_name, fontsize=12)
        axes.set_xlabel('Domain', fontsize=12)







