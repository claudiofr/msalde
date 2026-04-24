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
        # Colorblind-safe palette (Wong 2011, Nature Methods)
        BLUE       = "#0072B2"
        VERMILLION = "#D55E00"
        GREY       = "#999999"

        LW_DATA   = 1.0
        LW_SPINE  = 0.5
        FS_TITLE  = 7.5
        FS_LABEL  = 7
        FS_TICK   = 6
        FS_LEGEND = 6

        # ROC curve
        axes.plot(fpr, tpr, color=BLUE, linewidth=LW_DATA,
                  label=f"AUC = {auc:.2f}")
        # Diagonal reference (random classifier)
        axes.plot([0, 1], [0, 1], color=GREY, linewidth=LW_DATA * 0.8,
                  linestyle="--", zorder=0)
        # Optimal Youden index
        axes.axvline(optimal_youden_fpr, color=VERMILLION,
                     linestyle=":", linewidth=LW_DATA * 0.8,
                     label="Youden index")

        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel("False positive rate", fontsize=FS_LABEL, labelpad=3)
        axes.set_ylabel("True positive rate", fontsize=FS_LABEL, labelpad=3)
        axes.set_title(
            f"{title}\n(+{num_positive} / \u2212{num_negative})",
            fontsize=FS_TITLE, pad=6)
        axes.tick_params(axis="both", which="major",
                         labelsize=FS_TICK, width=LW_SPINE,
                         length=2, direction="out")
        axes.legend(fontsize=FS_LEGEND, frameon=False,
                    handlelength=1.2, handletextpad=0.4)
        axes.patch.set_facecolor("white")
        for spine in axes.spines.values():
            spine.set_linewidth(LW_SPINE)
            spine.set_color("black")
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)

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
                                         gof_assay_threshold,
                                         lof_assay_threshold,
                                         title):
        # Colorblind-safe palette (Wong 2011, Nature Methods)
        BLUE       = "#0072B2"
        VERMILLION = "#D55E00"
        GREY       = "#999999"
        # Secondary-structure colours (muted, print-safe)
        SS_COLORS  = {"H": "#CC79A7", "E": "#F0E442", "C": "#DDDDDD", "?": "white"}

        LW_DATA    = 1.0   # data lines
        LW_THRESH  = 0.6   # threshold reference lines
        LW_SPINE   = 0.5   # axis borders
        FS_TITLE   = 8
        FS_LABEL   = 7
        FS_TICK    = 6
        FS_LEGEND  = 6

        min_position = min(position)
        max_position = max(position)
        active_axes = [ax for ax in (axes_top, axes_middle, axes_bottom) if ax is not None]
        for ax in active_axes:
            ax.set_xlim(min_position, max_position)
            ax.patch.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_linewidth(LW_SPINE)
                spine.set_color("black")
            ax.tick_params(axis="both", which="major",
                           labelsize=FS_TICK, width=LW_SPINE, length=2,
                           direction="out")

        # --- Top panel: secondary structure + domain annotations ---
        ss_bar_colors = [SS_COLORS[c] for c in ss_track]
        axes_top.bar(residue_nums, [0.3] * len(residue_nums),
                     color=ss_bar_colors, width=1.0, bottom=0.05,
                     linewidth=0, zorder=1)
        for domain in domains:
            start = max(domain["start"], min_position)
            end   = min(domain["end"],   max_position)
            if start >= end:
                continue
            axes_top.axvspan(start, end, ymin=0.55, ymax=0.80,
                             color=domain["color"], alpha=0.25, zorder=2,
                             linewidth=0)
            axes_top.text((start + end) / 2, 0.88, domain["name"],
                          ha="center", va="center",
                          fontsize=FS_LEGEND, fontweight="bold", zorder=3)
        axes_top.set_ylim(0, 1)
        axes_top.set_yticks([])
        axes_top.set_xticks([])
        axes_top.set_title(title, fontsize=FS_TITLE, pad=8)
        for side in ("top", "right", "bottom", "left"):
            axes_top.spines[side].set_visible(False)

        # --- Middle panel: assay vs prediction scores ---
        axes_middle.plot(position, assay_scores,
                         linestyle="-", linewidth=LW_DATA,
                         color=BLUE, label="Assay score")
        axes_middle.plot(position, prediction_scores,
                         linestyle="--", linewidth=LW_DATA,
                         color=VERMILLION, label=prediction_label)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                                 where=(assay_scores > prediction_scores),
                                 color=BLUE, alpha=0.12, linewidth=0)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                                 where=(assay_scores < prediction_scores),
                                 color=VERMILLION, alpha=0.12, linewidth=0)
        axes_middle.axhline(y=gof_assay_threshold, color=GREY,
                            linestyle=":", linewidth=LW_THRESH, zorder=0)
        axes_middle.axhline(y=lof_assay_threshold, color=GREY,
                            linestyle=":", linewidth=LW_THRESH, zorder=0)
        axes_middle.set_ylabel("Mean score", fontsize=FS_LABEL, labelpad=3)
        axes_middle.set_xlabel("Residue position", fontsize=FS_LABEL, labelpad=3)
        axes_middle.yaxis.set_major_formatter(
            plt.ticker.FormatStrFormatter("%.1f"))
        for side in ("top", "right"):
            axes_middle.spines[side].set_visible(False)

    def plot_2d_landscape_by_position_aa_old1(self, axes_top, axes_middle,
                                         axes_bottom, position,
                                         assay_scores, prediction_scores,
                                         prediction_label,
                                         counts,
                                         count_label: str,
                                         ss_track: list, residue_nums: list,
                                         domains: list,
                                         gof_assay_threshold,
                                         lof_assay_threshold,
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
        axes_top.set_title(title, fontsize=16)

        # Create legend for secondary structure colors
        legend_patches = [
            mpatches.Patch(color=color_map["H"], label="Helix"),
            mpatches.Patch(color=color_map["E"], label="Strand"),
            mpatches.Patch(color=color_map["C"], label="Coil"),
        ]
        axes_top.legend(handles=legend_patches, loc="upper right",
                        bbox_to_anchor=(1.1, 1.0), fontsize=14) #, title="Secondary Structure")

        colors = [(0.118, 0.565, 1.000, 0.7), (0.235, 0.702, 0.443, 0.7), 'orange', 'purple', 'cyan', 'magenta']
        axes_middle.plot(position, assay_scores, linestyle='-', color=colors[0], label='Assay Score')
        axes_middle.set_ylabel('Mean Score', fontsize=12)
        axes_middle.plot(position, prediction_scores, linestyle='--', color=colors[1],
                      label=prediction_label)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores > prediction_scores), color='lightcoral', alpha=0.5)
        axes_middle.fill_between(position, assay_scores, prediction_scores,
                              where=(assay_scores < prediction_scores), color='yellow', alpha=0.5)
        axes_middle.axhline(y=gof_assay_threshold, color='lightgray',
                            linestyle='dashed', linewidth=1.5)
        axes_middle.axhline(y=lof_assay_threshold, color='lightgray',
                            linestyle='dashed', linewidth=1.5)
        axes_middle.legend(loc='lower right', bbox_to_anchor=(1.1, 0.8),
                           fontsize=14)
        axes_bottom.plot(position, counts, linestyle='-', color=colors[2],
                         label=count_label)
        axes_bottom.set_ylabel(count_label, fontsize=12)
        axes_bottom.set_xlabel('Residue Position', fontsize=12)
        axes_middle.patch.set_facecolor('white')
        axes_bottom.patch.set_facecolor('white')
        for spine in chain(axes_middle.spines.values(), axes_bottom.spines.values()):
            spine.set_visible(True)        # make sure they are visible
            spine.set_color('black')       # set border color
            spine.set_linewidth(1.0)       # set thickness

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

        # Colorblind-safe palette (Wong 2011, Nature Methods)
        COLORS = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # green
            "#CC79A7",  # pink
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#F0E442",  # yellow
            "#000000",  # black
        ]
        LW_DATA   = 1.0
        LW_SPINE  = 0.5
        FS_TITLE  = 7.5
        FS_LABEL  = 7
        FS_TICK   = 6
        FS_LEGEND = 6
        MS        = 3

        if len(results_list) > len(COLORS):
            raise ValueError("Too many result sets to plot; increase color palette.")

        for i, results in enumerate(results_list):
            aucs = results["auc_results"]
            label = results["label"]
            llr_auc = results["llr_auc"]
            num_positive = results["num_positive"]
            num_negative = results["num_negative"]
            rounds = aucs["round_num"].astype(int)
            max_rounds = rounds.max()
            axes.set_xlim(right=max_rounds + 1)
            axes.errorbar(rounds, aucs["auc"],
                          yerr=aucs["auc_std"],
                          marker='o', markersize=MS, linestyle='-',
                          linewidth=LW_DATA, capsize=2, capthick=LW_DATA * 0.8,
                          elinewidth=LW_DATA * 0.8,
                          label=f'{label} (+{num_positive}/\u2212{num_negative})',
                          color=COLORS[i])
            axes.hlines(
                y=llr_auc,
                xmin=max_rounds + 0.2,
                xmax=max_rounds + 1.0,
                colors=COLORS[i],
                linestyles='dashed',
                linewidth=LW_DATA)

        axes.axhline(0.5, color="#999999", linewidth=LW_SPINE, linestyle="--", zorder=0)
        axes.set_title(title, fontsize=FS_TITLE, pad=6)
        axes.set_ylabel('AUC', fontsize=FS_LABEL, labelpad=3)
        axes.set_xlabel('Round', fontsize=FS_LABEL, labelpad=3)
        axes.set_xticks(rounds)
        axes.tick_params(axis="both", which="major",
                         labelsize=FS_TICK, width=LW_SPINE,
                         length=2, direction="out")
        axes.text(
            1.02, 0.5, "LLR AUC",
            transform=axes.transAxes,
            fontsize=FS_TICK, rotation=90,
            va='center', ha='left')
        axes.legend(fontsize=FS_LEGEND, frameon=False,
                    handlelength=1.2, handletextpad=0.4,
                    loc='upper left')
        axes.patch.set_facecolor("white")
        for spine in axes.spines.values():
            spine.set_linewidth(LW_SPINE)
            spine.set_color("black")
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)


    def plot_metric_by_domain_multi(self, axes, results_list,
                                    metric_name: str,
                       title):

        # Colorblind-safe palette (Wong 2011, Nature Methods)
        COLORS = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # green
            "#CC79A7",  # pink
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#F0E442",  # yellow
            "#000000",  # black
        ]
        LW_DATA   = 1.0
        LW_SPINE  = 0.5
        FS_TITLE  = 7.5
        FS_LABEL  = 7
        FS_TICK   = 6
        FS_LEGEND = 6
        MS        = 3    # marker size

        if len(results_list) > len(COLORS):
            raise ValueError("Too many result sets to plot; increase color palette.")

        xtick_label_info_list = None
        display_domain_counts = True
        for i, result_dict in enumerate(results_list):
            results = result_dict["results"]
            label = result_dict["label"]
            if not xtick_label_info_list:
                domain_names = results["domain"]
                xtick_label_info_list = [(domain_name, []) for domain_name in domain_names]
            for domain_ind, result in results.iterrows():
                if "num_positive" not in result:
                    display_domain_counts = False
                    break
                xtick_label_info_list[domain_ind][1].append(
                    (result["num_positive"], result["num_negative"]))
            axes.errorbar(domain_names,
                          results["metric"],
                          yerr=results["metric_std"],
                          marker='o', markersize=MS, linestyle='-',
                          linewidth=LW_DATA, capsize=2, capthick=LW_DATA * 0.8,
                          elinewidth=LW_DATA * 0.8,
                          label=label,
                          color=COLORS[i])

        axes.set_title(title, fontsize=FS_TITLE, pad=6)
        axes.set_ylabel(metric_name, fontsize=FS_LABEL, labelpad=3)
        axes.axhline(0.5, color="#999999", linewidth=LW_SPINE, linestyle="--", zorder=0)

        axes.set_xticks(range(len(results)))
        axes.set_xticklabels([])
        positions = axes.get_xticks()
        for xpos, label_info in zip(positions, xtick_label_info_list):
            axes.text(xpos, -0.08, label_info[0], ha="center", va="top",
                      fontsize=FS_TICK, rotation=0,
                      transform=axes.get_xaxis_transform())
            if not display_domain_counts:
                continue
            for j, (variant_counts, color) in enumerate(
                    zip(label_info[1], COLORS[:len(results_list)])):
                axes.text(xpos, -0.22 - 0.10 * j,
                          f"+{variant_counts[0]}/\u2212{variant_counts[1]}",
                          ha="center", va="top", fontsize=FS_TICK - 1,
                          rotation=0, color=color,
                          transform=axes.get_xaxis_transform())

        axes.tick_params(axis="both", which="major",
                         labelsize=FS_TICK, width=LW_SPINE,
                         length=2, direction="out")
        axes.patch.set_facecolor("white")
        for spine in axes.spines.values():
            spine.set_linewidth(LW_SPINE)
            spine.set_color("black")
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)

        if len(results_list) > 1:
            axes.legend(fontsize=FS_LEGEND, frameon=False,
                        handlelength=1.2, handletextpad=0.4,
                        loc='upper left', bbox_to_anchor=(1.02, 1))


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







