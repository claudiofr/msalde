import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ─────────────────────────────────────────────────────────────────
rounds_df     = pd.read_csv(os.path.join(OUT_DIR, "auc_by_label_method_by_round.csv"))
domain_df     = pd.read_csv(os.path.join(OUT_DIR, "auc_by_domain.csv"))
landscape_df  = pd.read_csv(os.path.join(OUT_DIR, "protein_landscape_data.csv"))
domains_df    = pd.read_csv(os.path.join(OUT_DIR, "protein_landscape_domains.csv"))
domains_df    = domains_df.drop_duplicates(subset=["name", "assay_source"])
auc_data_df   = pd.read_csv(os.path.join(OUT_DIR, "auc_data.csv")).drop_duplicates(subset=["assay_source", "model"])

# ── Parse model names from landscape data ─────────────────────────────────────
_MODEL_MAP = {
    "LogLikelihood":        "LLR",
    "RandomForest AL":      "RF AL",
    "RandomForest MonteCarlo": "RF MC",
    "5 FOLD CV":            "RF MC",
}
landscape_df["model_short"] = landscape_df["model"].apply(
    lambda s: _MODEL_MAP.get(s.split("/", 1)[1], s.split("/", 1)[1]) if "/" in s else s
)

# ── Compute protein-level Pearson correlations ─────────────────────────────────
protein_pearson = {}
for protein in ["MC4R", "HXK4", "PTEN", "SRC"]:
    protein_pearson[protein] = {}
    for m in ["LLR", "RF AL", "RF MC"]:
        sub = landscape_df[(landscape_df["assay_source"] == protein) &
                           (landscape_df["model_short"] == m)]
        protein_pearson[protein][m] = (
            sub["assay_score"].corr(sub["prediction_score"])
            if len(sub) >= 3 else np.nan
        )

# ── Compute domain-level Pearson correlations ──────────────────────────────────
domain_pearson = {}
for protein in ["MC4R", "HXK4", "PTEN", "SRC"]:
    prot_doms = domains_df[domains_df["assay_source"] == protein].sort_values("start")
    domain_pearson[protein] = {"domains": [], "LLR": [], "RF AL": []}
    for _, row in prot_doms.iterrows():
        domain_pearson[protein]["domains"].append(row["name"])
        for m in ["LLR", "RF AL"]:
            sub = landscape_df[
                (landscape_df["assay_source"] == protein) &
                (landscape_df["model_short"] == m) &
                (landscape_df["mutation_position"] >= row["start"]) &
                (landscape_df["mutation_position"] <= row["end"])
            ]
            domain_pearson[protein][m].append(
                sub["assay_score"].corr(sub["prediction_score"])
                if len(sub) >= 3 else np.nan
            )

PROTEINS = ["MC4R", "HXK4", "PTEN", "SRC"]
LABEL_COLORS = {
    "GOF/LOF": "#1f77b4",
    "GOF_10%": "#ff7f0e",
    "GOF_20%": "#2ca02c",
    "LOF_10%": "#d62728",
    "LOF_20%": "#9467bd",
}

# ── Publication-quality formatting (Nature journal standards) ──────────────────
PUB_RC = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             7,
    "axes.labelsize":        7,
    "axes.titlesize":        7,
    "axes.titleweight":      "bold",
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.major.size":      3.0,
    "ytick.major.size":      3.0,
    "xtick.major.width":     0.75,
    "ytick.major.width":     0.75,
    "axes.linewidth":        0.75,
    "lines.linewidth":       1.25,
    "lines.markersize":      4,
    "patch.linewidth":       0.5,
    "legend.fontsize":       6,
    "legend.title_fontsize": 7,
    "legend.frameon":        True,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "0.8",
    "legend.handlelength":   1.5,
    "legend.handleheight":   0.7,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            600,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
}

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Active-learning convergence curves (GOF/LOF strict labels)
# ══════════════════════════════════════════════════════════════════════════════
gof_lof = rounds_df[rounds_df["model"] == "GOF/LOF"]

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
fig.suptitle("Active Learning Convergence (GOF/LOF Strict Labels)", fontsize=13, y=1.02)

for ax, protein in zip(axes, PROTEINS):
    sub = gof_lof[gof_lof["assay_source"] == protein].sort_values("round_num")
    rounds = sub["round_num"].values
    auc    = sub["auc"].values
    llr    = sub["llr_auc"].iloc[0]

    ax.plot(rounds, auc, marker="o", color="#1f77b4", linewidth=2, label="RF AL")
    ax.axhline(llr, color="#d62728", linestyle="--", linewidth=1.8, label=f"LLR ({llr:.3f})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)

    ax.set_title(protein, fontsize=12, fontweight="bold")
    ax.set_xlabel("Active Learning Round")
    ax.set_xticks(rounds)
    ax.set_ylim(0.45, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("ROC AUC")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure1_active_learning_convergence.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved figure1_active_learning_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Domain-level ROC AUC heatmap (RF AL vs LLR)
# ══════════════════════════════════════════════════════════════════════════════
domain_df = domain_df.dropna(subset=["metric"])

# Build a tidy table: rows = (protein, domain), columns = (RF AL, LLR)
pivot = domain_df.pivot_table(index=["assay_source", "domain"],
                               columns="model", values="metric")
pivot = pivot.reset_index()

# Sort by protein then domain_start
starts = domain_df[["assay_source","domain","domain_start"]].drop_duplicates()
pivot = pivot.merge(starts, on=["assay_source","domain"], how="left")
pivot = pivot.sort_values(["assay_source","domain_start"])

# Create a label combining protein + domain
pivot["label"] = pivot["assay_source"] + " — " + pivot["domain"]

rf_vals  = pivot["RF AL"].values.astype(float)
llr_vals = pivot["LLR"].values.astype(float)
labels   = pivot["label"].values
proteins = pivot["assay_source"].values

n = len(labels)
data = np.column_stack([rf_vals, llr_vals])  # shape (n, 2)

fig, ax = plt.subplots(figsize=(6, n * 0.38 + 1.2))

cmap = plt.cm.RdYlGn
im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0,
               extent=[-0.5, 1.5, n - 0.5, -0.5])

ax.set_xticks([0, 1])
ax.set_xticklabels(["RF AL", "LLR"], fontsize=11)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8)

# Annotate cells with AUC value
for i in range(n):
    for j, val in enumerate([rf_vals[i], llr_vals[i]]):
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black")

# Draw separator lines between proteins
prev_p = None
for i, p in enumerate(proteins):
    if prev_p is not None and p != prev_p:
        ax.axhline(i - 0.5, color="black", linewidth=1.5)
    prev_p = p

plt.colorbar(im, ax=ax, label="ROC AUC", shrink=0.4, pad=0.02)
ax.set_title("Domain-Level ROC AUC\n(RF AL vs. LLR, Strict GOF/LOF Labels)",
             fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure2_domain_auc_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved figure2_domain_auc_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Final round AUC by labeling strategy, per protein
# ══════════════════════════════════════════════════════════════════════════════
_LABEL_DISPLAY = {
    "GOF/LOF": "GOF/LOF (strict)",
    "GOF_10%": "GOF 10%",
    "GOF_20%": "GOF 20%",
    "LOF_10%": "LOF 10%",
    "LOF_20%": "LOF 20%",
}
_LABEL_COLORS_PUB = {
    "GOF/LOF": "#1B4F8A",
    "GOF_10%": "#D55E00",
    "GOF_20%": "#E69F00",
    "LOF_10%": "#7B3F94",
    "LOF_20%": "#B08CC0",
}
label_methods  = ["GOF/LOF", "GOF_10%", "GOF_20%", "LOF_10%", "LOF_20%"]
_max_round     = int(rounds_df["round_num"].max())
_final_round   = rounds_df[(rounds_df["round_num"] == _max_round) &
                            rounds_df["model"].isin(label_methods)].copy()

n_methods  = len(label_methods)
n_proteins = len(PROTEINS)
x          = np.arange(n_proteins)
bar_width  = 0.13
offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

with plt.rc_context(PUB_RC):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))   # double-column (183 mm)

    for idx, method in enumerate(label_methods):
        sub      = _final_round[_final_round["model"] == method]
        auc_vals, auc_errs = [], []
        for p in PROTEINS:
            row = sub[sub["assay_source"] == p]
            auc_vals.append(row["auc"].values[0]     if len(row) > 0 else np.nan)
            auc_errs.append(row["auc_std"].values[0] if len(row) > 0 else np.nan)

        ax.bar(x + offsets[idx], auc_vals, bar_width,
               label=_LABEL_DISPLAY[method],
               color=_LABEL_COLORS_PUB[method],
               edgecolor="white", linewidth=0.25,
               yerr=auc_errs, capsize=2.5,
               error_kw=dict(elinewidth=0.75, ecolor="#333333"))

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.75, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(PROTEINS, fontweight="bold")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.44, 0.88)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.legend(title="Label strategy", loc="upper right",
              ncol=1, borderpad=0.5, labelspacing=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(OUT_DIR, "figure3_label_strategy_auc.png"))
    plt.close()
    print("Saved figure3_label_strategy_auc.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — Protein-level Pearson correlation by model
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_DISPLAY_PEARSON = {
    "LLR":   "LLR",
    "RF AL": "RF AL",
    "RF MC": "RF 5 Fold CV",
}
_MODEL_COLORS_PUB = {
    "LLR":   "#D55E00",
    "RF AL": "#0072B2",
    "RF MC": "#009E73",
}
models_pearson = ["LLR", "RF AL", "RF MC"]

n_proteins = len(PROTEINS)
x          = np.arange(n_proteins)
bar_width  = 0.22
offsets    = np.linspace(-1, 1, 3) * bar_width

with plt.rc_context(PUB_RC):
    fig, ax = plt.subplots(figsize=(3.54, 2.8))   # single-column (89 mm)

    for idx, model in enumerate(models_pearson):
        vals = [protein_pearson[p][model] for p in PROTEINS]
        ax.bar(x + offsets[idx], vals, bar_width,
               label=_MODEL_DISPLAY_PEARSON[model],
               color=_MODEL_COLORS_PUB[model],
               edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(PROTEINS, fontweight="bold")
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(title="Model", loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              borderpad=0.4, labelspacing=0.25, columnspacing=0.8,
              bbox_transform=ax.transAxes)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    fig.savefig(os.path.join(OUT_DIR, "protein_pearson_correlation.png"))
    plt.close()
    print("Saved protein_pearson_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — Domain-level Pearson correlation (LLR vs RF AL)
# ══════════════════════════════════════════════════════════════════════════════
with plt.rc_context(PUB_RC):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True)

    for ax, protein, panel in zip(axes, PROTEINS, "abcd"):
        dp      = domain_pearson[protein]
        domains = dp["domains"]
        xd      = np.arange(len(domains))

        ax.plot(xd, dp["LLR"],   marker="o", color="#D55E00",
                linewidth=1.25, markersize=4, label="LLR", clip_on=False)
        ax.plot(xd, dp["RF AL"], marker="s", color="#0072B2",
                linewidth=1.25, markersize=4, label="RF AL", clip_on=False)
        ax.axhline(0, color="#888888", linestyle=":", linewidth=0.75)

        ax.set_title(f"{panel}  {protein}", loc="left", pad=4, fontsize=8)
        ax.set_xticks(xd)
        ax.set_xticklabels(domains, rotation=70, ha="right", fontsize=6)
        ax.set_ylim(-0.8, 1.05)
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        if protein == "MC4R":
            ax.set_ylabel("Pearson r")
            ax.legend(loc="upper right", borderpad=0.4, labelspacing=0.2)

    fig.savefig(os.path.join(OUT_DIR, "domain_pearson_correlation.png"))
    plt.close()
    print("Saved domain_pearson_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — Summary: final model performance by protein (all three models)
# ══════════════════════════════════════════════════════════════════════════════
# RF AL std from final round (GOF/LOF strict)
_r_final_strict = rounds_df[(rounds_df["round_num"] == rounds_df["round_num"].max()) &
                              (rounds_df["model"] == "GOF/LOF")]
_rf_al_std      = {row["assay_source"]: row["auc_std"]
                   for _, row in _r_final_strict.iterrows()}

_AUC_MODELS  = ["ESM2_LLR_ALL_PRED", "RF_AL_VAL", "RF_5_FOLD_CV"]
_AUC_DISPLAY = {
    "ESM2_LLR_ALL_PRED": "LLR",
    "RF_AL_VAL":         "RF AL",
    "RF_5_FOLD_CV":      "RF 5 Fold CV",
}
_AUC_COLORS  = {
    "ESM2_LLR_ALL_PRED": "#D55E00",
    "RF_AL_VAL":         "#0072B2",
    "RF_5_FOLD_CV":      "#009E73",
}

n_models  = len(_AUC_MODELS)
x         = np.arange(len(PROTEINS))
bar_width = 0.22
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

with plt.rc_context(PUB_RC):
    fig, ax = plt.subplots(figsize=(3.54, 2.8))   # single-column (89 mm)

    for idx, model in enumerate(_AUC_MODELS):
        vals, errs = [], []
        for p in PROTEINS:
            row = auc_data_df[(auc_data_df["assay_source"] == p) &
                               (auc_data_df["model"] == model)]
            vals.append(row["auc"].values[0] if len(row) > 0 else np.nan)
            errs.append(_rf_al_std.get(p, 0) if model == "RF_AL_VAL" else 0)

        ax.bar(x + offsets[idx], vals, bar_width,
               label=_AUC_DISPLAY[model],
               color=_AUC_COLORS[model],
               edgecolor="white", linewidth=0.25,
               yerr=errs, capsize=2.5,
               error_kw=dict(elinewidth=0.75, ecolor="#333333"))

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.75, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(PROTEINS, fontweight="bold")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.44, 1.00)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(title="Model", loc="upper left",
              borderpad=0.5, labelspacing=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(OUT_DIR, "protein_final_auc.png"))
    plt.close()
print("Saved protein_final_auc.png")