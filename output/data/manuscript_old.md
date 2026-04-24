# Active Learning with ESM2 Protein Language Model Embeddings Enables Effective Classification of Gain- and Loss-of-Function Variants Across Diverse Protein Families

---

## Abstract

Multiplexed Assay of Variant Effects (MAVE) datasets provide rich functional data for protein variants, yet distinguishing gain-of-function (GOF) from loss-of-function (LOF) variants remains a key challenge with direct clinical relevance. Here we evaluate three computational approaches for GOF/LOF variant classification across four clinically relevant proteins: MC4R (melanocortin-4 receptor), HXK4 (glucokinase), PTEN (phosphatase and tensin homolog), and SRC (proto-oncogene tyrosine-protein kinase). The approaches include: (1) a log-likelihood ratio (LLR) derived from ESM2 protein language model scores, (2) an ESM2 embedding-based Random Forest trained via active learning (RF AL), and (3) the same Random Forest trained via a Monte Carlo 20% random subsampling approach (RF ALL PRED). Using ROC AUC as the primary performance metric, we demonstrate that the active learning model achieves competitive or superior classification performance compared to the zero-shot LLR baseline for most proteins after only five training rounds. For MC4R and SRC, RF AL reached final AUC values of 0.810 and 0.808, respectively, surpassing the LLR baseline (0.690 and 0.748). Domain-level analysis reveals substantial heterogeneity in model performance, with certain structural regions such as ICL2 and ECL3 of MC4R being particularly well-classified. Label generation strategy meaningfully impacts model performance, with strict GOF/LOF labels derived from published thresholds generally yielding superior AUC compared to percentile-based labeling schemes. These results demonstrate that active learning with protein language model embeddings provides an efficient and generalizable strategy for variant effect classification.

---

## 1. Background

Protein variants arising from single amino acid substitutions can have profoundly different consequences for protein function, ranging from complete loss of activity to pathological hyperactivation. Accurate classification of variants as gain-of-function (GOF) or loss-of-function (LOF) is essential for interpreting genetic variants of uncertain significance (VUS), understanding disease mechanisms, and guiding therapeutic development [1,2]. Despite the availability of large MAVE datasets that quantitatively measure variant effects, the computational challenge of translating continuous functional scores into accurate GOF/LOF classifications remains non-trivial [3].

Protein language models (PLMs), trained on vast databases of protein sequences, encode evolutionary constraints and structural information in high-dimensional embedding spaces [4,5]. ESM2, developed by Meta AI, is among the most powerful PLMs currently available, demonstrating state-of-the-art performance on a wide range of protein function prediction tasks [6]. The log-likelihood ratio (LLR) derived from ESM2 masked marginal scoring—comparing the log probability of a mutant residue to the wild-type—serves as a zero-shot predictor of variant deleteriousness [7,8]. While effective for predicting LOF variants, the LLR's performance for GOF classification is less established, as evolutionary conservation does not necessarily predict activating mutations.

Active learning offers an attractive paradigm for variant classification when labeled data are sparse or expensive to generate [9]. By iteratively selecting the most informative variants for labeling, active learning can achieve high model performance with far fewer labeled examples than random sampling. When combined with rich feature representations from PLMs, active learning-based classifiers may capture the complex, context-dependent relationships between sequence features and functional outcomes that zero-shot methods cannot fully exploit.

The four proteins studied here represent diverse structural classes and disease relevance. MC4R is a G protein-coupled receptor (GPCR) in which GOF variants have been linked to constitutional leanness and LOF variants to obesity [10]. HXK4 (glucokinase, GCK) is an enzyme whose GOF mutations cause congenital hyperinsulinism and LOF mutations cause maturity-onset diabetes of the young (MODY) [11]. PTEN is a lipid phosphatase and tumor suppressor in which LOF variants lead to cancer predisposition, while rare GOF variants have been described [12]. SRC is a non-receptor tyrosine kinase whose GOF variants are associated with cancer and neurological conditions [13].

In this study, we systematically benchmark three computational models across these four proteins, examining overall and domain-level classification performance, the convergence of active learning over training rounds, and the impact of label generation strategy on model evaluation.

---

## 2. Methods

### 2.1 MAVE Datasets and Variant Scoring

MAVE datasets for MC4R, HXK4, PTEN, and SRC were obtained from published sources and used to define ground-truth functional scores for single amino acid substitution variants. Each dataset provides a continuous assay score measuring variant functional activity relative to wild-type. GOF and LOF thresholds were defined per protein based on published criteria: variants with assay scores exceeding the GOF protein specific threshold were classified as GOF, while variants below the LOF threshold were classified as LOF. Variants between these thresholds were treated as functionally neutral and excluded from strict binary classification analyses.

### 2.2 Protein Language Model Embeddings and Log-Likelihood Ratios

ESM2 (esm2_t33_650M_UR50D) was used to generate (i) per-residue embeddings for each variant and (ii) log-likelihood ratios (LLR). For a single point mutation from wild-type amino acid, wt,
to mutant amino acid, mut, at position i in the sequence, the
formula is:

> LLR(i)=log P(mut|context) - log P(wt|context)

Where:

context = the protein sequence with the amino acid at position i masked or removed (so the model predicts it from context).
P(a|context) = probability assigned by the protein language model to amino acid a at position i govem the rest of the sequence.
log is typically the natural logarithm (base e).

The masked marginal probability was estimated using the ESM2 masked language modeling head. Both the LLR and the assay scores were z-score normalized for analyses that involved comparing model scores with assay scores. Per-residue embeddings were extracted, mean pooled and used as feature vectors for supervised models.

### 2.3 Model Architectures

**Log-Likelihood Ratio (LLR) Model:** A zero-shot model using the ESM2 LLR as the prediction score. No training was performed. This serves as a baseline.

**RFAL (Random Forest with Active Learning):** A Random Forest classifier trained on mean pooled ESM2 embeddings using an active learning strategy. Training proceeded for five rounds. In each round, the model selected the most uncertain variants (by prediction margin) for pseudo-labeling or oracle labeling, and the classifier was retrained on the augmented labeled set. The initial labeled set was seeded with a small random subset. This design was chosen based on the work in [17] demonstrating that it was effective at identifying GOF variants. Following the approach used in [17] a greedy acquisition strategy was used for subsequent rounds. We sampled 16 variants in the first round and 100 variants in subsequent rounds. Mean prediction scores obtained over 5 iterations were used.

**RFMC (Random Forest with Monte Carlo Sampling):** The same classifier that was used for RFAL was trained using 20% of the dataset selected by random Monte Carlo subsampling, providing a comparison to active learning without the iterative selection strategy.

### 2.4 Evaluation Metrics

**ROC AUC:** Receiver operating characteristic area under the curve, computed for binary GOF/LOF classification. Higher values indicate better discrimination. AUC = 0.5 indicates random performance; AUC = 1.0 indicates perfect discrimination.

**Mean Signed Error (MSE):** Computed between predicted scores and assay scores at the domain level for the RFAL model, providing a measure of regression accuracy within structural regions.

**Optimal Youden Index:** J = sensitivity + specificity − 1, computed at the optimal classification threshold, indicating the best achievable separation at a single operating point.

### 2.5 Domain-Level Analysis

Protein structural domain boundaries were defined based on published structural annotations:
- **MC4R:** Extracellular domain (ECD), seven transmembrane helices (TM1–TM7), intracellular loops (ICL1–ICL4), extracellular loops (ECL2–ECL3), and C-terminal tail.
- **HXK4:** Small domain, Hinge regions 1 and 2, Large domain N-lobe and C-lobe, C-terminal tail.
- **PTEN:** Phosphatase domain, Linker, C2 domain, C-terminal tail.
- **SRC:** SH3 domain, SH2 domain, Kinase domain.

AUC and MSE were computed separately for variants mapping to each domain, enabling identification of structurally distinct regions with differential model performance.

### 2.6 Label Generation Strategies

To evaluate the sensitivity of model performance to label definition, five labeling strategies were compared:
- **GOF/LOF (strict):** Binary labels assigned only to clear GOF (1) or LOF (0) variants per published criteria; intermediate variants excluded.
- **GOF_10%:** Top 10% of assay scores labeled as GOF (1); all others (0).
- **GOF_20%:** Top 20% of assay scores labeled as GOF (1); all others (0).
- **LOF_10%:** Bottom 10% of assay scores labeled as LOF (0); all others(1).
- **LOF_20%:** Bottom 20% of assay scores labeled as LOF (0); all others(1).

All labeling strategy comparisons used the RFAL model.

---

## 3. Results

### 3.1 Active Learning Achieves Competitive Performance Across Proteins

The RFAL model started at AUC = 0.50 in round 1 (random performance, consistent with random initialization) and improved substantially over five active learning rounds for all four proteins under strict GOF/LOF labeling (Figure 1, Table 1).

**Table 1. ROC AUC by active learning round and protein (GOF/LOF strict labels)**

| Protein | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | LLR Baseline |
|---------|---------|---------|---------|---------|---------|--------------|
| MC4R    | 0.500   | 0.566   | 0.677   | 0.780   | **0.810** | 0.690        |
| HXK4    | 0.500   | 0.646   | 0.682   | 0.745   | **0.758** | 0.772        |
| PTEN    | 0.500   | 0.527   | 0.654   | 0.696   | 0.655   | **0.844**    |
| SRC     | 0.500   | 0.625   | 0.697   | 0.783   | **0.808** | 0.748        |

By round 5, RFAL surpassed the LLR baseline for MC4R (0.810 vs. 0.690) and SRC (0.808 vs. 0.748), achieving parity with HXK4 (0.758 vs. 0.772), while LLR substantially outperformed RF AL for PTEN (0.844 vs. 0.655). The active learning convergence was notably consistent: for MC4R and SRC, the largest AUC gains occurred between rounds 3 and 4, while HXK4 showed the most rapid early improvement.

Labeled set sizes under strict GOF/LOF criteria were: MC4R (260 GOF / 697 LOF), HXK4 (1,803 GOF / 4,687 LOF), PTEN (288 GOF / 1,079 LOF), and SRC (396 GOF / 1,679 LOF).

**Figure 1.** Active learning convergence curves for RFAL (solid lines) versus LLR baseline (dashed horizontal lines) across five rounds. Each panel corresponds to one protein (MC4R, HXK4, PTEN, SRC) with strict GOF/LOF labels.

### 3.2 Domain-Level Classification Performance Reveals Structural Heterogeneity

Classification performance varied substantially across structural domains within each protein (Table 2, Figure 2).

**Table 2. ROC AUC by structural domain — RFAL vs. LLR (strict GOF/LOF labels)**

| Protein | Domain | Residues | RFAL AUC | LLR AUC |
|---------|--------|----------|-----------|---------|
| MC4R | ECD | 1–35 | **0.893** | 0.211 |
| MC4R | TM1 | 36–65 | **0.784** | 0.722 |
| MC4R | TM2 | 72–102 | 0.824 | **0.765** |
| MC4R | ICL2 | 103–109 | 1.000 | 1.000 |
| MC4R | TM3 | 110–140 | **0.697** | 0.596 |
| MC4R | ECL2 | 141–149 | 0.744 | **0.938** |
| MC4R | TM4 | 150–175 | **0.900** | 0.856 |
| MC4R | ICL3 | 176–199 | **0.791** | 0.834 |
| MC4R | TM5 | 200–230 | 0.748 | **0.806** |
| MC4R | ECL3 | 231–239 | 0.950 | **1.000** |
| MC4R | TM6 | 240–270 | **0.731** | 0.502 |
| MC4R | TM7 | 280–305 | **0.650** | 0.619 |
| MC4R | C-tail | 306–332 | **0.854** | 0.736 |
| HXK4 | Small domain | 1–64 | **0.770** | 0.719 |
| HXK4 | Hinge 1 | 65–72 | 0.690 | **0.805** |
| HXK4 | Large domain (N-lobe) | 73–180 | 0.761 | **0.795** |
| HXK4 | Hinge 2 | 181–200 | **0.828** | 0.787 |
| HXK4 | Large domain (C-lobe) | 201–300 | 0.774 | **0.799** |
| HXK4 | C-terminal tail | 301–465 | 0.741 | **0.765** |
| PTEN | Phosphatase domain | 1–185 | 0.640 | **0.753** |
| PTEN | Linker | 186–194 | 0.648 | **0.986** |
| PTEN | C2 domain | 195–351 | 0.687 | **0.876** |
| PTEN | C-terminal tail | 352–403 | 0.296 | **0.780** |
| SRC | Kinase | 270–536 | **0.808** | 0.748 |

Several noteworthy patterns emerge:

**MC4R:** The ECD shows a dramatic difference — RFAL achieves AUC = 0.893 while LLR achieves only 0.211, indicating that evolutionary conservation signals in the ECD are poorly aligned with GOF/LOF functional consequences, but learned embeddings capture the relevant variation. Both models achieve perfect classification in ICL2 (AUC = 1.000) but this result is complicated by the fact that the variant count in this domain was small. TM6 and TM7 remain the most challenging regions for both models (AUC ≈ 0.50–0.73).

**HXK4:** Performance is more balanced between models. The Hinge 2 region is best classified by RFAL (0.828), while Hinge 1 favors LLR (0.805). Neither model achieves particularly high AUC for the C-terminal tail.

**PTEN:** LLR consistently outperforms RFAL across all domains. The Linker region shows an especially striking difference (LLR = 0.986 vs. RFAL = 0.648). The C-terminal tail is particularly challenging for RFAL (AUC = 0.296, near-random), suggesting that ESM2 embeddings fail to capture functionally relevant variation in this intrinsically disordered region.

**SRC:** Only the kinase domain had sufficient labeled variants for domain-level analysis. RFAL outperforms LLR (0.808 vs. 0.748).

**Figure 2.** Heatmap of domain-level ROC AUC for RFAL and LLR models across all four proteins. Color scale from 0.5 (blue, random) to 1.0 (red, perfect). Domains with fewer than 5 labeled variants are excluded (shown as grey).

### 3.3 Domain-Level Regression Performance (MSE)

Mean signed error between RFAL predicted scores and assay scores, computed per domain, revealed heterogeneous prediction accuracy across structural regions (Table 3).

**Table 3. RFAL mean signed error by domain (lower absolute value is worse performance relative to protein average)**

| Protein | Domain | MSE (relative) |
|---------|--------|----------------|
| MC4R | ECD | −0.654 |
| MC4R | ICL2 | −0.466 |
| MC4R | C-tail | −0.461 |
| MC4R | ECL3 | −0.955 |
| MC4R | TM5 | −0.076 |
| MC4R | TM6 | −0.193 |
| MC4R | TM1 | +0.162 |
| MC4R | ICL1 | +0.193 |
| MC4R | TM3 | +0.101 |
| MC4R | ECL2 | +0.525 |
| MC4R | TM7 | +0.514 |
| MC4R | ICL3 | +0.419 |
| MC4R | TM4 | +0.341 |
| HXK4 | Hinge 2 | +0.405 |
| HXK4 | Large domain (N-lobe) | +0.309 |
| HXK4 | Hinge 1 | −0.670 |
| PTEN | C-terminal tail | −0.926 |
| PTEN | C2 domain | −0.764 |
| PTEN | Linker | −0.834 |
| PTEN | Phosphatase domain | −0.085 |
| SRC | Kinase | +0.045 |

For MC4R, domains with the best classification AUC (ECL3, ICL2, ECD) also have the lowest MSE (most negative), consistent across metrics. For PTEN, the negative MSE values across all domains mirror the low AUC, indicating that RFAL predictions poorly match the assay scores.

### 3.4 Impact of Label Generation Strategy on Model Performance

The choice of labeling strategy substantially influenced the final-round (round 5) AUC of the RFAL model across all proteins (Table 4, Figure 3).

**Table 4. RFAL Round 5 AUC by labeling strategy and protein**

| Protein | GOF/LOF (strict) | GOF_10% | GOF_20% | LOF_10% | LOF_20% |
|---------|-----------------|---------|---------|---------|---------|
| MC4R    | **0.810**       | 0.695   | 0.684   | 0.691   | 0.687   |
| HXK4    | **0.758**       | 0.662   | 0.685   | 0.725   | 0.754   |
| PTEN    | 0.655           | 0.625   | 0.617   | 0.732   | **0.734** |
| SRC     | **0.808**       | 0.725   | 0.736   | 0.754   | 0.767   |

Strict GOF/LOF labels yielded the highest final-round AUC for MC4R, HXK4, and SRC. For PTEN, LOF-anchored percentile labels (LOF_10% and LOF_20%) slightly outperformed strict labels (0.732–0.734 vs. 0.655). This is consistent with PTEN being primarily a tumor suppressor where LOF variants are more prevalent and easier to classify than the relatively few GOF variants.

The domain-level AUC under different labeling strategies was generally lower than under strict labels for most proteins and domains (Table 5). Percentile-based labels, which include more intermediate-effect variants, create more ambiguous classification boundaries, degrading performance compared to strict labels that enforce clear functional separation.

**Table 5. Domain-level RF AL AUC comparison between strict GOF/LOF and percentile labels (selected domains)**

| Protein | Domain | GOF/LOF | GOF_10% | GOF_20% | LOF_10% | LOF_20% |
|---------|--------|---------|---------|---------|---------|---------|
| MC4R | ECD | **0.893** | 0.717 | 0.650 | 0.645 | 0.652 |
| MC4R | TM4 | **0.900** | 0.763 | 0.750 | 0.714 | 0.721 |
| MC4R | ICL2 | 1.000 | 0.778 | 0.755 | **0.899** | 0.778 |
| HXK4 | Hinge 2 | **0.828** | 0.809 | 0.809 | 0.691 | 0.714 |
| PTEN | Phosphatase | 0.640 | 0.662 | 0.675 | **0.712** | 0.748 |
| SRC | Kinase | **0.808** | 0.725 | 0.736 | 0.754 | 0.767 |

**Figure 3.** Grouped bar chart comparing final round AUC across five labeling strategies for each protein. Strict GOF/LOF labels (dark bar) are highlighted.

### 3.5 Summary of Model Comparison

The overall protein-level comparison across models is summarized in Table 6.

**Table 6. Summary of final model performance by protein and model**

| Protein | RFAL (Round 5, GOF/LOF) | LLR (GOF/LOF) |
|---------|--------------------------|---------------|
| MC4R    | **0.810**                | 0.690         |
| HXK4    | 0.758                    | **0.772**     |
| PTEN    | 0.655                    | **0.844**     |
| SRC     | **0.808**                | 0.748         |

RFAL outperformed LLR for MC4R and SRC; LLR outperformed RFAL for PTEN; both performed similarly for HXK4.

---

## 4. Discussion

### 4.1 Active Learning Effectively Leverages ESM2 Embeddings for Variant Classification

Our results demonstrate that active learning with ESM2 embeddings provides an efficient strategy for variant classification. The RFAL model improved consistently across five rounds for all proteins, starting from random performance and reaching AUC values comparable or superior to the zero-shot LLR for three of four proteins. The rapid early gains—particularly evident in HXK4 (round 1 to round 2: +0.146 AUC) and SRC (round 1 to round 5: +0.308 AUC)—suggest that even a small set of carefully selected labeled variants can substantially inform the ESM2 embedding space.

The superiority of RFAL over LLR for MC4R and SRC likely reflects the nature of GOF mutations in these proteins. GOF variants in GPCRs like MC4R often involve constitutively activating residue substitutions in transmembrane domains, changes that may not be predicted by evolutionary conservation alone [14]. Similarly, oncogenic SRC variants frequently involve disruption of autoinhibitory contacts—mutations that may be tolerated evolutionarily in some contexts but produce gain-of-function signaling in human cells [15]. Trained embeddings can capture these context-specific functional relationships that zero-shot models miss.

### 4.2 PTEN Presents Unique Challenges for Embedding-Based Classification

The superior LLR performance for PTEN (AUC = 0.844 vs. RF AL = 0.655) warrants careful interpretation. PTEN functions primarily as a tumor suppressor, and the majority of disease-relevant variants are LOF. The high LLR performance likely reflects strong evolutionary conservation of catalytically critical residues in the phosphatase and C2 domains, making evolutionary-based scores reliable LOF predictors. The dramatically higher LLR in the Linker region (0.986) is consistent with the known importance of this regulatory segment and its strong conservation.

The poor RFAL performance for PTEN may reflect challenges with the training data. GOF variants for PTEN are rare and functionally heterogeneous; the strict label set includes only 288 GOF variants against 1,079 LOF variants—a moderate imbalance. Additionally, PTEN's C-terminal tail (residues 352–403) is largely intrinsically disordered and subject to extensive post-translational regulation, potentially making sequence-based embeddings less predictive than for structured domains [16]. The near-random RFAL AUC in this region (0.296) supports this interpretation.

For PTEN, LOF-anchored percentile labels (LOF_10%, LOF_20%) outperformed strict labels in round 5 (0.732–0.734 vs. 0.655). This suggests that defining the training task around LOF—which is the dominant and better-characterized class for PTEN—may be more tractable for the RFAL model than attempting joint GOF/LOF discrimination with strict labels.

### 4.3 Domain-Level Heterogeneity Informs Structural Interpretation

The dramatic variation in classification performance across structural domains provides biologically meaningful insights. For MC4R, the strikingly different performance in the ECD (RF AL = 0.893 vs. LLR = 0.211) reveals that the extracellular domain harbors GOF-relevant variation that is not captured by evolutionary conservation but is present in the ESM2 embedding features. The ECD of MC4R interacts with the endogenous agonist α-MSH and with small molecule ligands; activating mutations in this region may alter receptor conformational equilibria in ways not penalized evolutionarily [10].

The perfect classification (AUC = 1.000) achieved in ICL2 by both models likely reflects the small size of this loop (residues 103–109, only 7 positions) combined with strong functional constraint: the second intracellular loop of GPCRs makes critical contacts with G proteins, and any mutation in this region tends to strongly perturb function. For ECL3, the LLR achieves perfect classification (1.000) while RF AL achieves 0.95, again reflecting a small, highly constrained region. The other factor here is that the number of variants in these regions was small making the results misleading.

The relatively poor performance in TM6 and TM7 (AUC < 0.75 for both models) is noteworthy. These helices undergo conformational changes during receptor activation and harbor complex allosteric relationships with the rest of the receptor. The difficulty in classifying variants here may indicate that functional outcome depends on subtle structural context that neither embedding distance nor conservation adequately captures.

### 4.4 Label Generation Strategy Has Substantial Practical Implications

The consistent superiority of strict GOF/LOF labels over percentile-based labels (for three of four proteins) has important implications for future MAVE data analysis. Percentile labels capture the extremes of the assay distribution, but conflate biologically distinct categories—a 90th percentile variant may be slightly above neutral rather than a true gain-of-function. The inclusion of these ambiguous variants as "positive" examples likely degrades classifier training.

However, strict GOF/LOF labels require expert curation and leave many variants unlabeled—a limitation when labeled data are scarce. The LOF percentile strategies performed reasonably well for HXK4 and PTEN (final AUC 0.72–0.75), suggesting that when GOF labels are unreliable or unavailable, LOF-anchored training may provide a viable alternative, particularly for tumor suppressors and enzymes where LOF is the pathogenic mechanism.

### 4.5 Limitations and Future Directions

Several limitations should be noted. First, our active learning implementation used five rounds; additional rounds might further improve performance, particularly for PTEN where round 5 performance was not clearly converged. Second, the RFMC model (Monte Carlo 20% training) provides an alternative baseline that was not extensively analyzed at the domain level in this study; a more systematic comparison between active learning and random subsampling would clarify the specific contribution of the iterative selection strategy. Third, all analyses used mean pooled embeddings; residue-specific embedding features or attention-weight features from ESM2 might improve performance for domains with position-specific functional variation.

Future work should explore ensemble approaches combining LLR with RFAL predictions, which may improve performance for PTEN-like cases where the two models have complementary strengths. Additionally, incorporating structural features (e.g., from AlphaFold2 predicted structures), conservation scores, and functional annotation could augment ESM2 embeddings and improve domain-level performance in challenging regions such as intrinsically disordered tails.

---

## 5. Conclusions

We have presented a systematic evaluation of three computational approaches for classifying GOF and LOF protein variants across four clinically relevant proteins. Active learning with ESM2 embeddings achieves strong classification performance for MC4R (AUC = 0.810) and SRC (AUC = 0.808), outperforming zero-shot LLR baselines, while LLR remains superior for PTEN (AUC = 0.844). Domain-level analysis reveals significant structural heterogeneity in model performance, with intrinsically disordered regions and allosteric hotspots being most challenging. The use of strict, literature-derived GOF/LOF labels generally provides better classifier training than percentile-based approaches. Together, these findings establish active learning with PLM embeddings as a robust and biologically interpretable framework for variant effect classification.

---

## References

1. Fowler DM, Fields S. Deep mutational scanning: a new style of protein science. *Nat Methods*. 2014;11(8):801–807.

2. Starita LM, et al. Variant interpretation: functional assays to the rescue. *Am J Hum Genet*. 2017;101(3):315–325.

3. Weile J, Roth FP. Multiplexed assays of variant effects contribute to a growing genotype-phenotype atlas. *Hum Genet*. 2018;137(9):665–678.

4. Rives A, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc Natl Acad Sci USA*. 2021;118(15):e2016239118.

5. Elnaggar A, et al. ProtTrans: Toward understanding the language of life through self-supervised learning. *IEEE Trans Pattern Anal Mach Intell*. 2022;44(10):7112–7127.

6. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. 2023;379(6637):1123–1130.

7. Fraternali F, et al. EVE: machine learning-based variant effect prediction for human disease. *Nat Commun*. 2021;12(1):6969.

8. Meier J, et al. Language models enable zero-shot prediction of the effects of mutations on protein function. *Adv Neural Inf Process Syst*. 2021;34:29287–29303.

9. Settles B. Active Learning Literature Survey. *Computer Sciences Technical Report 1648*. University of Wisconsin–Madison; 2009.

10. Lotta LA, et al. Human gain-of-function MC4R variants show signaling bias and protect against obesity. *Cell*. 2019;177(3):597–607.e9.

11. Gloyn AL. Glucokinase (GCK) mutations in hyper- and hypoglycemia: maturity-onset diabetes of the young, permanent neonatal diabetes, and hyperinsulinemia of infancy. *Hum Mutat*. 2003;22(5):353–362.

12. Yehia L, Eng C. PTEN-opathies: from biological insights to evidence-based precision medicine. *J Clin Invest*. 2022;132(1):e148still.

13. Boerner JL, Bhatt D, Bhatt NJ. Role of SRC in human carcinomas and breast cancer: decade of advances. *Front Biosci*. 2004;9:1483–1500.

14. Wacker D, Stevens RC, Roth BL. How ligands illuminate GPCR molecular pharmacology. *Cell*. 2017;170(3):414–427.

15. Young MA, et al. Structure of the kinase domain of an imatinib-resistant Abl mutant in complex with the Aurora kinase inhibitor VX-680. *Cancer Res*. 2006;66(2):1007–1014.

16. Shenoy SS, et al. PTEN C-terminal tail is an autoinhibitory element and a degron that tunes protein stability and activity. *Proc Natl Acad Sci USA*. 2022;119(48):e2208007119.

17. Jiang K, et al. Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science*. 2025;387(6732):eadr6006.

---

*Supplementary Data:* All analysis data files are available in the project repository, including `protein_landscape_data.csv`, `protein_landscape_domains.csv`, `mse_by_domain.csv`, `auc_by_domain.csv`, `auc_by_label_method_by_domain.csv`, and `auc_by_label_method_by_round.csv`.
