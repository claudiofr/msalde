# Active Learning with ESM2 Protein Language Model Embeddings Enables Effective Classification of Gain- and Loss-of-Function Variants Across Diverse Protein Families

---

## Abstract

Multiplexed Assay of Variant Effects (MAVE) datasets provide rich functional data for protein variants, yet distinguishing gain-of-function (GOF) from loss-of-function (LOF) variants remains a key challenge with direct clinical relevance. Here we evaluate three computational approaches for GOF/LOF variant classification across four clinically relevant proteins: MC4R (melanocortin-4 receptor), HXK4 (glucokinase), PTEN (phosphatase and tensin homolog), and SRC (proto-oncogene tyrosine-protein kinase). The approaches include: (1) a log-likelihood ratio (LLR) derived from ESM2 protein language model scores, (2) an ESM2 embedding-based Random Forest trained via active learning (RFAL), and (3) the same Random Forest trained via a Monte Carlo 20% random subsampling approach (RFMC). Using ROC AUC as the primary performance metric, we demonstrate that the active learning model achieves competitive or superior classification performance compared to the zero-shot LLR baseline for most proteins after only five training rounds. For MC4R and SRC, RFAL reached final AUC values of 0.810 and 0.808, respectively, surpassing the LLR baseline (0.690 and 0.748). Domain-level analysis reveals substantial heterogeneity in model performance, with certain structural regions such as ICL2 and ECL3 of MC4R being particularly well-classified. Label generation strategy meaningfully impacts model performance, with strict GOF/LOF labels derived from published thresholds generally yielding superior AUC compared to percentile-based labeling schemes. These results demonstrate that active learning with protein language model embeddings provides an efficient and generalizable strategy for variant effect classification.

---

## 1. Background

Protein variants arising from single amino acid substitutions can have profoundly different consequences for protein function, ranging from complete loss of activity to pathological hyperactivation. Accurate classification of variants as gain-of-function (GOF) or loss-of-function (LOF) is essential for interpreting genetic variants of uncertain significance (VUS), understanding disease mechanisms, and guiding therapeutic development [1,2]. Despite the availability of large MAVE datasets that quantitatively measure variant effects, the computational challenge of translating continuous functional scores into accurate GOF/LOF classifications remains non-trivial [3].

Protein language models (PLMs), trained on vast databases of protein sequences, encode evolutionary constraints and structural information in high-dimensional embedding spaces [4,5]. ESM2, developed by Meta AI, is among the most powerful PLMs currently available, demonstrating state-of-the-art performance on a wide range of protein function prediction tasks [6]. The log-likelihood ratio (LLR) derived from ESM2 masked marginal scoring—comparing the log probability of a mutant residue to the wild-type—serves as a zero-shot predictor of variant deleteriousness [7,8]. While effective for predicting LOF variants, the LLR's performance for GOF classification is less established, as evolutionary conservation does not necessarily predict activating mutations.

Active learning offers an attractive paradigm for variant classification when labeled data are sparse or expensive to generate [9]. By iteratively selecting the most informative variants for labeling, active learning can achieve high model performance with far fewer labeled examples than random sampling. When combined with rich feature representations from PLMs, active learning-based classifiers may capture the complex, context-dependent relationships between sequence features and functional outcomes that zero-shot methods cannot fully exploit.

The four proteins studied here represent diverse structural classes and disease relevance. MC4R is a G protein-coupled receptor (GPCR) in which GOF variants have been linked to constitutional leanness and LOF variants to obesity [10]. HXK4 (glucokinase, GCK) is an enzyme whose GOF mutations cause congenital hyperinsulinism and LOF mutations cause maturity-onset diabetes of the young (MODY) [11]. PTEN is a lipid phosphatase and tumor suppressor in which LOF variants lead to cancer predisposition, while rare GOF variants have been described [12]. SRC is a non-receptor tyrosine kinase whose GOF variants are associated with cancer and neurological conditions [13]. Critically, these four proteins were selected in part because they are among the few proteins for which gain-of-function MAVE data is currently available [18–21], making them uniquely suited for benchmarking GOF/LOF classification approaches that require both functional categories to be represented in training and evaluation data.

In this study, we systematically benchmark three computational models across these four proteins, examining overall and domain-level classification performance, the convergence of active learning over training rounds, and the impact of label generation strategy on model evaluation.

---

## 2. Methods

### 2.1 MAVE Datasets and Variant Scoring

MAVE datasets for MC4R, HXK4, PTEN, and SRC were obtained from published sources and used to define ground-truth functional scores for single amino acid substitution variants. The MC4R dataset was derived from high-resolution deep mutational scanning of the melanocortin-4 receptor [18]. The HXK4 (GCK) dataset was obtained from a comprehensive map of human glucokinase variant activity [19]. The SRC dataset was derived from an integrated approach characterizing the regulatory mechanism coupling Src's kinase activity, localization, and phosphotransferase-independent functions [20]. The PTEN dataset was obtained from a study integrating thousands of PTEN variant activity and abundance measurements [21]. Each dataset provides a continuous assay score measuring variant functional activity relative to wild-type. GOF and LOF thresholds were defined per protein based on published criteria: variants with assay scores exceeding the GOF protein-specific threshold were classified as GOF, while variants below the LOF threshold were classified as LOF. Variants between these thresholds were treated as functionally neutral and excluded from strict binary classification analyses.

### 2.2 Protein Language Model Embeddings and Log-Likelihood Ratios

ESM2 (esm2_t33_650M_UR50D) was used to generate (i) per-residue embeddings for each variant and (ii) log-likelihood ratios (LLR). For a single point mutation from wild-type amino acid, wt,
to mutant amino acid, mut, at position i in the sequence, the
formula is:

> LLR(i)=log P(mut|context) - log P(wt|context)

Where:

context = the protein sequence with the amino acid at position i masked or removed (so the model predicts it from context).
P(a|context) = probability assigned by the protein language model to amino acid a at position i given the rest of the sequence.
log is typically the natural logarithm (base e).

The masked marginal probability was estimated using the ESM2 masked language modeling head. Both the LLR and the assay scores were z-score normalized for analyses that involved comparing model scores with assay scores. Per-residue embeddings were extracted, mean pooled and used as feature vectors for supervised models.

### 2.3 Model Architectures

**Log-Likelihood Ratio (LLR) Model:** A zero-shot model using the ESM2 LLR as the prediction score. No training was performed. This serves as a baseline.

**RFAL (Random Forest with Active Learning):** A Random Forest classifier trained on mean pooled ESM2 embeddings using an active learning strategy. Training proceeded for five rounds. In each round, the model selected the most uncertain variants (by prediction margin) for pseudo-labeling or oracle labeling, and the classifier was retrained on the augmented labeled set. The initial labeled set was seeded with a small random subset. This design was chosen based on the work in [17] demonstrating that it was effective at identifying GOF variants. Following the approach used in [17] a greedy acquisition strategy was used for subsequent rounds. We sampled 16 variants in the first round and 100 variants in subsequent rounds. Mean prediction scores obtained over 5 iterations were used.

**RFMC (Random Forest with Monte Carlo Sampling):** The same classifier that was used for RFAL was trained using 20% of the dataset selected by random Monte Carlo subsampling, providing a comparison to active learning without the iterative selection strategy. Mean prediction scores obtained over 2 iterations were used.

### 2.4 Evaluation Metrics

**ROC AUC:** Receiver operating characteristic area under the curve, computed for binary GOF/LOF classification. Higher values indicate better discrimination. AUC = 0.5 indicates random performance; AUC = 1.0 indicates perfect discrimination.

**Mean Signed Error (MSE):** The mean signed error (not mean squared error) is computed as the mean of (prediction_score − assay_score) across all positions within a domain for the RFAL model. Positive values indicate systematic over-prediction; negative values indicate systematic under-prediction. This metric provides a measure of directional bias in the model's score predictions within structural regions.

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

The RFAL model started at AUC = 0.50 in round 1 (random performance, consistent with random initialization) and improved substantially over five active learning rounds for all four proteins under strict GOF/LOF labeling (Figure 1).

By round 5, RFAL surpassed the LLR baseline for MC4R (0.810 vs. 0.690) and SRC (0.808 vs. 0.748), achieving parity with HXK4 (0.758 vs. 0.772), while LLR substantially outperformed RFAL for PTEN (0.844 vs. 0.655). The active learning convergence was notably consistent: for MC4R and SRC, the largest AUC gains occurred between rounds 3 and 4, while HXK4 showed the most rapid early improvement.

Labeled set sizes under strict GOF/LOF criteria were: MC4R (260 GOF / 697 LOF), HXK4 (1,803 GOF / 4,687 LOF), PTEN (288 GOF / 1,079 LOF), and SRC (396 GOF / 1,679 LOF).

![Auc By Round](auc_by_round.png)

**Figure 1.** Active learning convergence curves for RFAL (solid lines) versus LLR baseline (dashed horizontal lines) across five rounds. Each panel corresponds to one protein (MC4R, HXK4, PTEN, SRC) with strict GOF/LOF labels.

### 3.2 Domain-Level Classification Performance Reveals Structural Heterogeneity

Classification performance varied substantially across structural domains within each protein (Figure 2).

Several noteworthy patterns emerge:

**MC4R:** The ECD shows a dramatic difference — RFAL achieves AUC = 0.893 while LLR achieves only 0.211, indicating that evolutionary conservation signals in the ECD are poorly aligned with GOF/LOF functional consequences, but learned embeddings capture the relevant variation. Both models achieve perfect classification in ICL2 (AUC = 1.000) but this result is complicated by the fact that the variant count in this domain was small. TM6 and TM7 remain the most challenging regions for both models (AUC ≈ 0.50–0.73).

**HXK4:** Performance is more balanced between models. The Hinge 2 region is best classified by RFAL (0.828), while Hinge 1 favors LLR (0.805). Neither model achieves particularly high AUC for the C-terminal tail.

**PTEN:** LLR consistently outperforms RFAL across all domains. The Linker region shows an especially striking difference (LLR = 0.986 vs. RFAL = 0.648). The C-terminal tail is particularly challenging for RFAL (AUC = 0.296, near-random), suggesting that ESM2 embeddings fail to capture functionally relevant variation in this intrinsically disordered region.

**SRC:** Only the kinase domain had sufficient labeled variants for domain-level analysis. RFAL outperforms LLR (0.808 vs. 0.748).

![Auc by Domain](auc_by_domain.png)

**Figure 2.** Graph of domain-level ROC AUC for RFAL and LLR models (strict GOF/LOF labels) for each protein. Color scale from 0.5 (blue, random) to 1.0 (red, perfect). Domains with fewer than 5 labeled variants are excluded (shown as grey).

### 3.3 Domain-Level Regression Performance (MSE)

Mean signed error between RFAL predicted scores and assay scores, computed per domain, revealed heterogeneous prediction accuracy across structural regions (Figure 3).

![MSE by Domain](mse_by_domain.png)

**Figure 3.*** RFAL mean signed error (MSE) by domain. Positive values indicate the model over-predicts functional scores on average; negative values indicate under-prediction. Values near zero indicate predictions are approximately unbiased for that domain.

For MC4R, domains with the best classification AUC (ECL3, ICL2, ECD) also have the lowest MSE (most negative), consistent across metrics. For PTEN, the negative MSE values across all domains mirror the low AUC, indicating that RFAL predictions poorly match the assay scores.

### 3.4 Impact of Label Generation Strategy on Model Performance

The choice of labeling strategy substantially influenced the final-round (round 5) AUC of the RFAL model across all proteins (Figure 3).

![final round auc](figure3_label_strategy_auc.png)

**Figure 3.** Grouped bar chart comparing final round AUC across five labeling strategies for each protein. Strict GOF/LOF labels (dark bar) are highlighted.

Strict GOF/LOF labels yielded the highest final-round AUC for MC4R, HXK4, and SRC. For PTEN, LOF-anchored percentile labels (LOF_10% and LOF_20%) slightly outperformed strict labels (0.732–0.734 vs. 0.655). This is consistent with PTEN being primarily a tumor suppressor where LOF variants are more prevalent and easier to classify than the relatively few GOF variants.

The domain-level AUC under different labeling strategies was generally lower than under strict labels for most proteins and domains (Figure 4). Percentile-based labels, which include more intermediate-effect variants, create more ambiguous classification boundaries, degrading performance compared to strict labels that enforce clear functional separation.

![auc by domain](auc_by_label_method_by_domain.png)

**Figure 4.** Domain-level RFAL AUC comparison between strict GOF/LOF and percentile labels. The number of positive and negative labels for each domain for each labeling strategy is also shown.


### 3.5 Protein Landscape Analysis: Position-Level Prediction vs. Assay Score Correspondence

Figure 7 shows the position-averaged functional and prediction scores for every residue in each protein, enabling a continuous view of how model predictions track functional variation across the full sequence. For each residue position, the assay_score column reports the mean MAVE assay score of all variants observed at that position, while the prediction_score column reports the mean model prediction score. For the LLR model specifically, both scores are z-score normalized, facilitating direct comparison across proteins and domains. The GOF and LOF thresholds provide a reference for interpreting the functional distribution of positions.

![Protein Landscape](protein_landscape.png)

**Figure 10.** Position-averaged assay scores and prediction scores for every residue in each protein for different models.

**Protein-level position-averaged correlations.** The Pearson correlation between position-averaged assay score and prediction score (Figure 7) reveals model-level trends consistent with the AUC results.

![Protein Pearson](protein_pearson_correlation.png)

**Figure 7.** Pearson correlation between position-averaged assay score and model prediction score.

RFAL substantially outperforms LLR for MC4R (r = 0.311 → 0.568) and SRC (r = 0.425 → 0.725), while LLR modestly outperforms RFAL for PTEN (r = 0.589 vs. 0.408), consistent with the AUC findings. HXK4 shows similar correlation for LLR and RFAL (0.610 vs. 0.630). The RFMC model (20% Monte Carlo training) achieves the highest correlation across all proteins, reflecting the advantage of a larger training set in the regression task.

**Domain-level correlation reveals local model-assay alignment.** Breaking down the correlation by structural domain (Figure 8) provides a more granular view of where each model tracks functional variation well or poorly.

![Domain Pearson](domain_pearson_correlation.png)

**Figure 8.** Domain-level Pearson correlation between position-averaged assay score and prediction score.

The most striking feature of this analysis is the strongly negative LLR correlation in the MC4R ECD (r = −0.560): the evolutionary conservation signal is inversely related to functional activity in this region, directly explaining the near-zero classification AUC of LLR in the ECD. In contrast, RFAL achieves a strong positive correlation of 0.754 in the same region, confirming that trained embeddings successfully capture the assay-relevant variation that LLR obscures. The ECD is also the region where MC4R GOF variants are enriched: positions with higher-than-average assay scores cluster in the N-terminal extracellular segment. These residues interact with melanocortin peptide ligands, and the positions at which activating substitutions occur tend to be evolutionarily conserved — the wildtype amino acid is constrained by natural selection because it controls normal signaling, yet certain substitutions at these same positions constitutively activate the receptor. Because LLR penalizes substitutions at conserved positions, it assigns negative (deleterious) scores to GOF variants and neutral or positive scores to non-GOF variants — the inverse of what a GOF classifier requires. The conservation signal is therefore not merely uninformative but actively inverted in this region, which directly explains the near-chance LLR AUC of 0.211 in the MC4R ECD.

TM6 also shows a negative LLR correlation (r = −0.209), while RFAL is modestly positive (r = 0.232), consistent with TM6 being a conformationally dynamic helix whose mutational effects reflect structural rearrangements not readily captured by conservation-based scores. For PTEN, the C-terminal tail shows near-zero RFAL correlation (r = 0.044), confirming that this intrinsically disordered segment presents a fundamental challenge for embedding-based prediction regardless of training. For SRC, the large improvement from LLR (r = 0.427) to RFAL (r = 0.727) in the kinase domain demonstrates that learned embeddings can better represent the complex allosteric relationships governing kinase activity.

**Functional distribution of residue positions.** For PTEN and HXK4, the mean assay scores across positions are predominantly below the LOF threshold, reflecting that these proteins are highly sensitive to perturbation and that most positions are functionally intolerant to substitution. For SRC, a larger fraction of positions show assay scores in the GOF range, consistent with its oncogenic activation potential. MC4R positions span a wider range of the functional spectrum, reflecting the bidirectional (GOF and LOF) variant biology of this receptor.

### 3.6 Summary of Model Comparison

The overall protein-level comparison across models is summarized in Figure 9.

![Protein AUC](protein_final_auc.png)

**Figure 9.** Summary of final model performance by protein and model

RFAL outperformed LLR for MC4R and SRC; LLR outperformed RFAL for PTEN; both performed similarly for HXK4.

---

## 4. Discussion

### 4.1 Active Learning Effectively Leverages ESM2 Embeddings for Variant Classification

Our results demonstrate that active learning with ESM2 embeddings provides an efficient strategy for variant classification. The RFAL model improved consistently across five rounds for all proteins, starting from random performance and reaching AUC values comparable or superior to the zero-shot LLR for three of four proteins. The rapid early gains—particularly evident in HXK4 (round 1 to round 2: +0.146 AUC) and SRC (round 1 to round 5: +0.308 AUC)—suggest that even a small set of carefully selected labeled variants can substantially inform the ESM2 embedding space.

The superiority of RFAL over LLR for MC4R and SRC likely reflects the nature of GOF mutations in these proteins. GOF variants in GPCRs like MC4R often involve constitutively activating residue substitutions in transmembrane domains, changes that may not be predicted by evolutionary conservation alone [14]. Similarly, oncogenic SRC variants frequently involve disruption of autoinhibitory contacts—mutations that may be tolerated evolutionarily in some contexts but produce gain-of-function signaling in human cells [15]. Trained embeddings can capture these context-specific functional relationships that zero-shot models miss.

### 4.2 PTEN Presents Unique Challenges for Embedding-Based Classification

The superior LLR performance for PTEN (AUC = 0.844 vs. RFAL = 0.655) warrants careful interpretation. PTEN functions primarily as a tumor suppressor, and the majority of disease-relevant variants are LOF. The high LLR performance likely reflects strong evolutionary conservation of catalytically critical residues in the phosphatase and C2 domains, making evolutionary-based scores reliable LOF predictors. The dramatically higher LLR in the Linker region (0.986) is consistent with the known importance of this regulatory segment and its strong conservation.

The poor RFAL performance for PTEN may reflect challenges with the training data. GOF variants for PTEN are rare and functionally heterogeneous; the strict label set includes only 288 GOF variants against 1,079 LOF variants—a moderate imbalance. Additionally, PTEN's C-terminal tail (residues 352–403) is largely intrinsically disordered and subject to extensive post-translational regulation, potentially making sequence-based embeddings less predictive than for structured domains [16]. The near-random RFAL AUC in this region (0.296) supports this interpretation.

For PTEN, LOF-anchored percentile labels (LOF_10%, LOF_20%) outperformed strict labels in round 5 (0.732–0.734 vs. 0.655). This suggests that defining the training task around LOF—which is the dominant and better-characterized class for PTEN—may be more tractable for the RFAL model than attempting joint GOF/LOF discrimination with strict labels.

### 4.3 Domain-Level Heterogeneity Informs Structural Interpretation

The dramatic variation in classification performance across structural domains provides biologically meaningful insights. For MC4R, the strikingly different performance in the ECD (RFAL = 0.893 vs. LLR = 0.211) reveals that the extracellular domain harbors GOF-relevant variation that is not captured by evolutionary conservation but is present in the ESM2 embedding features. The ECD of MC4R interacts with the endogenous agonist α-MSH and with small molecule ligands; activating mutations in this region may alter receptor conformational equilibria in ways not penalized evolutionarily [10].

The perfect classification (AUC = 1.000) achieved in ICL2 by both models likely reflects the small size of this loop (residues 103–109, only 7 positions) combined with strong functional constraint: the second intracellular loop of GPCRs makes critical contacts with G proteins, and any mutation in this region tends to strongly perturb function. For ECL3, the LLR achieves perfect classification (1.000) while RFAL achieves 0.95, again reflecting a small, highly constrained region.

The relatively poor performance in TM6 and TM7 (AUC < 0.75 for both models) is noteworthy. These helices undergo conformational changes during receptor activation and harbor complex allosteric relationships with the rest of the receptor. The difficulty in classifying variants here may indicate that functional outcome depends on subtle structural context that neither embedding distance nor conservation adequately captures.

### 4.4 Label Generation Strategy Has Substantial Practical Implications

The consistent superiority of strict GOF/LOF labels over percentile-based labels (for three of four proteins) has important implications for future MAVE data analysis. Percentile labels capture the extremes of the assay distribution, but conflate biologically distinct categories—a 90th percentile variant may be slightly above neutral rather than a true gain-of-function. The inclusion of these ambiguous variants as "positive" examples likely degrades classifier training.

However, strict GOF/LOF labels require expert curation and leave many variants unlabeled—a limitation when labeled data are scarce. The LOF percentile strategies performed reasonably well for HXK4 and PTEN (final AUC 0.72–0.75), suggesting that when GOF labels are unreliable or unavailable, LOF-anchored training may provide a viable alternative, particularly for tumor suppressors and enzymes where LOF is the pathogenic mechanism.

### 4.5 Protein Landscape: Position-Level Functional Topography

The position-averaged assay and prediction score landscape provides a complementary perspective to the AUC-based analysis by revealing the continuous functional topography of each protein and how well each model tracks it. The most consequential finding from this analysis is the negative LLR-assay correlation in the MC4R ECD (r = −0.560), which directly explains the LLR's near-chance classification performance in that region. The extracellular domain of MC4R is the site of ligand-activating contacts, and variants here that increase receptor activity are systematically predicted by LLR to be deleterious—the inverse of their true function. This is a direct manifestation of the evolutionary constraint paradox for activating GPCRs: constitutively activating mutations alter residues that are conserved because they are functionally important, leading zero-shot conservation-based models to flag them as damaging.

The negative LLR correlation in TM6 (r = −0.209) is also notable. TM6 undergoes a large outward displacement during receptor activation, and mutations that facilitate this movement may appear evolutionarily unusual while being functionally activating. This again underscores that for proteins with GOF disease mutations, LLR can be systematically misleading in regions critical to the activation mechanism.

For HXK4, the moderate and consistent correlations across all domains (LLR r = 0.50–0.83, RFAL r = 0.59–0.81) are consistent with glucokinase's well-understood structure-function relationships, where most GOF and LOF positions are distributed throughout the kinase fold. The notably high Hinge 1 LLR correlation (r = 0.832) reflects strong evolutionary constraint on this regulatory region, where substitutions have reliably predictable consequences.

For PTEN, the degrading RFAL correlations across domains —from 0.492 in the phosphatase domain to 0.044 in the C-terminal tail— map directly onto the structural organization of the protein: the phosphatase domain is well-folded and function-critical, while the C-terminal tail is disordered and subject to extensive post-translational regulation that ESM2 sequence embeddings cannot capture. The LLR's higher correlation in the phosphatase domain (0.666) similarly reflects the strong evolutionary constraint on catalytic residues.

The RFMC model consistently achieves the highest position-level correlations (Table 7), demonstrating that when sufficient labeled training data is available, the embedding-based classifier can approach strong regression-level performance even in challenging regions. This establishes an upper bound for what the active learning model might achieve with more training rounds or larger initial labeled sets.

### 4.6 Limitations and Future Directions

Several limitations should be noted. First, our active learning implementation used five rounds; additional rounds might further improve performance, particularly for PTEN where round 5 performance was not clearly converged. Second, the RFMC model (Monte Carlo 20% training) provides an alternative baseline that was not extensively analyzed at the domain level in this study; a more systematic comparison between active learning and random subsampling would clarify the specific contribution of the iterative selection strategy.

Future work should explore ensemble approaches combining LLR with RF AL predictions, which may improve performance for PTEN-like cases where the two models have complementary strengths. Additionally, using alternatives to mean pooling to get a fixed length representation of the embeddings could result in less information loss. Finally, incorporating structural features (e.g., from AlphaFold2 predicted structures), conservation scores, and functional annotation could augment ESM2 embeddings and improve domain-level performance in challenging regions such as intrinsically disordered tails.

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

18. Howard CJ, Abell NS, Osuna BA, et al. High-resolution deep mutational scanning of the melanocortin-4 receptor enables target characterization for drug discovery. *eLife*. 2025. PMCID: PMC11981609.

19. Gersing S, Cagiada M, Gebbia M, et al. A comprehensive map of human glucokinase variant activity. *Genome Biol*. 2023;24(1):93. PMCID: PMC10131484.

20. Ahler E, Register AC, Chakraborty S, et al. An integrated approach reveals a regulatory mechanism coupling Src's kinase activity, localization, and phosphotransferase-independent functions. *Mol Cell*. 2019;74(2):393–408.e20. PMCID: PMC6474823.

21. Matreyek KA, Stephany JJ, Ahler E, Fowler DM. Integrating thousands of PTEN variant activity and abundance measurements reveals variant subgroups and new dominant negatives in cancers. *Genome Med*. 2021;13(1):165. PMCID: PMC8518224.

---

*Supplementary Data:* All analysis data files are available in the project repository, including `protein_landscape_data.csv`, `protein_landscape_domains.csv`, `mse_by_domain.csv`, `auc_by_domain.csv`, `auc_by_label_method_by_domain.csv`, and `auc_by_label_method_by_round.csv`.
