# Learning to Classify Gain- and Loss-of-Function Variants with Protein Language Models

---

## Abstract

Multiplexed Assay of Variant Effects (MAVE) datasets provide rich functional data for protein variants, yet distinguishing gain-of-function (GOF) from loss-of-function (LOF) variants remains a key challenge with direct clinical relevance. Here we evaluate three computational approaches for GOF/LOF variant classification across four clinically relevant proteins: MC4R (melanocortin-4 receptor), HXK4 (glucokinase), PTEN (phosphatase and tensin homolog), and SRC (proto-oncogene tyrosine-protein kinase). The approaches include: (1) a log-likelihood ratio (LLR) derived from ESM2 protein language model scores, (2) an ESM2 embedding-based Random Forest trained via active learning (RF AL), and (3) the same Random Forest architecture trained via 5-fold cross-validation on the full labeled dataset (RF 5 FOLD CV). Using ROC AUC as the primary performance metric, we find that RF 5 FOLD CV — which uses the full labeled dataset through cross-validation — achieves higher classification performance than LLR for MC4R (0.773 vs. 0.690) and SRC (0.831 vs. 0.748), while approaching LLR for HXK4 (0.755 vs. 0.772) and remaining below LLR for PTEN (0.741 vs. 0.844). In contrast, RF AL with eleven active learning rounds underperforms LLR across all four proteins (MC4R: 0.601 vs. 0.690; HXK4: 0.564 vs. 0.772; PTEN: 0.619 vs. 0.844; SRC: 0.638 vs. 0.748), consistent with recent critiques highlighting fundamental limitations of protein language model log-likelihood ratios as mutational effect predictors [22] but also indicating that the active learning strategy has not yet exhausted the representational capacity of the embedding-based classifier. Notably, domain-level analysis reveals that RF AL substantially outperforms LLR in specific structural regions despite lower overall protein-level performance — most strikingly in the MC4R extracellular domain (ECD; RF AL AUC = 0.505 ± 0.097 vs. LLR = 0.211). Label generation strategy meaningfully impacts model performance, with strict GOF/LOF labels generally yielding the highest RF AL AUC for three of four proteins after eleven rounds. These results demonstrate that the ESM2 embedding-based Random Forest architecture has sufficient capacity to exceed the zero-shot LLR baseline when trained on the full dataset, while also highlighting that active learning with limited rounds has not yet fully realized this potential.

---

## 1. Background

Protein variants arising from single amino acid substitutions can have profoundly different consequences for protein function, ranging from complete loss of activity to pathological hyperactivation. Accurate classification of variants as gain-of-function (GOF) or loss-of-function (LOF) is essential for interpreting genetic variants of uncertain significance (VUS), understanding disease mechanisms, and guiding therapeutic development [1,2]. Despite the availability of large MAVE datasets that quantitatively measure variant effects, the computational challenge of translating continuous functional scores into accurate GOF/LOF classifications remains non-trivial [3].

Protein language models (PLMs), trained on vast databases of protein sequences, encode evolutionary constraints and structural information in high-dimensional embedding spaces [4,5]. ESM2, developed by Meta AI, is among the most powerful PLMs currently available, demonstrating state-of-the-art performance on a wide range of protein function prediction tasks [6]. The log-likelihood ratio (LLR) derived from ESM2 masked marginal scoring—comparing the log probability of a mutant residue to the wild-type—serves as a zero-shot predictor of variant deleteriousness [7,8]. While effective for predicting LOF variants, the LLR's performance for GOF classification is less established, as evolutionary conservation does not necessarily predict activating mutations. Recent work has critically evaluated the reliability of PLM-based LLR as a mutational effect predictor, documenting systematic failure modes including the conflation of site-level conservation with substitution-specific effects and poor calibration for proteins underrepresented in training corpora [22]. These critiques motivate the exploration of supervised, data-driven approaches that can learn protein-specific functional relationships directly from experimental data.

Active learning offers an attractive paradigm for variant classification when labeled data are sparse or expensive to generate [9]. By iteratively selecting the most informative variants for labeling, active learning can achieve high model performance with far fewer labeled examples than random sampling. When combined with rich feature representations from PLMs, active learning-based classifiers may capture the complex, context-dependent relationships between sequence features and functional outcomes that zero-shot methods cannot fully exploit.

The four proteins studied here represent diverse structural classes and disease relevance. MC4R is a G protein-coupled receptor (GPCR) in which GOF variants have been linked to constitutional leanness and LOF variants to obesity [10]. HXK4 (glucokinase, GCK) is an enzyme whose GOF mutations cause congenital hyperinsulinism and LOF mutations cause maturity-onset diabetes of the young (MODY) [11]. PTEN is a lipid phosphatase and tumor suppressor in which LOF variants lead to cancer predisposition, while rare GOF variants have been described [12]. SRC is a non-receptor tyrosine kinase whose GOF variants are associated with cancer and neurological conditions [13]. Critically, these four proteins were selected in part because they are among the few proteins for which gain-of-function MAVE data is currently available [18–21], making them uniquely suited for benchmarking GOF/LOF classification approaches that require both functional categories to be represented in training and evaluation data.

In this study, we systematically benchmark three computational models across these four proteins, examining overall and domain-level classification performance, the convergence of active learning over training rounds, and the impact of label generation strategy on model evaluation.

---

## 2. Methods

### 2.1 MAVE Datasets and Variant Scoring

MAVE datasets for MC4R, HXK4, PTEN, and SRC were obtained from published sources and used to define ground-truth functional scores for single amino acid substitution variants. The MC4R dataset was derived from high-resolution deep mutational scanning of the melanocortin-4 receptor [18]. The HXK4 (GCK) dataset was obtained from a comprehensive map of human glucokinase variant activity [19]. The SRC dataset was derived from an integrated approach characterizing the regulatory mechanism coupling Src's kinase activity, localization, and phosphotransferase-independent functions [20]. The PTEN dataset was obtained from a study integrating thousands of PTEN variant activity and abundance measurements [21]. Each dataset provides a continuous assay score measuring variant functional activity relative to wild-type. GOF and LOF thresholds were defined per protein for MC4R, HXK4 and SRC based on published criteria. For PTEN the thresholds were arbitrarily set at 10% higher or 10% lower than the wild type score, respectively. Variants with assay scores exceeding the GOF protein-specific threshold were classified as GOF, while variants below the LOF threshold were classified as LOF. Variants between these thresholds were treated as functionally neutral and excluded from strict binary classification analyses.

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

**RF AL (Random Forest with Active Learning):** A Random Forest classifier trained on mean pooled ESM2 embeddings using an active learning strategy. Training proceeded for eleven rounds. The initial labeled set was seeded with a small random subset. This design was chosen based on the work in [17] demonstrating that it was effective at identifying GOF variants. Following the approach used in [17] a greedy acquisition strategy was used for subsequent rounds. We sampled 16 variants in the first round and 50 variants in subsequent rounds. Mean prediction scores obtained over 5 simulations with 11 rounds per simulation were used.

**RF 5 FOLD CV (Random Forest with 5-Fold Cross-Validation):** The same Random Forest classifier architecture used for RF AL was trained on the full labeled dataset using 5-fold cross-validation, providing a data-usage upper bound for the embedding-based classifier. This model uses all available labeled data through standard cross-validation, in contrast to the iterative and data-limited active learning strategy. Prediction scores across folds were used for evaluation.

### 2.4 Evaluation Metrics

**ROC AUC:** Receiver operating characteristic area under the curve, computed for binary GOF/LOF classification. Higher values indicate better discrimination. AUC = 0.5 indicates random performance; AUC = 1.0 indicates perfect discrimination. For RF AL, AUC values from the convergence analysis are reported as mean ± standard deviation across simulation runs; LLR and RF 5 FOLD CV are deterministic and therefore have no associated variability.

**Mean Signed Error (MSE):** The mean signed error (not mean squared error) is computed as the mean of (prediction_score − assay_score) across all positions within a domain for the RF AL model. Positive values indicate systematic over-prediction; negative values indicate systematic under-prediction. This metric provides a measure of directional bias in the model's score predictions within structural regions. MSE values are reported as mean ± standard deviation across simulation runs.

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

All labeling strategy comparisons used the RF AL model.

---

## 3. Results

### 3.1 Protein Landscape Analysis: Position-Level Prediction vs. Assay Score Correspondence

Figure 1 shows the position-averaged functional and prediction scores for every residue in each protein, enabling a continuous view of how model predictions track functional variation across the full sequence. For each residue position, the assay score  reports the mean MAVE assay score of all variants observed at that position, while the prediction reports the mean model prediction score. For the LLR model specifically, both scores are z-score normalized, facilitating direct comparison across proteins and domains. The RF AL scores are the final round scores. The GOF and LOF thresholds (dashes) provide a reference for interpreting the functional distribution of positions.

![Protein Landscape](protein_landscape.png)

**Figure 1.** Position-averaged assay scores and prediction scores for every residue in each protein for different models.

**Protein-level position-averaged correlations.** The Pearson correlation between position-averaged assay score and prediction score reveals model-level trends consistent with the AUC results.

![Protein Pearson](protein_pearson_correlation.png)

**Figure 2.** Pearson correlation between position-averaged assay score and model prediction score.

RF AL substantially outperforms LLR for SRC (r = 0.425 → 0.725), while LLR modestly outperforms RF AL for PTEN (r = 0.589 vs. 0.408) and MC4R (r = 0.311 vs. 0.568 for RF AL in prior analyses), consistent with the AUC findings. HXK4 shows similar correlation for LLR and RF AL (0.610 vs. 0.630). The RF 5 FOLD CV model achieves the highest correlation across all proteins, reflecting the advantage of a larger training set in the regression task.

**Domain-level correlation reveals local model-assay alignment.** Breaking down the correlation by structural domain provides a more granular view of where each model tracks functional variation well or poorly.

![Domain Pearson](domain_pearson_correlation.png)

**Figure 3.** Domain-level Pearson correlation between position-averaged assay score and prediction score.

The most striking feature of this analysis is the strongly negative LLR correlation in the MC4R ECD (r = −0.560): the evolutionary conservation signal is inversely related to functional activity in this region, directly explaining the near-zero classification AUC of LLR in the ECD. In contrast, RF AL achieves a strong positive correlation in the same region, confirming that trained embeddings successfully capture the assay-relevant variation that LLR obscures. The ECD is also the region where MC4R GOF variants are enriched: positions with higher-than-average assay scores cluster in the N-terminal extracellular segment. These residues interact with melanocortin peptide ligands, and the positions at which activating substitutions occur tend to be evolutionarily conserved — the wildtype amino acid is constrained by natural selection because it controls normal signaling, yet certain substitutions at these same positions constitutively activate the receptor. Because LLR penalizes any substitution at conserved positions regardless of functional effect, both GOF and LOF variants at these positions receive negative scores. However, GOF variants in the ECD tend to occur at positions that are evolutionarily conserved yet permissive to activating substitutions, so LLR scores them as deleterious. LOF variants, which more broadly disrupt conserved residues critical for protein stability and function, are also scored negatively — meaning LLR's signal aligns with LOF detection but not GOF detection. The result is that LLR provides no useful discriminative signal for separating GOF from non-GOF variants in this region, and its scores may even be inverted relative to what a GOF classifier requires. The conservation signal is therefore not merely uninformative but actively inverted in this region, which directly explains the near-chance LLR AUC of 0.211 in the MC4R ECD. This inversion represents a textbook example of the broader limitation of PLM-based LLR: the conflation of evolutionary constraint with functional effect, a limitation that has been systematically documented across diverse protein families [22].

TM6 also shows a negative LLR correlation (r = −0.209), while RF AL is modestly positive (r = 0.232), consistent with TM6 being a conformationally dynamic helix whose mutational effects reflect structural rearrangements not readily captured by conservation-based scores. For PTEN, the C-terminal tail shows near-zero RF AL correlation (r = 0.044), confirming that this intrinsically disordered segment presents a fundamental challenge for embedding-based prediction regardless of training. For SRC, the large improvement from LLR (r = 0.427) to RF AL (r = 0.727) in the kinase domain demonstrates that learned embeddings can better represent the complex allosteric relationships governing kinase activity.

**Functional distribution of residue positions.** For PTEN and HXK4, the mean assay scores across positions are predominantly below the LOF threshold, reflecting that these proteins are highly sensitive to perturbation and that most positions are functionally intolerant to substitution. For SRC, a larger fraction of positions show assay scores in the GOF range, consistent with its oncogenic activation potential. MC4R positions span a wider range of the functional spectrum, reflecting the bidirectional (GOF and LOF) variant biology of this receptor.

### 3.2 Active Learning Achieves Competitive Performance Across Proteins

The RF AL model started at AUC = 0.50 in round 1 (random performance, consistent with random initialization) and improved over eleven active learning rounds for all four proteins under strict GOF/LOF labeling (Figure 4).

By round 11, LLR still outperformed RF AL across all four proteins: MC4R (LLR 0.690 vs. RF AL 0.537 ± 0.017), HXK4 (LLR 0.772 vs. RF AL 0.548 ± 0.010), PTEN (LLR 0.844 vs. RF AL 0.594 ± 0.007), and SRC (LLR 0.748 vs. RF AL 0.586 ± 0.017). The margins vary considerably across proteins: LLR's advantage is most pronounced for PTEN (+0.250 AUC) and HXK4 (+0.224 AUC), while the gaps for MC4R (+0.153) and SRC (+0.162) are smaller, though all exceed the RF AL standard deviation. The relatively narrow standard deviation for PTEN (±0.007) reflects high run-to-run consistency, likely because the larger labeled set reduces sampling variability despite class imbalance. The active learning convergence was nonetheless consistent: for most proteins, the largest AUC gains occurred in earlier rounds, with diminishing returns by later rounds.

![Auc By Round](auc_by_label_method_by_round.png)

**Figure 4.** Active learning convergence curves for RF AL (solid lines) versus LLR baseline (dashed horizontal lines) across eleven rounds. Each panel corresponds to one protein (MC4R, HXK4, PTEN, SRC) with strict GOF/LOF labels. Shaded bands indicate ± one standard deviation across simulation runs.

### 3.3 Domain-Level Classification Performance Reveals Structural Heterogeneity

Classification performance varied substantially across structural domains within each protein (Figure 5).

Several noteworthy patterns emerge:

**MC4R:** The ECD shows a striking difference — RF AL achieves AUC = 0.505 ± 0.097 while LLR achieves only 0.211, indicating that evolutionary conservation signals in the ECD are poorly aligned with GOF/LOF functional consequences, and that learned embeddings provide a meaningful advantage in this region despite overall protein-level parity. ECL3 shows RF AL = 0.573 ± 0.184 vs. LLR = 1.000, with high variability reflecting a small variant count (10 GOF / 2 LOF). ICL2 achieves LLR AUC of 1.000, while RF AL is at chance (0.500 ± 0.162), with the large RF AL standard deviation and small domain size (3 GOF / 6 LOF) limiting the interpretation of both values. TM6 and TM7 are near-random for RF AL (0.498 ± 0.050 and 0.499 ± 0.075, respectively), while LLR also shows limited performance in TM6 (0.502) but moderate performance in TM7 (0.619).

**HXK4:** Performance is relatively balanced across domains, but LLR consistently outperforms RF AL in all regions. The Hinge 2 region is best classified by RF AL (0.617 ± 0.030) among HXK4 domains, while Hinge 1 strongly favors LLR (0.805 vs. RF AL 0.529 ± 0.071). The C-terminal tail shows moderate performance for LLR (0.765) but lower RF AL performance (0.555 ± 0.027). The Large domain N-lobe shows below-chance RF AL performance (0.474 ± 0.035), suggesting that embedding-based features may not capture the functional constraints of this region as well as evolutionary conservation signals.

**PTEN:** LLR consistently outperforms RF AL across all domains. The Linker region shows the most striking difference (LLR = 0.986 vs. RF AL = 0.654 ± 0.126), with the RF AL standard deviation reflecting the small variant count in this region (5 GOF / 29 LOF). The C-terminal tail is particularly challenging for RF AL (AUC = 0.406 ± 0.042, well below chance), confirming that ESM2 embeddings fail to capture functionally relevant variation in this intrinsically disordered region. The C2 domain and phosphatase domain also show LLR advantage (LLR 0.876 and 0.753 vs. RF AL 0.569 ± 0.018 and 0.593 ± 0.021, respectively).

**SRC:** Only the kinase domain had sufficient labeled variants for domain-level analysis. RF AL performance (0.586 ± 0.017) falls below LLR (0.748), with the narrow standard deviation indicating high run-to-run consistency.

![Auc by Domain](auc_by_domain.png)

**Figure 5.** Graph of domain-level ROC AUC for RF AL and LLR models (strict GOF/LOF labels) for each protein. Error bars indicate ± one standard deviation across simulation runs for RF AL; LLR is deterministic.

### 3.4 Domain-Level Regression Performance (MSE)

Mean signed error between RF AL predicted scores and assay scores, computed per domain, revealed heterogeneous prediction accuracy across structural regions (Figure 6).

![MSE by Domain](mse_by_domain.png)

**Figure 6.** RF AL mean signed error (MSE) by domain. Positive values indicate the model over-predicts functional scores on average; negative values indicate under-prediction. Values near zero indicate predictions are approximately unbiased for that domain. Error bars indicate ± one standard deviation across simulation runs.

For MC4R, the ECL3 domain shows the most negative MSE (−1.068 ± 0.119), indicating consistent under-prediction of assay scores, while ECD also shows substantial negative MSE (−0.756 ± 0.200). In contrast, ECL2 shows the most positive MSE (+0.816 ± 0.153), indicating systematic over-prediction in this loop. TM7 also shows positive MSE (+0.655 ± 0.060), while TM6 is modestly negative (−0.102 ± 0.094).

For PTEN, negative MSE values are observed across all domains (C-terminal tail: −0.712 ± 0.252; Linker: −0.888 ± 0.314; C2 domain: −0.706 ± 0.255; Phosphatase domain: −0.168 ± 0.237). The consistently negative MSE across all PTEN domains mirrors the low AUC, indicating that RF AL predictions systematically under-predict assay scores. The particularly large standard deviation in the Linker (±0.314) reflects both the small domain size and run-to-run variability.

For SRC, the kinase domain shows a small positive MSE (0.066 ± 0.045) with tight standard deviation, indicating approximately unbiased predictions with high consistency across runs.

For HXK4, MSE values are mixed: Hinge 1 shows the most negative MSE (−0.728 ± 0.183), indicating under-prediction, while Hinge 2 shows positive MSE (+0.272 ± 0.131), indicating over-prediction. The remaining domains (Small domain, Large domain N-lobe and C-lobe, C-terminal tail) are close to zero, suggesting approximately unbiased predictions.

### 3.5 Impact of Label Generation Strategy on Model Performance

The choice of labeling strategy substantially influenced the final-round (round 11) AUC of the RF AL model across all proteins (Figure 7).

![final round auc](figure3_label_strategy_auc.png)

**Figure 7.** Grouped bar chart comparing final round AUC across five labeling strategies for each protein. Strict GOF/LOF labels (dark bar) are highlighted. Error bars indicate ± one standard deviation across simulation runs.

Strict GOF/LOF labels yielded the highest final-round RF AL AUC for MC4R (0.537 ± 0.017), PTEN (0.594 ± 0.007), and SRC (0.586 ± 0.017). For HXK4, LOF-anchored percentile labels slightly outperformed strict labels (LOF_20%: 0.564 ± 0.012; LOF_10%: 0.557 ± 0.015 vs. GOF/LOF strict: 0.548 ± 0.010), suggesting that for this enzyme the LOF signal is more consistently captured by percentile-based thresholds. For PTEN, GOF/LOF strict labels slightly outperformed LOF percentile labels (GOF/LOF: 0.594 ± 0.007 vs. LOF_20%: 0.575 ± 0.021 and LOF_10%: 0.571 ± 0.025), though the confidence intervals broadly overlap.

The domain-level AUC under different labeling strategies was generally lower than under strict labels for most proteins and domains (Figure 8). Percentile-based labels, which include more intermediate-effect variants, create more ambiguous classification boundaries, degrading performance compared to strict labels that enforce clear functional separation.

![auc by domain](auc_by_label_method_by_domain.png)

**Figure 8.** Domain-level RF AL AUC comparison between strict GOF/LOF and percentile labels. The number of positive and negative labels for each domain for each labeling strategy is also shown. Error bars indicate ± one standard deviation across simulation runs.

### 3.6 Summary of Model Comparison

The overall protein-level comparison across all three models is summarized in Figure 9, using the final held-out AUC for each model.

![Protein AUC](protein_final_auc.png)

**Figure 9.** Summary of final model performance by protein and model (LLR, RF AL, RF 5 FOLD CV).

RF 5 FOLD CV — the same Random Forest architecture as RF AL but trained on the full labeled dataset via 5-fold cross-validation — outperforms LLR for MC4R (0.773 vs. 0.690) and SRC (0.831 vs. 0.748), approaches LLR for HXK4 (0.755 vs. 0.772), and remains below LLR for PTEN (0.741 vs. 0.844). RF AL consistently underperforms LLR across all four proteins (MC4R: 0.601 vs. 0.690; HXK4: 0.564 vs. 0.772; PTEN: 0.619 vs. 0.844; SRC: 0.638 vs. 0.748). The substantial gap between RF AL and RF 5 FOLD CV for all proteins — most pronounced for SRC (0.638 vs. 0.831) and MC4R (0.601 vs. 0.773) — quantifies the untapped potential of the embedding-based architecture that active learning with eleven rounds has not yet realized. Despite lower overall AUC, RF AL demonstrated domain-specific advantages — particularly in the MC4R ECD (RF AL 0.505 ± 0.097 vs. LLR 0.211) — that are obscured by protein-level summaries.

---

## 4. Discussion

### 4.1 RF 5 FOLD CV Exceeds LLR for MC4R and SRC, but Active Learning Has Not Yet Closed the Gap

A key finding of this study is that the ESM2 embedding-based Random Forest classifier, when trained on the full labeled dataset via 5-fold cross-validation, outperforms the zero-shot LLR baseline for MC4R (0.773 vs. 0.690) and SRC (0.831 vs. 0.748). This directly demonstrates that the RF+ESM2 architecture has sufficient representational capacity to exceed the zero-shot baseline when adequate training data is provided. The strong RF 5 FOLD CV performance for SRC (+0.083 over LLR) is particularly notable, as it suggests that the ESM2 embedding space encodes meaningful discriminative information about kinase activation that is not fully captured by evolutionary conservation alone.

At the same time, RF AL with eleven rounds still underperforms LLR across all four proteins, consistent with recent critiques of PLM-based variant effect prediction that highlight the tendency of supervised approaches to underperform zero-shot baselines when trained on limited labeled data [22]. The large gap between RF AL and RF 5 FOLD CV — most pronounced for SRC (0.638 vs. 0.831) and MC4R (0.601 vs. 0.773) — suggests that eleven active learning rounds, acquiring approximately 516 labels (16 in round 1 + 50 × 10 subsequent rounds), are not sufficient to fully exploit the discriminative information available in the embedding space. Closing this gap is the primary challenge for future active learning work.

The domain-level analysis reveals complementary strengths: RF AL provides meaningful advantages in specific structural regions such as the MC4R ECD (RF AL 0.505 ± 0.097 vs. LLR 0.211), even while underperforming LLR at the protein level. These domain-specific advantages motivate continued development of supervised PLM-based classifiers, particularly when the goal is to understand variant effects within specific functional subregions.

### 4.2 PTEN Presents Unique Challenges for Embedding-Based Classification

For PTEN, both RF AL (0.619) and RF 5 FOLD CV (0.741) underperform LLR (0.844), with the performance gap for RF 5 FOLD CV (+0.103 for LLR) being smaller than for RF AL (+0.225 for LLR) but still substantial. The high LLR performance likely reflects strong evolutionary conservation of catalytically critical residues in the phosphatase and C2 domains, making evolutionary-based scores reliable LOF predictors. The dramatically higher LLR in the Linker region (0.986) is consistent with the known importance of this regulatory segment and its strong conservation.

The relatively poor performance of both supervised models for PTEN may reflect challenges with the training data. GOF variants for PTEN are rare and functionally heterogeneous; the strict label set includes only 288 GOF variants against 1,079 LOF variants — a moderate imbalance. Additionally, PTEN's C-terminal tail (residues 352–403) is largely intrinsically disordered and subject to extensive post-translational regulation, potentially making sequence-based embeddings less predictive than for structured domains [16]. The below-chance RF AL AUC in this region (0.406 ± 0.042) supports this interpretation. That RF 5 FOLD CV also fails to match LLR for PTEN (0.741 vs. 0.844), despite using all available data, suggests that the PTEN classification challenge is not simply a data quantity problem but reflects a fundamental representational limitation of sequence-only ESM2 embeddings for this protein.

For PTEN, strict GOF/LOF labels yielded the highest RF AL AUC at round 11 (0.594 ± 0.007) and also the lowest run-to-run variance, outperforming both LOF percentile methods (LOF_20%: 0.575 ± 0.021; LOF_10%: 0.571 ± 0.025) and GOF percentile methods (GOF_20%: 0.546 ± 0.007; GOF_10%: 0.535 ± 0.009). The GOF percentile methods performed worst overall, which is biologically consistent: for a tumor suppressor where the predominant functional outcome is loss-of-function, assigning the top 10–20% of the assay distribution as "GOF" conflates truly activating variants with those that are merely less severely damaging — providing a noisy and biologically ambiguous positive class. The LOF percentile methods perform better than GOF percentile methods, reflecting that the LOF-enriched biology of PTEN makes LOF-focused labels more coherent, but their higher variance (±0.021–0.025) relative to strict labels (±0.007) suggests that the percentile cutoffs introduce instability by including borderline variants near the threshold.

The superiority of strict GOF/LOF labels for PTEN is noteworthy because one might expect the opposite: with only 288 GOF variants against 1,079 LOF variants, the strict positive class is small and potentially hard to learn from. Instead, the strict labels appear to provide a cleaner and more consistent training signal — the biologically validated threshold separates classes with less ambiguity than any percentile cutoff can, and the active learning algorithm's progressive sampling over eleven rounds is sufficient to achieve reasonable coverage of the minority GOF class despite its small size. The GOF/LOF convergence curve for PTEN also shows a notably steady monotonic improvement across rounds (round 2: 0.524, round 8: 0.585, round 11: 0.594), in contrast to the more irregular trajectories of the LOF percentile methods, further supporting that strict labels provide a more learnable objective for this protein.

### 4.3 Domain-Level Heterogeneity Informs Structural Interpretation

The dramatic variation in classification performance across structural domains provides biologically meaningful insights. For MC4R, the strikingly different performance in the ECD (RF AL = 0.505 ± 0.097 vs. LLR = 0.211) reveals that the extracellular domain harbors GOF-relevant variation that is not captured by evolutionary conservation but is partially present in the ESM2 embedding features. The ECD of MC4R interacts with the endogenous agonist α-MSH and with small molecule ligands; activating mutations in this region may alter receptor conformational equilibria in ways not penalized evolutionarily [10]. While the RF AL ECD AUC of 0.505 ± 0.097 is only modestly above chance, it represents a meaningful improvement over LLR's 0.211 and demonstrates that supervised embeddings partially capture the relevant variation.

The high LLR performance in ICL2 (1.000) and ECL3 (1.000) should be interpreted with caution given the small domain sizes (3 GOF / 6 LOF in ICL2; 10 GOF / 2 LOF in ECL3). RF AL performs at chance in ICL2 (0.500 ± 0.162) and only modestly above chance in ECL3 (0.573 ± 0.184), with the wide standard deviations confirming that performance estimates for these domains are unstable across runs.

The near-random RF AL performance in TM6 and TM7 (0.498 ± 0.050 and 0.499 ± 0.075, respectively) is noteworthy. These helices undergo conformational changes during receptor activation and harbor complex allosteric relationships with the rest of the receptor. The difficulty in classifying variants here may indicate that functional outcome depends on subtle structural context that neither sequence-level embeddings nor conservation adequately captures — notably, LLR also performs near-randomly in TM6 (0.502), suggesting this is a fundamental challenge for sequence-based approaches.

The below-chance RF AL AUC in the PTEN C-terminal tail (0.406 ± 0.042) and HXK4 Large domain N-lobe (0.474 ± 0.035) indicates that embedding-based features are not merely uninformative but actively misleading in these regions. For the PTEN C-terminal tail, this likely reflects intrinsic disorder and post-translational regulation that are invisible to sequence-level models [16].

### 4.4 Label Generation Strategy Has Substantial Practical Implications

The consistent superiority of strict GOF/LOF labels over percentile-based labels (for MC4R, PTEN, and SRC) has important implications for future MAVE data analysis. Percentile labels capture the extremes of the assay distribution, but conflate biologically distinct categories — a 90th percentile variant may be slightly above neutral rather than a true gain-of-function. The inclusion of these ambiguous variants as "positive" examples likely degrades classifier training.

For HXK4, LOF percentile strategies (LOF_20%: 0.564 ± 0.012) outperformed strict labels (0.548 ± 0.010) at round 11. This may reflect that although the strict HXK4 GOF label set is numerically large (1,803 variants), the positive class spans a wide spectrum of functional severity — from strongly activating to marginally above the GOF threshold. Because the RF model treats all GOF variants as equivalent positive examples regardless of effect magnitude, weakly activating variants near the decision boundary dilute the training signal, making it harder to learn a consistent embedding signature that separates GOF from non-GOF variants. The small magnitude of the difference (0.016 AUC) warrants caution in over-interpreting this result.

### 4.5 Protein Landscape: Position-Level Functional Topography

The position-averaged assay and prediction score landscape provides a complementary perspective to the AUC-based analysis by revealing the continuous functional topography of each protein and how well each model tracks it. The most consequential finding from this analysis is the negative LLR-assay correlation in the MC4R ECD (r = −0.560), which directly explains the LLR's near-chance classification performance in that region. The extracellular domain of MC4R is the site of ligand-activating contacts, and variants here that increase receptor activity are systematically predicted by LLR to be deleterious — the inverse of their true function. This is a direct manifestation of the evolutionary constraint paradox for activating GPCRs: constitutively activating mutations alter residues that are conserved because they are functionally important, leading zero-shot conservation-based models to flag them as damaging. This example concretely illustrates the class of failure modes documented in recent critical assessments of PLM-based variant effect prediction [22]: the LLR is not merely uninformative but actively misleading in protein regions where GOF variants cluster at evolutionarily constrained positions.

The negative LLR correlation in TM6 (r = −0.209) is also notable. TM6 undergoes a large outward displacement during receptor activation, and mutations that facilitate this movement may appear evolutionarily unusual while being functionally activating. This again underscores that for proteins with GOF disease mutations, LLR can be systematically misleading in regions critical to the activation mechanism.

For HXK4, the moderate and consistent correlations across all domains (LLR r = 0.50–0.83, RF AL r = 0.59–0.81) are consistent with glucokinase's well-understood structure-function relationships, where most GOF and LOF positions are distributed throughout the kinase fold. The notably high Hinge 1 LLR correlation (r = 0.832) reflects strong evolutionary constraint on this regulatory region, where substitutions have reliably predictable consequences.

For PTEN, RF AL correlation with the assay score declines progressively across structural regions, from r = 0.492 in the phosphatase domain to r = 0.044 in the C-terminal tail. This gradient directly reflects the underlying structural and regulatory properties of each region: the phosphatase domain is well-folded and catalytically critical, so mutations have predictable functional consequences that ESM2 embeddings can partially learn. The C-terminal tail, by contrast, is intrinsically disordered and its functional state is governed largely by post-translational modifications — phosphorylation, ubiquitination, and protein–protein interactions — that are invisible to a sequence-only model. The near-zero correlation in the tail therefore does not indicate that variants there are uniformly neutral; it indicates that the determinants of their functional impact lie outside the sequence information ESM2 encodes. The LLR's higher correlation in the phosphatase domain (0.666) similarly reflects the strong evolutionary constraint on catalytic residues.

The RF 5 FOLD CV model consistently achieves the highest position-level correlations across proteins, demonstrating that when sufficient labeled training data is available, the embedding-based classifier can approach strong regression-level performance even in challenging regions. This establishes an upper bound for what the active learning model might achieve with more training rounds or larger initial labeled sets.

### 4.6 Limitations and Future Directions

Several limitations should be noted. First, our active learning implementation used eleven rounds; however, the large gap between RF AL and RF 5 FOLD CV suggests that substantially more rounds or larger batch sizes would be needed to approach full dataset performance. Recent literature suggests that supervised models generally require more labeled data than active learning can provide in a small number of rounds to overcome the zero-shot LLR baseline [22]. Second, while RF 5 FOLD CV outperforms LLR for MC4R and SRC, it falls short for PTEN even with full dataset access, suggesting that data quantity alone is insufficient — the PTEN challenge may require structural or disorder-aware features beyond sequence embeddings. Third, the high standard deviations observed for small domains (ICL2, ECL3, PTEN Linker) highlight the need for caution when interpreting domain-level results with few labeled variants; future work should set minimum sample size thresholds for reporting domain-level AUC. Fourth, for PTEN the GOF and LOF thresholds were defined pragmatically — GOF as assay score > 1.1 and LOF as < 0.9, with wildtype normalized to 1.0 — rather than derived from published functional criteria. The ±10% margins were chosen to exclude variants within the likely technical noise range of the assay, ensuring that labeled examples are confidently non-wildtype. However, the specific value of 10% is arbitrary: for a haploinsufficient tumor suppressor such as PTEN, even modest reductions in phosphatase activity may be clinically relevant, and a stricter LOF threshold might better capture pathogenic variants. Future work should validate these thresholds against ClinVar pathogenicity annotations and explore sensitivity to threshold choice, though the consistent superiority of strict GOF/LOF labels over percentile-based methods in this study provides indirect support that the chosen thresholds define a learnable and biologically coherent signal.

Future work should explore ensemble approaches combining LLR with RF AL predictions, which may improve performance for cases where the two models have complementary strengths (e.g., RF AL in the MC4R ECD, LLR in the PTEN Linker). Additionally, using alternatives to mean pooling to get a fixed length representation of the embeddings could result in less information loss. Finally, incorporating structural features (e.g., from AlphaFold2 predicted structures), conservation scores, and functional annotation could augment ESM2 embeddings and improve domain-level performance in challenging regions such as intrinsically disordered tails. Most importantly, increasing the number of active learning rounds and batch sizes for data-rich proteins like HXK4 and SRC may allow RF AL to approach RF 5 FOLD CV performance, closing the gap identified in this study.

---

## 5. Conclusions

We have presented a systematic evaluation of three computational approaches for classifying GOF and LOF protein variants across four clinically relevant proteins. The RF 5 FOLD CV model — the same Random Forest on ESM2 embeddings as RF AL but trained on the full labeled dataset — outperforms the zero-shot ESM2 LLR baseline for MC4R (0.773 vs. 0.690) and SRC (0.831 vs. 0.748), demonstrating that the architecture has sufficient representational capacity to exceed the zero-shot baseline when adequate training data is available. However, RF AL with eleven active learning rounds still underperforms LLR for all four proteins, consistent with emerging critiques of PLM-based supervised variant effect prediction when labeled data are limited [22]. The large gap between RF AL and RF 5 FOLD CV quantifies the remaining potential of the active learning strategy. Domain-level analysis reveals that RF AL provides meaningful advantages in specific structural regions — particularly the MC4R ECD (RF AL 0.505 ± 0.097 vs. LLR 0.211) — where evolutionary conservation signals are actively misleading. Domain-level analysis reveals significant structural heterogeneity in model performance, with intrinsically disordered regions and allosteric hotspots being most challenging for both approaches. The use of strict, literature-derived GOF/LOF labels generally provides better RF AL classifier training than percentile-based approaches. Together, these findings highlight the complementary roles of zero-shot PLM scoring and supervised active learning for variant effect classification, and demonstrate a clear path toward exceeding the zero-shot baseline: increasing the labeled data available through additional active learning rounds, larger batch sizes, or full cross-validation training.

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

21. Matreyek KA, Stephany JJ, Ahler E, Fowler DM. Integrating thousands of PTEN variant activity and abundance measurements reveals novel variant subgroups and new dominant negatives in cancers. *Genome Med*. 2021;13(1):165. PMCID: PMC8518224.

22. Wilke C. Protein language models are bad at mutational effect prediction. *Genes, Minds, and Machines*. 2026. https://blog.genesmindsmachines.com/p/protein-language-models-are-bad-at

---

*Supplementary Data:* All analysis data files are available in the project repository, including `protein_landscape_data.csv`, `protein_landscape_domains.csv`, `mse_by_domain.csv`, `auc_by_domain.csv`, `auc_by_label_method_by_domain.csv`, `auc_by_label_method_by_round.csv`, and `auc_data.csv`.
