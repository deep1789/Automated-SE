# Novelty / related-work report (search date 2026-10-03)

Method: WebSearch (extended for hard queries) + WebFetch of arXiv abstract pages. "Verified" = title/authors/venue read from a fetched or search-returned source. Items marked (memory) were NOT re-checked in this session.

## Verdict by claim
- C1 (conformal risk control for vuln-detection triage): LOW-MEDIUM risk. I found NO paper applying CRC/conformal prediction to learning-based vulnerability detection or defect prediction triage (several query formulations). Closest: RisCoSet (risk-controlling prediction sets for code generation), Transcendent (conformal rejection under drift, malware), UQ-in-defect-prediction study. Caveat: absence of search hits is not proof; recheck right before submission (arXiv, SANER/ICSE/ASE 2026).
- C2 (guarantee failure under shift + repairs): MEDIUM risk. Individual repairs are standard (Tibshirani 2019 weighted CP; Gibbs & Candes 2021 ACI), and Transcendent already did conformal+drift in security. Novelty is the SE-specific empirical study (project-disjoint/temporal/cross-dataset). Frame as empirical, not methodological.
- C3 (leakage audit + formatting shortcut): HIGH risk for the audit, LOW-MEDIUM for the shortcut. Duplication/label-noise in Big-Vul/DiverseVul/CVEfixes is established (Croft; PrimeVul; R+R; VulGate). Specific numeric overlap among DiverseVul/Big-Vul/PrimeVul and a whitespace/comment-statistics-only AUROC 0.92 were not found in any source; spurious-feature work targets identifiers/variable names, not formatting.
- C4 (frozen CodeBERT): not a contribution by itself; weak, expected reviewer complaint (no LLMs, no fine-tuned models). Justify as a controlled score source.

## Verified papers and overlap
1. Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?" ICSE 2025 (arXiv 2403.18624). PrimeVul; dedup, chronological split; shows Big-Vul F1 collapse. Overlaps C3 and temporal part of C2. Must cite.
2. Croft, Babar, Kholoosi, "Data Quality for Software Vulnerability Datasets", ICSE 2023 (arXiv 2301.05456). 17-99% duplication, 20-71% label inaccuracy across four datasets. Biggest threat to C3 audit novelty.
3. Chakraborty et al., "Deep Learning based Vulnerability Detection: Are We There Yet?" TSE/ICSE-22 journal-first (arXiv 2009.07235). Realistic-setting drop >50%, models learn dataset artifacts. Overlaps C2/C3.
4. Chen, Ding, Alowain, Chen, Wagner, "DiverseVul", RAID 2023 (arXiv 2304.00409). Dataset; reports ~60% label accuracy (via secondary source).
5. Yadav & Wilson, "R+R: Security Vulnerability Dataset Quality Is Critical", ACSAC 2024 (arXiv 2503.06387). Duplication/label accuracy; deduped retraining. Overlaps C3.
6. Safdar et al., "Data and Context Matter: ... VulGate", arXiv 2508.16625 (no venue found). Cleans/unifies many datasets incl. DiverseVul, PrimeVul, MegaVul; closest to a cross-dataset overlap audit. Check before claiming overlap numbers are new.
7. Paramitha, Feng, Massacci, "Today's Cat Is Tomorrow's Dog...", FSE 2025 (arXiv 2506.11939). Time-aware labels/splits; CodeBERT et al. Overlaps temporal shift in C2.
8. Chakraborty et al. (Real-Vul), "Revisiting the Performance of DL-Based Vulnerability Detection on Realistic Datasets", TSE 2024 / ICSE-25 journal-first (arXiv 2407.03093; author list not verified here). Realistic drop; embedding overlap of vulnerable and uncertain samples. Overlaps C2.
9. Rahman et al., "Towards Causal Deep Learning for Vulnerability Detection", ICSE 2024 (arXiv 2310.07958). Spurious features (variable names), OOD. Overlaps C3 shortcut.
10. Das et al., "Are We Learning the Right Features?", ICSE 2025 (arXiv 2501.13291). VIPer perturbations; spurious vs. vulnerability features. Overlaps C3 shortcut.
11. Xu et al., "Uncertainty Quantification for LLM-based Code Generation" (RisCoSet), arXiv 2605.12201. Multiple-hypothesis-testing risk-controlling prediction sets for code generation. Closest conceptual precedent for C1 in SE; differentiate (generation vs. detection triage, FNR of auto-clear).
12. Peng et al., "Understanding Software Defect Prediction: ... Uncertainty Quantification and Performance Evaluation", arXiv 2607.01842. 16 classifiers, within/cross-project; UQ signals do not transfer across projects. Supports your C2 motivation; no conformal guarantee. Must cite.
13. Tung, Du, Neelofar, Aleti, "UntrustVul", IEEE TSE (accepted; arXiv 2503.14852). Flags untrustworthy vuln-detector alerts via line semantics; no statistical guarantee. Overlaps triage framing.
14. Barbero, Pendlebury, Pierazzi, Cavallaro, "Transcending Transcend", IEEE S&P 2022 (arXiv 2010.03856). Conformal rejection under concept drift in malware classification. Closest conformal+drift+security precedent; reviewers will raise it for C2.
15. Spiess et al., "Calibration and Correctness of Language Models for Code", ICSE 2025 (arXiv 2402.02047). Calibration (Platt scaling) for code LMs; no distribution-free guarantee.
16. Siddiq, Rahman, Santos, "An Empirical Study of Security Calibration in LLMs for Code", ICSME 2026 (arXiv 2606.31159). Overconfidence in security judgements of generated code. Peripheral.
17. Ullah et al., "LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?)", IEEE S&P 2024 (arXiv 2312.12575). LLM non-determinism/non-robustness. Peripheral (LLM reliability).
18. Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control", ICLR 2024 (venue confirmed; arXiv 2208.02814 id from memory). Method foundation.
19. Static-warning triage: Yang et al. "Understanding Static Code Warnings: an Incremental AI Approach", arXiv 1911.01387 (authors not verified); human-in-loop filtering of alerts, no guarantees. Peripheral.
Memory-only (not verified; check before citing): Gibbs & Candes ACI (NeurIPS 2021); Tibshirani et al. covariate-shift CP (NeurIPS 2019); Bates et al. Learn-then-Test; Angelopoulos & Bates gentle intro (arXiv 2107.07511, appeared in search).

## Most likely reviewer attacks
1. "Conformal on a classifier threshold is textbook CRC/LTT; where is the SE contribution?" Answer by the shift study and practical triage cost analysis.
2. "Duplication/leakage already shown by Croft, PrimeVul, R+R, VulGate." Differentiate with exact pairwise overlap numbers and downstream effect on guarantee validity.
3. "Why only frozen CodeBERT, and does Big-Vul shortcut matter on PrimeVul?" Add fine-tuned and LLM scorer, and shortcut check on PrimeVul/DiverseVul.
4. "Transcendent/UQ-in-defect-prediction already show conformal/uncertainty failing under drift."

## Must cite and differentiate
Ding 2025, Croft 2023, Paramitha 2025, Chakraborty 2021, Barbero 2022, Angelopoulos 2024 (CRC), Xu RisCoSet, Peng UQ-defect study, Rahman 2024, Das 2025, VulGate, R+R, Spiess 2025, plus ACI/weighted CP originals.

## Search limits
WebSearch is US-only/snippet-based; Google Scholar, ACM DL, IEEE Xplore not queried directly. Absence results above are "not found", not "does not exist".
