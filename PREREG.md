# Pre-registration v1 (DRAFT, decisions filled)

Drafted 2026-10-04. Commit this file **before the first injection-recovery run**. After that commit, every change goes in section 11 (Deviations log), never silently in place.

How to read it:
- **[SET]** = a value taken from the literature or the project README and proposed as the default.
- **[CHOSEN]** = a judgement call, made by Claude on 2026-10-04 at Shaurya's request ("sanest working scope"). Each has a one-line reason. These are proposals until Shaurya commits the file; change any of them before the commit, not after.
- **[UNVERIFIED]** = could not be checked from the material read so far.
- **(extra)** = not from the papers or README; from memory or general practice.

---

## 1. Objective, question and claim scope

**Objective.** A working, reusable *method*, not a catalogue. Four parts:
1. a detector (HDBSCAN + significance test) with one config file;
2. an injection-recovery protocol that measures what the detector finds;
3. a **detectability model** that predicts, for any hypothetical group (distance, surviving members, parallax error, local density), the probability that this detector finds it;
4. a validation chain showing the model predicts on **unseen sky** and on **real known clusters**, so the method can be re-run on another sky region, magnitude limit or survey release (DR4) by changing the config, not the code.

**Question.** For a density-based search over Gaia DR3 stars within 500 pc, how many visible member stars does a co-moving group need to be detected, as a function of distance, parallax error and local field density? And does the pipeline also find kinematically coherent groups that are not in the Hunt & Reffert 2023 catalogue?

**We claim:**
- a measured detection-probability model for *this* pipeline, with its out-of-sample accuracy;
- a false-positive rate under a shuffle null;
- a short list of candidate groups with evidence tiers.

**We do not claim:** discoveries, common birth, ages, bound status, the completeness of Hunt & Reffert 2023 or any other catalogue, or anything beyond 500 pc.

**Fixed scope [SET].** 50 to 500 pc, Gaia DR3 only. The distance range is not expanded in the DR3 analysis. Gaia DR4 (planned 2 December 2026) is a separate follow-up and needs its own v2.

**Context.** The 2025 selection-function paper injected clusters only beyond 500 pc and states that the Hunt & Reffert catalogue is "likely 100% complete" in the range covered by their nearby runs, citing Hunt & Reffert 2024 §5.1 [UNVERIFIED: that section has not been read, so its basis is unknown]. Our test applies to our own detector only.

## 2. Data and quality cuts

**Source.** `gaiadr3.gaia_source` via `query_gaia.py`, in HEALPix level-3 chunks.

**Cuts [SET]** (Alfonso et al. 2024, §2):
- `ruwe` < 1.4
- `parallax_over_error` > 10
- `visibility_periods_used` > 6
- `astrometric_excess_noise` < 1
- `phot_g_mean_flux_over_error`, `phot_bp_mean_flux_over_error`, `phot_rp_mean_flux_over_error` > 10

**Zero point [SET].** Apply `gaiadr3_zeropoint` (Lindegren et al. 2021). Keep 5- and 6-parameter solutions (`astrometric_params_solved` in {31, 95}) and 6 < G < 21 (the package's validity range). Keep corrected parallax >= 2 mas (about 500 pc).

**Not an HR23 replica.** Hunt & Reffert 2023 instead uses a Rybizki et al. 2022 quality score >= 0.5 and G < 20. Our pipeline is therefore a different detector, and results are stated for it alone.

**Status.** `query_gaia.py` currently applies only the RUWE and `parallax_over_error` cuts. The other four cuts must be added before any real pull. The same cuts apply to injected stars.

**Data freeze.** Record the archive access date and per-pixel row counts in a manifest file.

**Sealed data [CHOSEN].** The sealed block's data (section 7) is never written to disk or inspected until v2 is committed. Download chunks are ICRS HEALPix pixels, which overlap the Galactic sealed block, so its stars pass through memory; `query_gaia.py` drops them before anything is saved. Reason: not having it removes the temptation to look.

## 3. Search method (the detector under test)

**Features [SET]** (Hunt & Reffert 2021, §2.3): five dimensions: sky position, two proper-motion components, corrected parallax. Each field is re-centred on its own centre and rotated to (0, 0), then robust-scaled (median 0, interquartile range 1).

**HDBSCAN [SET]** (Hunt & Reffert 2021 and 2023, 2025 paper):
- `min_samples` = 10
- `min_cluster_size` in {10, 20, 40, 80}
- `cluster_selection_method` = `leaf` (their finding for small clusters in large fields)
- for each group, keep the run with the highest CST

**Significance test [CHOSEN].** The cluster significance test (CST, Hunt & Reffert 2021) is part of the detector. First code step: check the licence of the authors' code. If it is permissive (MIT/BSD/Apache), vendor it with attribution; if the licence is missing or restrictive, reimplement from the paper's description. Either way, add a unit test on a synthetic Gaussian blob in uniform noise. Reason: avoids a licence blocker without stalling the project.

**Tiling [SET, proposed].** For 50 to 500 pc, follow Hunt & Reffert 2023's scheme for nearby sky: HEALPix level-2 pixels, each field being the pixel plus its 8 neighbours (about 44° x 44°). Reason: Hunt & Reffert 2021 note that 5° level-5 fields are too small for clusters closer than about 350 pc.

**Inside 250 pc [CHOSEN].** Hunt & Reffert 2023 ran one all-sky Cartesian field using Bailer-Jones distances. We do not do a separate run; the same level-2 tiling is used at all distances. Listed as a known limitation (section 9). Reason: a second detector doubles the validation work for a small distance range; the Cartesian features remain available as an ablation.

**Edges and duplicates [CHOSEN].** Adopt Hunt & Reffert 2023: keep detections whose centroid lies in the central pixel, with a tolerance of about 5 pc, and merge duplicates across runs.

**Allowed ablations (reported as ablations, never as the main result):**
- Cartesian position plus velocity scaled by tau (`to_features_cartesian`), tau in {5, 10, 20} Myr [CHOSEN]
- Alfonso et al. 2024 settings: `min_samples` = 5 with a Mahalanobis metric
- run on the validation blocks only, so the test blocks are not spent on ablations [CHOSEN]

The main configuration is not changed after seeing results.

## 4. Injection design

**One cluster per field per run** (2025 paper, §2.3), so injected clusters never contaminate each other.

**Parameter table [CHOSEN]** unless marked [SET].

| Parameter | Distribution | Reason |
| --- | --- | --- |
| Injected members N | log-uniform 10 to 2,000, then topped up per cell (below) | 2025 paper: floor near 10, always detected above ~1,000 surviving |
| Distance | uniform 50 to 500 pc | gives even coverage of the distance bins |
| Half-mass radius `r_half` (Plummer) | log-uniform 0.5 to 10 pc | brackets the 2025 paper's 0.1 to 8 pc core radii; mapping from King core to Plummer radius is approximate (extra) |
| Velocity dispersion `sigma_v` | log-uniform 0.2 to 2 km/s | README proposal; 2025 paper derives it from virial ratio instead |
| Bulk velocity [SET] | drawn from real Gaia stars with RVs | README decision |
| Sky position | uniform on the sphere, inside eligible level-2 pixels of the split's blocks | section 7 |
| Binaries | none | limitation (section 9) |

**Cluster profile [CHOSEN].** Plummer only, as implemented. King (2025 paper) is a listed limitation. Size dependence is checked by comparing `r_half` < 3 pc against `r_half` >= 3 pc. The 2025 paper reports physical size has negligible effect beyond 500 pc; that may not hold nearby, so the check is part of the results.

**Bulk velocity fallback [SET]** (2025 Table 1), relative to circular velocity:
- V_R ~ 0.8 N(0, 35) + 0.2 U(-150, 150) km/s
- V_phi ~ 0.8 N(0, 25) + 0.2 U(-150, 150) km/s
- V_Z ~ 0.8 N(0, 15) + 0.2 U(-100, 100) km/s

Report completeness against velocity relative to the local field. The 2025 paper finds a +25 km/s boost can raise detection probability up to about 3x beyond 500 pc.

**Magnitudes and errors [SET].** Each injected member is paired with a real "twin" Gaia star of similar magnitude near the cluster's sky position and parallax. The twin's astrometric and photometric errors are assigned, noise is drawn from them, and the section 2 cuts are then applied (2025 paper, §2.1). **Not yet implemented:** injected members currently have no magnitudes. No real injection run starts until this exists.

**Grid and trial counts [CHOSEN].**
- Distance bins (5): 50-140, 140-230, 230-320, 320-410, 410-500 pc.
- n_* bins (5): [5, 10), [10, 25), [25, 60), [60, 150), [150, 2000].
- Target: 20 trials per cell in train + validation (500 trials), 6 per cell in test (150 trials).
- Top-up: draw N so that under-filled cells get trials; stop a cell after 100 attempts and report it as empty. Empty cells are a result (the pipeline cannot be tested there), not a failure.
- Reason: roughly 650 detector runs, each repeated for 4 `min_cluster_size` values. The runtime per run is **unmeasured**, so the count is set after the pilot.

**Pilot stage [CHOSEN].** 100 trials in one training block, not used in any result. Purpose: measure runtime per run; check that the cut survival looks sane against distance; verify the Jaccard scorer. Compute ceiling: if the projected main run exceeds about 150 CPU-hours (extra: guessed laptop-scale budget), apply the fallback below.

**Fallback if too slow [CHOSEN].** Inject up to 3 clusters per field, at least 25° apart on the sky and at different distances. Before relying on it, run 60 matched trials both ways (1 vs 3 per field); proceed only if the detection rates differ by less than 10 percentage points. If they differ more, cut the grid (3 distance bins, 3 n_* bins) rather than loosen the one-per-field rule.

**Cross-check [CHOSEN].** Generate 100 clusters at 300 to 500 pc with `ocelot` (github.com/emilyhunt/ocelot; licence [UNVERIFIED], check first; skip and log if the licence forbids use). Pass criterion: survival fractions agree within 10 percentage points, and n_*,50 in the 320-410 pc bin agrees within 0.2 dex. If it fails, the discrepancy is reported and `injection.py` is not silently "fixed" to match.

Record the random seed for every trial.

## 5. Definitions

**n_*** = injected members that exist in Gaia after the section 2 cuts ("surviving members"). Injections with n_* = 0 are non-detections by construction and reported separately. Only trials with n_* >= 5 are used to fit curves [CHOSEN; matches the lowest n_* bin].

**Detected (primary)** = HDBSCAN returns a group **and** CST > tau, with tau = 3 (the Hunt & Reffert 2023 catalogue threshold). Also report tau = 5 and 7.

**Recovered (membership quality)** = Jaccard overlap between the retrieved group and the surviving injected members >= 0.5. Also report the full Jaccard distribution.

**Matching rule [CHOSEN].** Each injection is matched to the retrieved group with the highest Jaccard overlap with its surviving members. Ties go to the group with the higher CST. (The 2025 paper's own matching rule was not seen.)

**Overlap with known clusters [CHOSEN].** Follow the 2025 paper: discard trials whose retrieval holds more stars from a known catalogue cluster than from the injected one, and report how many were discarded.

## 6. Analysis plan and metrics

**Detectability model (the predictive core) [CHOSEN].**

P(detected) = sigmoid( (log10 n_* - a) / sigma ), with
a = a0 + a1 * (d / 100 pc) + a2 * log10(median parallax error / mas) + a3 * log10(rho_data)

- Fit by maximum likelihood on the training blocks, with 1,000-resample bootstrap confidence intervals over trials.
- Validation blocks decide which of a1, a2, a3 are kept (by log-likelihood and calibration) and nothing else.
- Test blocks are scored once.
- Sensitivity check: the generalised logistic of the 2025 paper (eq. 1), P = 1 - [1 + exp((log10 n_* - a) / sigma)]^(-k), free k.
- Also report the empirical n_*,50 (n_* at P = 0.5) per distance bin.
- Reason: the simple form has few parameters and can be fitted with ~500 trials; the generalised form needs more data to identify k.

**Covariates.**
- n_*
- median parallax error of surviving members
- `rho_data` [CHOSEN]: 1 / (distance to the 10th nearest neighbour, in the robust-scaled 5D space, of the injected cluster's centroid in the field *without* the injection). It is HDBSCAN's own core-distance idea, and it is defined at every distance, unlike the 2025 density map, which only reaches 2 mas.
- the optional density ratio R_rho is dropped.

**Predictive-validity tests (what makes the method generalisable).**
1. **Unseen sky.** On test blocks, report Brier score, ROC AUC, and calibration slope and intercept. Pass: calibration slope in 0.8 to 1.2, and Brier score better than a distance-only baseline. A failure is reported, not hidden.
2. **Sim-to-real.** Use the fitted model to predict the probability that each Hunt & Reffert 2023 cluster (d < 500 pc, in train/val/test blocks) is found, using its member count after our cuts as n_*. Compare the predicted number found with the number actually found (Poisson-binomial interval). Not a gate: catalogue membership is not our n_*, so differences are diagnosed, not tuned away.
3. **Transfer.** The same code and config run on another region, or on DR4, with only the config changed. If the DR4 table is compatible, the frozen model's predictions are tested there against fresh injections.

**Pre-stated expectation, not a gate.** Beyond 500 pc the 2025 paper finds n_* dominates, with parallax error and density adding smaller gains. We expect the same ordering nearby.

**Metrics.** Sensitivity, specificity and precision (Hunt & Reffert 2021, §4.1), plus classification accuracy and log-likelihood of the detectability model. Every table reports the number of trials per cell.

## 7. Splits and sealed region

**Frame [CHOSEN].** All blocks, splits and fields use nested HEALPix in the Galactic frame, so the four equatorial blocks (4 to 7) straddle b = 0.

**Unit [CHOSEN].** Level-0 HEALPix pixels ("blocks"; 12 in total). A level-2 pixel is an eligible injection centre only if all 8 of its neighbours have the same level-0 parent (checked with `astropy_healpix.neighbours`; healpy has no Windows build). Reason: its whole 44° field then lies inside one split, so training and test fields share no sky. About 4 eligible pixels per block are expected; the code checks. Eligible pixels are few, so repeated centres are fine and are drawn with different seeds.

**Assignment [CHOSEN].** 12 blocks: 7 train, 2 validation, 2 test, 1 sealed. Each split must include at least one equatorial block (the four that cross the Galactic plane), so the plane is represented everywhere; the sealed block is equatorial. The mapping comes from `splits.py` with seed 20261004 and is committed with v1, before any detector run.

**Frozen split (`splits.json`, committed with this file).** Train: blocks 0, 1, 3, 6, 8, 9, 11. Validation: 5, 10. Test: 2, 7. Sealed: 4. Four eligible level-2 centres per block (48 in total), listed in `splits.json`.

**Sealed region.** The one sealed block. No data is written to disk or inspected, no tuning is done and no injection is run there until v2 is committed. It is then used for the candidate search (section 9).

## 8. Rediscovery, false positives, vetter

**Rediscovery gate [CHOSEN].** Target list = Hunt & Reffert 2023 clusters with d < 500 pc and at least 30 members surviving our cuts, frozen at v1, in train/val/test blocks only.
- Must be recovered (Jaccard >= 0.5 against catalogue members after our cuts): Pleiades, Praesepe, alpha Per (about 136, 186, 175 pc; extra, from memory, unverified).
- Hyades sits at about 47 pc (extra, from memory, unverified), just inside the 50 pc edge, so it is reported but not part of the gate.
- Overall recall on the target list >= 70%. The value is a guess (extra); it is recalibrated on the training blocks only, and the recalibration is logged.
- Report recall by distance and n_*.

**Shuffle null [CHOSEN].** Shuffle proper motions within local cells (level-5 HEALPix pixel x 8 parallax-quantile bins) in real data, run the detector 3 times per validation/test field with different seeds, and count detections above tau. Budget: at most 1 false positive per 10^6 stars at the operating tau, evaluated *after* the CST cut. Reason: Hunt & Reffert 2023 report 7.1 million raw HDBSCAN detections before their signal-to-noise cut, nearly all false.

**Operating tau [CHOSEN].** The smallest tau in {3, 5, 7} that meets the budget on the validation blocks. This is a rule, so it transfers to other data. If none meets it, the pipeline is declared too noisy for candidate claims and only the detectability model is reported.

**Vetter training data.**
- Positives: injected clusters (training blocks).
- Negatives: (i) detector output on shuffled data; (ii) Hunt & Reffert 2023-style fakes, grown as a 5D hypersphere around a random star until it holds the required number of stars.
- Published catalogues are test-only and never used for training.

**Baseline** = HDBSCAN + CST at the operating tau.

**Vetter [CHOSEN].** Gradient-boosted trees (LightGBM, already in `requirements.txt`) on cluster-level features: member count, CST, HDBSCAN stability and persistence, mean membership probability, and 5D extent. The parallax-based absolute-magnitude scatter is **not** a training feature: twin stars give injected members field-like CMD scatter, so the vetter would learn that "injected" means field-like. It is used only as a vetting check on real candidates (section 9). Why not a deep model: few thousand tabular rows, and a simple model is easier to audit. The vetter must beat the baseline at 80% recall (on injected positives, test blocks) by at least 5 percentage points of precision, with a bootstrap interval excluding 0. If it does not, the baseline is the result and that is stated. Features and hyperparameters are frozen at v2.

## 9. Candidate rules and vetting

**Candidate** [CHOSEN] = a detection in the sealed block with CST above the operating tau, at least 10 members, and (if the vetter passed) a vetter score above the threshold frozen at v2.

**Tiers** [CHOSEN]:
- A: Jaccard >= 0.5 with a known entry (Hunt & Reffert 2023, Cantat-Gaudin 2020; members cut with our section 2 cuts), or a SIMBAD cluster within 1° and 10 pc
- B: Jaccard between 0.1 and 0.5, or centroid within 10 pc of a known entry
- C: unmatched

**Vetting for tier C.**
- Radial-velocity consistency (RVs were never used in the search): needs at least 5 members with Gaia RVs. Test: the member RV dispersion must be below the 5th percentile of the dispersions of same-size random draws from field stars in the same sky, parallax and proper-motion neighbourhood (a permutation test, p < 0.05). With fewer than 5 RVs, the group stays "unvetted".
- CMD sequence as supporting evidence only. In Hunt & Reffert 2023, human classifiers flagged just 59.2% of simulated fake clusters as false and called 14% highly likely real.
- Orbit traceback: out of scope for this version (stretch).

**Wording.** "Kinematically coherent candidate groups." Never "discovery", "cluster" or "siblings" without independent confirmation (spectroscopy, ages, abundances: out of scope).

**Known simplifications:** no binary astrometric errors and no binaries, Plummer instead of King, no differential reddening (the 2025 paper models it), no separate Cartesian run inside 250 pc, and a simulator and detector that are not the Hunt & Reffert 2023 pipeline.

## 10. Freeze procedure and order of work

- **v1** (this file): committed before the first injection-recovery run, together with `splits.py` output.
- **v2:** committed before the sealed block is downloaded. It freezes the operating tau, the vetter and its threshold, the fitted covariate set, and any grid change after the pilot, all with numbers.
- After v2, changes go in section 11 with date and reason, and the main configuration is not changed.

Order: (1) pilot, (2) main injection-recovery on train/val, (3) fit and validate the detectability model, (4) shuffle null and operating tau, (5) rediscovery and sim-to-real, (6) vetter, (7) test blocks scored once, (8) v2, (9) sealed block.

## 11. Deviations log

| Date | Section | Change | Reason |
| --- | --- | --- | --- |
| 2026-10-04 | 2 | Data now comes from the Gaia DR3 bulk files (cdn.gea.esac.esa.int, `gaia_source` GaiaSource_*.csv.gz, MD5-checked) through `ingest_bulk.py`, which applies the same cuts locally, instead of archive TAP jobs in level-3 chunks. Same table, cuts, zero point and output columns; sealed-block stars are still dropped in memory and never written. Block-first: the pilot block (6) is pulled first. | Archive jobs failed or took about 9 min each (about 190 h per block; parallel jobs all failed with HTTP 500). Checked: one level-5 archive chunk (665 stars) matches the bulk rows exactly in source_ids and columns, with value differences below 1e-6. |
| 2026-10-04 | 3 | HDBSCAN implementation changed from `sklearn.cluster.HDBSCAN` to the `hdbscan` package (0.8.44). Settings unchanged: `min_samples` = 10, leaf selection, `min_cluster_size` in {10, 20, 40, 80}. | The vetter (section 8) needs cluster stability and persistence, which sklearn does not expose. The two libraries count `min_samples` differently (sklearn includes the point itself): on synthetic fields `hdbscan` at 10 matches sklearn at 11 (ARI 0.96 to 1.00), so the detector now uses 10 neighbours besides the point itself. Whether this matches Hunt & Reffert's own setting is unverified. |
| 2026-10-04 | 1 | Documentation only, no method change. The Context paragraph's "likely 100% complete" claim (marked [UNVERIFIED]) is corrected: Hunt & Reffert 2024 §5.1 gives completeness as a function of cluster mass, R_100%(M) = alpha ln(M) + beta below R_break and flat above, with alpha = 633.1 ± 7.3 pc, beta = -1582.6 ± 39.5 pc, R_break = 2792.9 ± 8.2 pc. The census is stated complete within 1 kpc only for clusters above about 100 Msun, and within 1.8 kpc only above about 230 Msun; the paper calls the fit rough and proof-of-concept (fitted over 40 to 10^4 Msun, assuming a uniform cluster density). Our map therefore covers the low-member-count groups where completeness is not claimed. Not yet seen: how the 2025 paper's own citation of that section words it. | §5.1 read on 2026-10-04 (pp. 12-13 of the proof, from Shaurya's screenshots). The log base is not stated there; the natural log reproduces the stated 1.8 kpc for 230 Msun and the ~1000 Msun break. |

## 12. Sources

Read in part only. Pages read: Alfonso et al. 2024 pp. 1-6; Hunt & Reffert 2023 pp. 1-8; Hunt & Reffert 2021 pp. 1-9; Hunt et al. 2025 pp. 1-13. Appendices and later pages were not read.

- Alfonso, García-Varela & Vieira 2024, Exploring Galactic open clusters with Gaia I: https://www.aanda.org/articles/aa/full_html/2024/09/aa50901-24/aa50901-24.html
- Hunt et al. 2025, The selection function of the Gaia DR3 open cluster census: https://arxiv.org/abs/2510.18343
- Hunt & Reffert 2023, Improving the open cluster census II: https://arxiv.org/abs/2303.13424
- Hunt & Reffert 2021, Improving the open cluster census I: https://www.aanda.org/articles/aa/full_html/2021/02/aa39341-20/aa39341-20.html
- Hunt & Reffert 2024, Improving the open cluster census III (cited for the nearby-completeness claim; not read): https://www.aanda.org/articles/aa/full_html/2024/06/aa48662-23/aa48662-23.html
- `ocelot`: https://github.com/emilyhunt/ocelot
  - Selection-function code: https://github.com/emilyhunt/hr_selection_function
