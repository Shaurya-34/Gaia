# Gaia: validated search for nearby co-moving stellar groups

**Goal.** Build a pipeline that finds co-moving stellar groups within about 500 pc in Gaia DR3, and *measure* what it can and cannot detect. The deliverable is a validated pipeline with a measured completeness map and an honest shortlist of candidates. A new group would be a bonus, not the promise.

**Timing.** Gaia DR4 is planned for 2 December 2026. Running the frozen pipeline on DR4 is a stretch goal: whether its tables are compatible with DR3's is unverified. If they are not, the DR3 result is still the deliverable.

## Status

| Piece | Owner | State |
|---|---|---|
| `query_gaia.py` data pull | Claude | drafted, **not run against the real archive** |
| `features.py` feature conversion | Claude | drafted, **not run**; to be switched to the Hunt & Reffert feature space |
| `requirements.txt` | Claude | installs and imports cleanly |
| Injection simulator | Shaurya | not started |
| HDBSCAN baseline, tiling, scorer | Claude | not started |
| Vetter model | Shaurya | not started |
| `PREREG.md` | Shaurya | not started |

## Decisions (set before any run)

- **Feature space:** (l, b, parallax, pm_l, pm_b) with robust scaling, as in Hunt & Reffert 2021. Cartesian position plus velocity scaled by `tau` is kept only as an ablation, because parallax noise smears clusters along the line of sight (up to about 50 pc at 500 pc with a 10% distance error).
- **"Recovered" is reported two ways:** cluster significance score >= 3 (the 2025 selection-function paper's criterion) and membership Jaccard >= 0.5.
- **Injected clusters' bulk velocities:** drawn from real stars with Gaia radial velocities, not a Gaussian. The 2025 paper reports detection probability depends strongly on velocity relative to the field.
- **Scope:** under 500 pc; HDBSCAN minimum cluster sizes of 10, 20, 40 and 80.
- **Recovery is judged against surviving members**, i.e. injected stars that pass the same quality cuts as the real data, not the injected N.

## Conventions

Units: pc, km/s, mas, mas/yr. Frame: heliocentric Galactic Cartesian, x toward the Galactic centre. Column names: `l`, `b`, `parallax_corrected`, `pm_l`, `pm_b`, `*_error`, `is_injected`. Randomness: pass an `np.random.Generator` into every function; no global seeds.

## Injection simulator spec (`injection.py`)

```python
sample_plummer_positions(n, r_half_pc, rng) -> (n,3) pc         # a = r_half/1.305; r = a/sqrt(u^(-2/3)-1)
sample_member_velocities(n, sigma_v_kms, rng) -> (n,3) km/s
cluster_center(dist_pc, l_deg, b_deg) -> (3,) pc
bulk_velocity(field, rng) -> (3,) km/s                          # empirical, from stars with RVs
cartesian_to_observables(pos, vel) -> DataFrame[l,b,parallax,pm_l,pm_b]
assign_errors(obs, field, rng) -> DataFrame                     # errors from nearby real stars (simplification)
add_noise(obs, rng) -> DataFrame
apply_selection(obs) -> DataFrame                               # same cuts as the real pull
inject(field, params, rng) -> (field_with_cluster, truth)
sample_params(n_trials, rng) -> list[params]
```

`truth` holds the injection id, the surviving member `source_id`s and all input parameters.

Tests to write first: median 3D radius of 100,000 Plummer samples is about `r_half`; member velocity std is about `sigma_v`; with noise off, observables round-trip through the feature conversion; surviving fraction falls with distance; same seed gives identical output.

Known simplification: errors are copied from nearby real field stars, and Gaia proper-motion errors (RA/Dec) are approximated in l/b. Injected members do not yet follow a realistic CMD sequence.

## Plan

**This week**
1. Commit these files to the repo.
2. Read the first-kiloparsec paper and Table 1 of the 2025 selection-function paper; fix parameter ranges from them.
3. Write `PREREG.md`: recovery definitions, parameter ranges, quality cuts, false-positive budget.
4. Run `query_gaia.py` on one sky pixel and check the row count and columns (neither verified).
5. Switch `features.py` to the Hunt & Reffert feature space.

**Weeks 1-2**
6. Write the simulator (tests first).
7. Write the HDBSCAN baseline, tiling and scorer.
8. Pull a handful of sky regions first, not all 768 pixels.
9. Gate: the baseline re-finds the Hyades, Pleiades, Praesepe and alpha Per.

**Weeks 3-4**
10. Run injection-recovery; plot completeness against surviving member count, parallax error and field density.
11. Run the shuffle-decoy null to count false positives.

**Weeks 5-6**
12. Build the vetter; split train/test by sky region and audit for leakage.
13. Compare against the baseline at fixed recall; check sim-to-real by scoring real known clusters.
14. Freeze all thresholds in writing and seal one sky region.

**Week 7**
15. Open the sealed region; cross-match candidates against SIMBAD and published catalogs.
16. Check radial-velocity consistency (never used in the search) and CMD tightness.

**Around 2 December:** if DR4 tables are compatible, run the frozen pipeline on them. Otherwise write up DR3 and note DR4 as follow-up work.

## Validation table (what gets published)

| Test | Question | Reported as |
|---|---|---|
| Rediscovery | Does it re-find known clusters? | Recall and precision vs published catalogs, by distance and size |
| Injection-recovery | What would it detect? | Completeness heatmaps |
| Shuffle null | How often does it hallucinate? | False positives per 10^6 stars vs score threshold |
| Model vs baseline | Does the ML add anything? | PR curves at fixed recall, calibration plot |
| Sim-to-real | Is the simulator realistic? | Train on injections, score real known clusters, compare |
| Novelty cross-match | What is actually new? | Fraction matched to SIMBAD and catalogs; remainder tiered |
| Held-out checks | Are unmatched candidates physical? | RV consistency, CMD tightness, orbit traceback |

Candidates that survive are called "kinematically coherent candidate groups". Confirming common birth would need spectroscopy, ages or abundances, which is out of scope.

## Risks

- Parallax noise smearing clusters (addressed by the feature-space choice).
- Train/test leakage in the vetter.
- Injected clusters losing most members to the quality cuts.
- DR4 incompatibility with the DR3 pipeline.
- Prior art: the nearby regime is heavily mined. The open niche is measured completeness for nearby, low-membership groups.

## References

Details below were read through a summarizing tool and should be checked against the originals.

- Hunt & Reffert 2021, Improving the open cluster census I (clustering algorithm comparison): https://www.aanda.org/articles/aa/full_html/2021/02/aa39341-20/aa39341-20.html
- Hunt & Reffert 2023, Improving the open cluster census II (all-sky catalogue): https://arxiv.org/abs/2303.13424
- The selection function of the Gaia DR3 open cluster census (2025): https://arxiv.org/html/2510.18343
- Cantat-Gaudin et al. 2020: https://www.aanda.org/articles/aa/full_html/2020/03/aa37386-19/aa37386-19.html
- Kounkel & Covey 2019: https://iopscience.iop.org/article/10.3847/1538-3881/ab339a
- Exploring Galactic open clusters with Gaia I (2024), not yet read: https://www.aanda.org/articles/aa/full_html/2024/09/aa50901-24/aa50901-24.html
