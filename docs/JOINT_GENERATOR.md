# CardiAgent 0.6.0: joint conditional generation

The failed CVAE and the separate marginal quantile model retain their original
identities, reports and scientific limitations. This release adds the explicit
`copula` family. There is no silent replacement or retroactive CVAE qualification.

## Model and support

Each domain/severity/difficulty stratum fits empirical marginal quantiles and a
Gaussian rank copula across all 11 scalar fields. Average ranks handle tied values;
normal scores estimate latent dependence. Constants have zero cross-correlation
and remain constant at observed conditions. Correlation shrinks toward identity
(default 0.02) to maintain positive definiteness. Correlated normal samples pass
through Gaussian CDFs and midpoint empirical inverse quantiles. This estimates
dependence rather than assuming feature independence.

For previously unseen conditions inside observed support, SVD identifies the
affine condition dimension. Two-dimensional conditions use Delaunay simplices;
collinear conditions use linear segments; singleton strata support only that
exact condition. Convex weights mix marginal quantile functions and latent
correlation matrices. This is an explicit smooth interpolation assumption. It
cannot validate unmeasured intermediate conditions merely by producing samples.
Unseen domains, outside-hull points and points outside a collinear affine support
fail closed. Constants at every contributing stratum remain constant; differing
constants interpolate, which is not a mechanism-derived state change.

There must be 1..1,000 strata, each with at least 32 unique row IDs. Unique IDs do
not prove independent patients or independent observations. User datasets require
appropriate subject grouping, protocol/domain alignment and held-out subjects.
Sampling is limited to 100,000 agents per call. NumPy/SciPy are optional `[joint]`
dependencies; PyTorch is not used. Core installation remains dependency-free.

Checkpoints retain sorted marginals, correlation matrices, training source hash,
private RNG state, cached Gaussian state and sample counter. Loading validates
support, finite bounds, sizes, symmetry, unit diagonal, positive definiteness and
unqualified status. JSON duplicate keys are rejected. Reloading continues the
same private random sequence without changing Python's global random state.

All generated objects declare `quality_status: not_qualified`,
`patient_validated: false`, `privacy_guarantee: false`, and no temporal profile.
Scalar onset/persistence parameters do not constitute longitudinal trajectories.
The API's default CVAE selection remains explicit; callers select `copula`.

## Independent dependence-sensitive protocol

The source uses a declared Gaussian-copula law with three correlated pairs:
stress/inflammation (+0.85 latent correlation), contractile impairment/viability
loss (-0.70), and onset/persistence (+0.75). Another field is an exact zero
constant. Other fields vary independently. Means vary linearly with severity
and difficulty. The generator receives training observations only, not source
correlation parameters or held-out observations.

Training conditions are the four corners of [0.2,0.8]². Held-out conditions are
(0.35,0.35), (0.5,0.5), (0.65,0.65), absent from training. All 11 fields are
screened, unlike the historical eight-field quantile audit. Gates are maximum KS
0.15, maximum Spearman-matrix error 0.12, discriminator AUC 0.65 and exact training
vector copy fraction 0.01. A held-out discriminator split uses a fixed random
forest specification. Independent source controls are never fitted. Negative
controls independently permute columns, preserving every marginal and constant
while destroying dependence. Gates were declared before executing each protocol.

| Protocol | Rows per condition | Model screens passed | Independent controls passed | Broken-dependence controls passed |
|---|---:|---:|---:|---:|
| Preserved pilot | 1,024 | 6/9 | 6/9 | 0/9 |
| Fresh larger confirmation | 4,096 | 9/9 | 9/9 | 0/9 |

The pilot's failures came from maximum correlation error. Its independent control
failed in one of three unique conditions, repeated across three classifier/model
seeds. This exposes finite-sample failure of that multiple-pair gate. The report,
original protocol and exact harness snapshot remain in `validation/cpu/joint`.
They are not converted to passes.

`confirmation_protocol.json` separately declares larger samples and fresh
training/reference seeds, with thresholds and model specification unchanged.
Confirmation uses 16,384 training rows, 12,288 fresh reference rows, and 36,864
joint draws across seeds 17/29/41. Maximum KS is 0.04615, maximum Spearman error
0.07837, and discriminator AUC 0.50442–0.53404. Independent controls pass;
negative-control AUC is 0.80586–0.85722, with rank errors up to 0.87202. No generated
vectors exactly match training vectors in this experiment. Checkpoint continuation
passes for every model seed. These numbers are synthetic verification, not
patient response agreement, privacy or broad-domain qualification.

The harness checks source/protocol hashes before and after execution. Reports
record arrays' fingerprints and runtime versions. Default CI runs confirmation
and uploads the report. Reproduce it with `scripts/validate_joint_cpu.py`; the
preserved pilot can be rerun with `--protocol validation/cpu/joint/protocol.json`
and a separate `--output`, and is expected to exit nonzero. The original pilot
harness is archived as text because later harness changes only added explicit
protocol selection; its original source fingerprint remains independently checkable.

## Limits and required evidence

Gaussian copulas can miss nonlinear/multimodal dependence and joint extremes,
even when rank correlations look correct. They do not enforce physiological
identities, causal response laws, mechanistic constraints or patient trajectories.
Interpolation assumes locally smooth conditional distributions and is unvalidated
outside the declared synthetic law. Strong rank agreement does not guarantee
joint biological plausibility. There is no differential privacy or membership
inference protection; exact-copy absence is only a narrow memorization screen.

Patient realism requires measured phenotype mappings with units/assay context,
donor/subject grouping, relevant clinical strata, held-out subjects/sites,
validated temporal constraints and downstream utility. Privacy requires separate
attack and disclosure analyses. No clinical dataset was supplied or validated
here. Passing synthetic screens must not be promoted to patient-realistic status.

Primary methodological context:
- [Synthetic longitudinal tabular data generation via copula](https://www.biorxiv.org/content/10.64898/2026.08.03.742474v1.full): empirical inverse marginals with Gaussian latent dependence; not evidence that CardiAgent is biologically valid.
- [Fidelity, utility and privacy in synthetic healthcare data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12058740/): distinct evaluation dimensions; statistical fidelity alone is insufficient.

Local CPU release checks: 115 tests passed, 91.65% statement coverage, including
PyTorch software tests. Source and wheel builds pass. Clean core-wheel doctor and
benchmark checks pass without scientific extras; joint-wheel CLI training,
interpolated sampling and source-file identity checks pass. Hosted CI is not
claimed here. Software tests do not rehabilitate the failed CVAE.
