"""Held-out scientific validation for CardiAgent's CVAE generator.

This validation is intentionally stricter than the original Colab diagnostic.
It uses a stratified 60/20/20 train/validation/test split, evaluates generated
cases at the exact domain/severity of held-out observations, measures variance
collapse and distributional mismatch, compares novelty to a held-out baseline,
runs a generated-vs-held-out discriminator, and repeats ML training across
multiple seeds.

The source population is the deterministic CardiAgent generator itself. This
therefore tests internal computational generalization of the ML generator; it
is not external biological validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


OUT = Path("validation_colab_results_v2")
OUT.mkdir(exist_ok=True)

DATA_SEED = 20260917
SPLIT_SEED = 20260918
MODEL_SEEDS = (17, 29, 41)
REPRO_SEED = 17
SEVERITIES = tuple(round(float(x), 1) for x in np.linspace(0.1, 0.9, 9))
N_PER_STRATUM = 45
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
ML_EPOCHS = 100
BATCH_SIZE = 128
BOOTSTRAP_REPS = 400

PHENOTYPE_FIELDS = (
    "stress",
    "inflammation",
    "electrical_instability",
    "contractile_impairment",
    "viability_loss",
    "oxidative_stress",
    "metabolic_disruption",
    "remodeling_signal",
)

# Predeclared screening targets. These are screening heuristics rather than
# universal acceptance criteria; raw values remain in the JSON report.
SCREEN = {
    "max_standardized_mean_bias": 0.50,
    "min_variance_ratio": 0.50,
    "max_variance_ratio": 2.00,
    "max_ks_d": 0.20,
    "max_mean_abs_correlation_error": 0.20,
    "max_discriminator_auc": 0.65,
    "min_novelty_ratio": 0.50,
    "max_novelty_ratio": 2.00,
}


def canonical(obj):
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return canonical(vars(obj))
    return obj


def sha256_json(obj):
    raw = json.dumps(
        canonical(obj),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def phenotype_vector(agent):
    return np.asarray(
        [getattr(agent.phenotype, field) for field in PHENOTYPE_FIELDS],
        dtype=float,
    )


def environment_report():
    def command(cmd):
        try:
            return subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except Exception:
            return None

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": command(["git", "rev-parse", "HEAD"]),
    }

    try:
        import scipy

        result["scipy"] = scipy.__version__
    except Exception as exc:
        result["scipy_error"] = repr(exc)

    try:
        import sklearn

        result["scikit_learn"] = sklearn.__version__
    except Exception as exc:
        result["scikit_learn_error"] = repr(exc)

    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_error"] = repr(exc)

    return result


def generate_population():
    generator = ChallengeGenerator(seed=DATA_SEED)
    strata = {}

    for domain in ChallengeDomain:
        for severity in SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"
            strata[key] = [
                generator.generate(
                    domain,
                    severity=severity,
                    difficulty=severity,
                )
                for _ in range(N_PER_STRATUM)
            ]

    return strata


def split_population(strata):
    rng = np.random.default_rng(SPLIT_SEED)
    train, validation, test = [], [], []
    manifest = {}

    n_train = int(N_PER_STRATUM * TRAIN_FRACTION)
    n_validation = int(N_PER_STRATUM * VALIDATION_FRACTION)
    n_test = N_PER_STRATUM - n_train - n_validation

    if (n_train, n_validation, n_test) != (27, 9, 9):
        raise RuntimeError("Unexpected split sizes")

    for key, rows in strata.items():
        order = rng.permutation(len(rows))
        train_idx = order[:n_train]
        validation_idx = order[n_train:n_train + n_validation]
        test_idx = order[n_train + n_validation:]

        train.extend(rows[i] for i in train_idx)
        validation.extend(rows[i] for i in validation_idx)
        test.extend(rows[i] for i in test_idx)

        manifest[key] = {
            "train_ids": [rows[i].agent_id for i in train_idx],
            "validation_ids": [rows[i].agent_id for i in validation_idx],
            "test_ids": [rows[i].agent_id for i in test_idx],
        }

    return train, validation, test, manifest


def verify_split_integrity(train, validation, test):
    groups = {
        "train": [a.agent_id for a in train],
        "validation": [a.agent_id for a in validation],
        "test": [a.agent_id for a in test],
    }
    sets = {k: set(v) for k, v in groups.items()}

    overlaps = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    duplicate_count = sum(len(v) - len(set(v)) for v in groups.values())

    return {
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "duplicate_id_count": duplicate_count,
        "overlaps": overlaps,
        "status": (
            "PASS"
            if duplicate_count == 0 and all(not v for v in overlaps.values())
            else "FAIL"
        ),
    }


def safe_correlation(X):
    """Return NaN when a correlation is mathematically undefined."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    result = np.full((p, p), np.nan, dtype=float)

    if len(X) < 2:
        return result

    std = np.std(X, axis=0, ddof=1)
    for i in range(p):
        if std[i] > 1e-12:
            result[i, i] = 1.0
        for j in range(i + 1, p):
            if std[i] <= 1e-12 or std[j] <= 1e-12:
                continue
            value = np.corrcoef(X[:, i], X[:, j])[0, 1]
            result[i, j] = value
            result[j, i] = value

    return result


def bootstrap_mean_difference_ci(reference, generated, seed):
    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)
    rng = np.random.default_rng(seed)

    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        ref_sample = rng.choice(reference, size=len(reference), replace=True)
        gen_sample = rng.choice(generated, size=len(generated), replace=True)
        draws[i] = np.mean(gen_sample) - np.mean(ref_sample)

    return {
        "estimate": float(np.mean(generated) - np.mean(reference)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def univariate_metrics(reference, generated, seed_offset):
    from scipy.stats import ks_2samp, wasserstein_distance

    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)
    rows = []

    for index, feature in enumerate(PHENOTYPE_FIELDS):
        ref = reference[:, index]
        gen = generated[:, index]
        ref_sd = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
        gen_sd = float(np.std(gen, ddof=1)) if len(gen) > 1 else 0.0
        ref_mean = float(np.mean(ref))
        gen_mean = float(np.mean(gen))

        row = {
            "feature": feature,
            "reference_mean": ref_mean,
            "generated_mean": gen_mean,
            "reference_sd": ref_sd,
            "generated_sd": gen_sd,
            "mean_bias": gen_mean - ref_mean,
            "absolute_mean_bias": abs(gen_mean - ref_mean),
            "standardized_mean_bias": (
                abs(gen_mean - ref_mean) / ref_sd
                if ref_sd > 1e-12
                else None
            ),
            "variance_ratio": (
                (gen_sd / ref_sd) ** 2
                if ref_sd > 1e-12
                else None
            ),
            "reference_nonzero_rate": float(np.mean(np.abs(ref) > 1e-12)),
            "generated_nonzero_rate": float(np.mean(np.abs(gen) > 1e-12)),
            "bootstrap_mean_difference_95ci": bootstrap_mean_difference_ci(
                ref,
                gen,
                seed=SPLIT_SEED + seed_offset * 100 + index,
            ),
        }

        if ref_sd <= 1e-12:
            row.update({
                "reference_constant": True,
                "ks_d": None,
                "ks_pvalue": None,
                "wasserstein": float(wasserstein_distance(ref, gen)),
                "constant_support_violation": bool(
                    np.any(np.abs(gen - ref_mean) > 1e-8)
                ),
            })
        else:
            ks = ks_2samp(ref, gen, alternative="two-sided", method="auto")
            row.update({
                "reference_constant": False,
                "ks_d": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein": float(wasserstein_distance(ref, gen)),
                "constant_support_violation": False,
            })

        rows.append(row)

    return rows


def correlation_metrics(reference, generated):
    ref_corr = safe_correlation(reference)
    gen_corr = safe_correlation(generated)

    valid_errors = []
    undefined_reference = []
    undefined_generated = []

    for i, feature_i in enumerate(PHENOTYPE_FIELDS):
        for j in range(i + 1, len(PHENOTYPE_FIELDS)):
            feature_j = PHENOTYPE_FIELDS[j]
            ref_value = ref_corr[i, j]
            gen_value = gen_corr[i, j]

            if np.isfinite(ref_value) and np.isfinite(gen_value):
                valid_errors.append(abs(float(ref_value - gen_value)))
            elif not np.isfinite(ref_value):
                undefined_reference.append([feature_i, feature_j])
            elif not np.isfinite(gen_value):
                undefined_generated.append([feature_i, feature_j])

    return {
        "valid_pair_count": len(valid_errors),
        "mean_abs_correlation_error": (
            float(np.mean(valid_errors)) if valid_errors else None
        ),
        "max_abs_correlation_error": (
            float(np.max(valid_errors)) if valid_errors else None
        ),
        "undefined_reference_pairs": undefined_reference,
        "undefined_generated_pairs": undefined_generated,
        "reference_correlation_matrix": ref_corr.round(6).tolist(),
        "generated_correlation_matrix": gen_corr.round(6).tolist(),
    }


def nearest_neighbor_metrics(train, generated, heldout):
    """Compare generated->train distance against heldout->train distance."""
    train = np.asarray(train, dtype=float)
    generated = np.asarray(generated, dtype=float)
    heldout = np.asarray(heldout, dtype=float)

    scale = np.std(train, axis=0, ddof=1)
    active = scale > 1e-12

    if not np.any(active):
        return {
            "status": "UNDEFINED_ALL_TRAIN_FEATURES_CONSTANT",
            "active_feature_count": 0,
        }

    train = train[:, active] / scale[active]
    generated = generated[:, active] / scale[active]
    heldout = heldout[:, active] / scale[active]

    def nearest_mean(A, B):
        distances = []
        for start in range(0, len(A), 256):
            chunk = A[start:start + 256]
            d2 = (
                np.sum(chunk[:, None, :] ** 2, axis=2)
                + np.sum(B[None, :, :] ** 2, axis=2)
                - 2.0 * chunk @ B.T
            )
            d2 = np.maximum(d2, 0.0)
            distances.extend(np.sqrt(np.min(d2, axis=1)))
        return float(np.mean(distances))

    generated_distance = nearest_mean(generated, train)
    heldout_distance = nearest_mean(heldout, train)

    return {
        "status": "PASS",
        "active_feature_count": int(np.sum(active)),
        "generated_to_train_mean_nn": generated_distance,
        "heldout_to_train_mean_nn": heldout_distance,
        "novelty_ratio_generated_over_heldout": (
            float(generated_distance / heldout_distance)
            if heldout_distance > 1e-12
            else None
        ),
    }


def duplicate_metrics(generated):
    rounded = [tuple(np.round(phenotype_vector(a), 6)) for a in generated]
    unique = len(set(rounded))
    return {
        "n": len(rounded),
        "unique_vectors_1e-6": unique,
        "duplicate_vectors_1e-6": len(rounded) - unique,
        "unique_fraction_1e-6": (
            float(unique / len(rounded)) if rounded else None
        ),
    }


def discriminator_metrics(reference, generated, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_ref = np.asarray(reference, dtype=float)
    X_gen = np.asarray(generated, dtype=float)
    X = np.vstack([X_ref, X_gen])
    y = np.concatenate([
        np.zeros(len(X_ref), dtype=int),
        np.ones(len(X_gen), dtype=int),
    ])

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        classifier,
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )

    return {
        "fold_auc": [float(x) for x in scores],
        "mean_auc": float(np.mean(scores)),
        "sd_auc": float(np.std(scores, ddof=1)),
        "ideal_auc": 0.50,
    }


def condition_map(agents):
    result = {}
    for agent in agents:
        key = f"{agent.domain.value}|{agent.severity:.1f}"
        result.setdefault(key, []).append(agent)
    return result


def aggregate_reference_metrics(
    reference_agents,
    generated_agents,
    train_agents,
    heldout_for_novelty,
    seed_offset,
    discriminator_seed=None,
):
    reference = np.stack([phenotype_vector(a) for a in reference_agents])
    generated = np.stack([phenotype_vector(a) for a in generated_agents])
    train = np.stack([phenotype_vector(a) for a in train_agents])
    heldout = np.stack([phenotype_vector(a) for a in heldout_for_novelty])

    uni = univariate_metrics(reference, generated, seed_offset)
    corr = correlation_metrics(reference, generated)
    novelty = nearest_neighbor_metrics(train, generated, heldout)

    active = [r for r in uni if not r["reference_constant"]]
    standardized_bias = [
        r["standardized_mean_bias"]
        for r in active
        if r["standardized_mean_bias"] is not None
    ]
    variance_ratios = [
        r["variance_ratio"]
        for r in active
        if r["variance_ratio"] is not None
    ]
    ks_values = [
        r["ks_d"] for r in active if r["ks_d"] is not None
    ]

    result = {
        "n_reference": len(reference_agents),
        "n_generated": len(generated_agents),
        "univariate": uni,
        "correlation": corr,
        "novelty": novelty,
        "duplicates": duplicate_metrics(generated_agents),
        "summary": {
            "max_standardized_mean_bias": (
                float(max(standardized_bias)) if standardized_bias else None
            ),
            "median_standardized_mean_bias": (
                float(np.median(standardized_bias))
                if standardized_bias else None
            ),
            "min_variance_ratio": (
                float(min(variance_ratios)) if variance_ratios else None
            ),
            "max_variance_ratio": (
                float(max(variance_ratios)) if variance_ratios else None
            ),
            "max_ks_d": float(max(ks_values)) if ks_values else None,
            "mean_abs_correlation_error": corr[
                "mean_abs_correlation_error"
            ],
            "constant_support_violations": int(sum(
                r["constant_support_violation"] for r in uni
            )),
        },
        "generated_feature_means": np.mean(generated, axis=0).round(8).tolist(),
        "generated_feature_sds": np.std(generated, axis=0, ddof=1).round(8).tolist(),
    }

    if discriminator_seed is not None:
        result["discriminator"] = discriminator_metrics(
            reference,
            generated,
            discriminator_seed,
        )

    return result


def screen(metrics):
    summary = metrics["summary"]
    novelty_ratio = metrics["novelty"].get("novelty_ratio_generated_over_heldout")

    checks = {
        "standardized_mean_bias": (
            summary["max_standardized_mean_bias"] is None
            or summary["max_standardized_mean_bias"]
            <= SCREEN["max_standardized_mean_bias"]
        ),
        "variance_ratio": (
            summary["min_variance_ratio"] is None
            or (
                summary["min_variance_ratio"] >= SCREEN["min_variance_ratio"]
                and summary["max_variance_ratio"] <= SCREEN["max_variance_ratio"]
            )
        ),
        "ks": (
            summary["max_ks_d"] is None
            or summary["max_ks_d"] <= SCREEN["max_ks_d"]
        ),
        "correlation": (
            summary["mean_abs_correlation_error"] is None
            or summary["mean_abs_correlation_error"]
            <= SCREEN["max_mean_abs_correlation_error"]
        ),
        "constant_support": summary["constant_support_violations"] == 0,
        "novelty": (
            novelty_ratio is None
            or (
                SCREEN["min_novelty_ratio"]
                <= novelty_ratio
                <= SCREEN["max_novelty_ratio"]
            )
        ),
    }

    if "discriminator" in metrics:
        checks["discriminator_auc"] = (
            metrics["discriminator"]["mean_auc"]
            <= SCREEN["max_discriminator_auc"]
        )

    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def severity_response(test_map, generated_map):
    rows = []

    for domain in ChallengeDomain:
        test_means = []
        generated_means = []

        for severity in SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"
            test_X = np.stack([phenotype_vector(a) for a in test_map[key]])
            generated_X = np.stack([
                phenotype_vector(a) for a in generated_map[key]
            ])
            test_means.append(float(np.mean(test_X)))
            generated_means.append(float(np.mean(generated_X)))

        test_slope, test_intercept = np.polyfit(
            SEVERITIES,
            test_means,
            1,
        )
        generated_slope, generated_intercept = np.polyfit(
            SEVERITIES,
            generated_means,
            1,
        )

        rows.append({
            "domain": domain.value,
            "test_slope": float(test_slope),
            "generated_slope": float(generated_slope),
            "slope_difference": float(generated_slope - test_slope),
            "test_intercept": float(test_intercept),
            "generated_intercept": float(generated_intercept),
            "test_mean_burden_by_severity": test_means,
            "generated_mean_burden_by_severity": generated_means,
        })

    return rows


def train_models(train_agents):
    from cardiagent.ml import AgentGeneratorModel

    models = []
    summaries = []
    for seed in MODEL_SEEDS:
        start = time.time()
        model = AgentGeneratorModel(seed=seed, latent_dim=12)
        model.fit(
            train_agents,
            epochs=ML_EPOCHS,
            learning_rate=2e-3,
            beta=0.015,
            batch_size=BATCH_SIZE,
            verbose=False,
        )
        models.append(model)
        summaries.append({
            "seed": seed,
            "elapsed_seconds": time.time() - start,
            "training_summary": canonical(model.training_summary),
        })
    return models, summaries


def generate_conditioned(model):
    generated = []
    for domain in ChallengeDomain:
        for severity in SEVERITIES:
            generated.extend(model.sample(
                domain=domain,
                severity=severity,
                difficulty=severity,
                count=9,
                agent_id_prefix=f"V2-{domain.value}-{severity:.1f}",
            ))
    return generated


def evaluate_model(model, seed, train, validation, test):
    validation_map = condition_map(validation)
    test_map = condition_map(test)
    generated = generate_conditioned(model)
    generated_map = condition_map(generated)

    per_condition = {}
    for domain in ChallengeDomain:
        for severity in SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"
            train_rows = [
                a for a in train
                if a.domain == domain and abs(a.severity - severity) < 1e-12
            ]
            validation_rows = validation_map[key]
            test_rows = test_map[key]
            generated_rows = generated_map[key]

            validation_metrics = aggregate_reference_metrics(
                validation_rows,
                generated_rows,
                train_rows,
                test_rows,
                seed_offset=seed * 1000 + int(severity * 10),
            )
            test_metrics = aggregate_reference_metrics(
                test_rows,
                generated_rows,
                train_rows,
                test_rows,
                seed_offset=seed * 2000 + int(severity * 10),
            )
            per_condition[key] = {
                "validation": validation_metrics,
                "test": test_metrics,
                "screen": screen(test_metrics),
            }

    domain_aggregate = {}
    for domain in ChallengeDomain:
        train_rows = [a for a in train if a.domain == domain]
        test_rows = [a for a in test if a.domain == domain]
        generated_rows = [a for a in generated if a.domain == domain]

        metrics = aggregate_reference_metrics(
            test_rows,
            generated_rows,
            train_rows,
            test_rows,
            seed_offset=seed * 3000 + len(domain.value),
            discriminator_seed=seed,
        )
        metrics["screen"] = screen(metrics)
        domain_aggregate[domain.value] = metrics

    return {
        "model_seed": seed,
        "generated_count": len(generated),
        "generated_ids_unique": len({a.agent_id for a in generated}) == len(generated),
        "generated_object_hash": sha256_json([
            {
                "id": a.agent_id,
                "domain": a.domain.value,
                "severity": a.severity,
                "phenotype": phenotype_vector(a).round(8).tolist(),
            }
            for a in generated
        ]),
        "generated_vector_hash": sha256_json([
            phenotype_vector(a).round(8).tolist() for a in generated
        ]),
        "overall_duplicates": duplicate_metrics(generated),
        "by_condition": per_condition,
        "domain_aggregate": domain_aggregate,
        "severity_response": severity_response(test_map, generated_map),
    }


def ml_reproducibility(train_agents):
    from cardiagent.ml import AgentGeneratorModel

    def run(seed):
        model = AgentGeneratorModel(seed=seed, latent_dim=12)
        model.fit(
            train_agents,
            epochs=50,
            learning_rate=2e-3,
            beta=0.015,
            batch_size=BATCH_SIZE,
            verbose=False,
        )
        generated = model.sample(
            domain=ChallengeDomain.ISCHEMIC,
            severity=0.6,
            difficulty=0.6,
            count=32,
            agent_id_prefix="REPRO",
        )
        return sha256_json([
            phenotype_vector(a).round(10).tolist() for a in generated
        ])

    same_a = run(REPRO_SEED)
    same_b = run(REPRO_SEED)
    different = run(REPRO_SEED + 1)

    return {
        "same_seed_hash_a": same_a,
        "same_seed_hash_b": same_b,
        "different_seed_hash": different,
        "same_seed_identical": same_a == same_b,
        "different_seed_differs": same_a != different,
        "status": "PASS" if same_a == same_b and same_a != different else "FAIL",
    }


def cross_seed_summary(model_results):
    rows = []
    for domain in ChallengeDomain:
        vectors = []
        for result in model_results:
            vectors.append(np.asarray(
                result["domain_aggregate"][domain.value]["generated_feature_means"],
                dtype=float,
            ))
        pairwise = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pairwise.append(float(np.mean(np.abs(vectors[i] - vectors[j]))))
        rows.append({
            "domain": domain.value,
            "mean_absolute_feature_mean_difference_across_seeds": (
                float(np.mean(pairwise)) if pairwise else None
            ),
            "max_absolute_feature_mean_difference_across_seeds": (
                float(np.max(pairwise)) if pairwise else None
            ),
        })
    return rows


def aggregate_screen(model_results):
    failed_domains = []
    failed_conditions = []
    discriminator_failures = []

    for result in model_results:
        for domain, metrics in result["domain_aggregate"].items():
            if metrics["screen"]["status"] != "PASS":
                failed_domains.append({
                    "model_seed": result["model_seed"],
                    "domain": domain,
                    "failed_checks": [
                        key for key, passed in metrics["screen"]["checks"].items()
                        if not passed
                    ],
                })

            auc = metrics["discriminator"].get("mean_auc")
            if auc is not None and auc > SCREEN["max_discriminator_auc"]:
                discriminator_failures.append({
                    "model_seed": result["model_seed"],
                    "domain": domain,
                    "mean_auc": auc,
                })

        for condition, metrics in result["by_condition"].items():
            if metrics["screen"]["status"] != "PASS":
                failed_conditions.append({
                    "model_seed": result["model_seed"],
                    "condition": condition,
                    "failed_checks": [
                        key for key, passed in metrics["screen"]["checks"].items()
                        if not passed
                    ],
                })

    return {
        "failed_domain_count": len(failed_domains),
        "failed_condition_count": len(failed_conditions),
        "discriminator_failure_count": len(discriminator_failures),
        "failed_domains": failed_domains,
        "failed_conditions": failed_conditions[:200],
        "discriminator_failures": discriminator_failures,
        "status": (
            "PASS"
            if not failed_domains and not failed_conditions and not discriminator_failures
            else "FAIL"
        ),
    }


def write_condition_csv(report):
    path = OUT / "condition_metrics.csv"
    rows = []

    for result in report["models"]:
        for condition, metrics in result["by_condition"].items():
            summary = metrics["test"]["summary"]
            novelty = metrics["test"]["novelty"]
            rows.append({
                "model_seed": result["model_seed"],
                "condition": condition,
                "max_standardized_mean_bias": summary["max_standardized_mean_bias"],
                "min_variance_ratio": summary["min_variance_ratio"],
                "max_variance_ratio": summary["max_variance_ratio"],
                "max_ks_d": summary["max_ks_d"],
                "mean_abs_correlation_error": summary["mean_abs_correlation_error"],
                "constant_support_violations": summary["constant_support_violations"],
                "novelty_ratio": novelty.get("novelty_ratio_generated_over_heldout"),
                "screen_status": metrics["screen"]["status"],
            })

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report):
    overall = report["overall"]
    lines = [
        "# CardiAgent CVAE Scientific Validation v2",
        "",
        "## Scope",
        "",
        "Internal computational validation of the CVAE against held-out cases generated by the deterministic CardiAgent distribution. This is not external biological validation.",
        "",
        "## Design",
        "",
        f"- Train/validation/test: {TRAIN_FRACTION:.0%}/{VALIDATION_FRACTION:.0%}/{TEST_FRACTION:.0%}",
        f"- Domain × severity strata: {len(ChallengeDomain) * len(SEVERITIES)}",
        f"- Cases per stratum: {N_PER_STRATUM}",
        f"- Model seeds: {MODEL_SEEDS}",
        f"- CVAE epochs: {ML_EPOCHS}",
        "- Final evaluation is performed against an untouched test split.",
        "- Generated and held-out cases are matched by domain and severity.",
        "",
        "## Predeclared screening targets",
        "",
        "```json",
        json.dumps(SCREEN, indent=2),
        "```",
        "",
        "## Result",
        "",
        f"**Overall screening status: {overall['status']}**",
        "",
        f"Failed domain screens: {overall['aggregate_screen']['failed_domain_count']}",
        f"Failed condition screens: {overall['aggregate_screen']['failed_condition_count']}",
        f"Discriminator failures: {overall['aggregate_screen']['discriminator_failure_count']}",
        f"ML reproducibility: {report['ml_reproducibility']['status']}",
        "",
        "The screening targets are explicit heuristics, not claims that every valid biological simulator must satisfy them. Raw metrics should be interpreted together with the model purpose and, ultimately, independent empirical data.",
    ]
    (OUT / "scientific_validation_report.md").write_text("\n".join(lines) + "\n")


def main():
    start = time.time()

    report = {
        "schema_version": "2.0",
        "repository": "Virelion-Biotech/Virelion-CardiAgent",
        "validation_type": "Held-out conditional CVAE generative validation",
        "configuration": {
            "data_seed": DATA_SEED,
            "split_seed": SPLIT_SEED,
            "model_seeds": MODEL_SEEDS,
            "severities": SEVERITIES,
            "n_per_stratum": N_PER_STRATUM,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "ml_epochs": ML_EPOCHS,
            "batch_size": BATCH_SIZE,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "screen": SCREEN,
        },
        "environment": environment_report(),
    }

    print("[1/8] Generate stratified population and split")
    strata = generate_population()
    train, validation, test, manifest = split_population(strata)
    report["split_integrity"] = verify_split_integrity(train, validation, test)
    report["split_manifest_hash"] = sha256_json(manifest)

    if report["split_integrity"]["status"] != "PASS":
        raise RuntimeError("Split integrity failed")

    with open(OUT / "split_manifest.json", "w") as handle:
        json.dump(canonical(manifest), handle, indent=2)

    print("[2/8] Deterministic generator probe")
    probe_a = ChallengeGenerator(seed=DATA_SEED)
    probe_b = ChallengeGenerator(seed=DATA_SEED)
    probe_a_rows = [
        probe_a.generate(ChallengeDomain.ISCHEMIC, severity=0.6, difficulty=0.6)
        for _ in range(30)
    ]
    probe_b_rows = [
        probe_b.generate(ChallengeDomain.ISCHEMIC, severity=0.6, difficulty=0.6)
        for _ in range(30)
    ]
    report["deterministic_generator"] = {
        "hash_a": sha256_json([
            phenotype_vector(a).round(10).tolist() for a in probe_a_rows
        ]),
        "hash_b": sha256_json([
            phenotype_vector(a).round(10).tolist() for a in probe_b_rows
        ]),
    }
    report["deterministic_generator"]["status"] = (
        "PASS"
        if report["deterministic_generator"]["hash_a"]
        == report["deterministic_generator"]["hash_b"]
        else "FAIL"
    )

    print("[3/8] Train multiple CVAE seeds")
    models, training = train_models(train)
    report["training"] = training

    print("[4/8] ML same-seed reproducibility")
    report["ml_reproducibility"] = ml_reproducibility(train)

    print("[5/8] Held-out validation/test generation")
    model_results = []
    for model, seed in zip(models, MODEL_SEEDS):
        print(f"  evaluating model seed={seed}")
        model_results.append(
            evaluate_model(model, seed, train, validation, test)
        )
    report["models"] = model_results

    print("[6/8] Cross-seed stability and aggregate screening")
    report["cross_seed_summary"] = cross_seed_summary(model_results)
    report["overall"] = {
        "aggregate_screen": aggregate_screen(model_results),
        "status": aggregate_screen(model_results)["status"],
    }

    print("[7/8] Write reports")
    report["elapsed_seconds"] = time.time() - start

    with open(OUT / "scientific_validation_report.json", "w") as handle:
        json.dump(canonical(report), handle, indent=2)

    write_condition_csv(report)
    write_markdown(report)

    summary = [
        "VIRELION CARDIAGENT CVAE SCIENTIFIC VALIDATION V2",
        "================================================",
        f"Git commit: {report['environment'].get('git_commit')}",
        f"Train/validation/test: {len(train)}/{len(validation)}/{len(test)}",
        f"Model seeds: {MODEL_SEEDS}",
        f"Overall screen: {report['overall']['status']}",
        f"ML reproducibility: {report['ml_reproducibility']['status']}",
        f"Failed domains: {report['overall']['aggregate_screen']['failed_domain_count']}",
        f"Failed conditions: {report['overall']['aggregate_screen']['failed_condition_count']}",
        f"Discriminator failures: {report['overall']['aggregate_screen']['discriminator_failure_count']}",
        f"Elapsed seconds: {report['elapsed_seconds']:.2f}",
        "",
        "This is internal computational generative validation, not external biological validation.",
    ]
    (OUT / "scientific_validation_summary.txt").write_text(
        "\n".join(summary) + "\n"
    )

    print("[8/8] Complete")
    print("Results:", OUT.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error = {
            "status": "ERROR",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "environment": environment_report(),
        }
        with open(OUT / "validation_error.json", "w") as handle:
            json.dump(canonical(error), handle, indent=2)
        raise
