"""Colab scientific validation for the CardiAgent ML generator.

This script deliberately separates model fitting from final evaluation:

* 60% train / 20% validation / 20% held-out test within each domain x severity
  stratum.
* The final test set is never passed to ``AgentGeneratorModel.fit``.
* Generated cases are requested at the exact domain/severity represented by
  each held-out test stratum.
* Constant reference features are handled explicitly; they are never coerced
  into zero correlations with ``nan_to_num``.
* Multiple model seeds are evaluated to measure seed sensitivity.

The empirical source for this validation is the deterministic CardiAgent
challenge distribution itself. Therefore this is an internal generative
algorithm validation, not external biological validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

import numpy as np

from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


OUT = Path("validation_colab_results_v2")
OUT.mkdir(exist_ok=True)

DATA_SEED = 20260917
SPLIT_SEED = 20260918
MODEL_SEEDS = (17, 29, 41)
REPRO_SEED = 17
SEVERITIES = tuple(round(x, 1) for x in np.linspace(0.1, 0.9, 9))
N_PER_STRATUM = 45
TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20
TEST_FRAC = 0.20
ML_EPOCHS = 100
BATCH_SIZE = 128

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

# These are screening targets, not universal scientific laws. They are declared
# before reading the new results so the report cannot silently tune itself to
# the observed output.
SCREEN = {
    "max_abs_standardized_mean_bias": 0.50,
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
    if hasattr(obj, "__dict__"):
        return canonical(vars(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


def json_hash(obj):
    raw = json.dumps(
        canonical(obj),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def agent_key(agent):
    return (
        agent.agent_id,
        agent.domain.value,
        round(float(agent.severity), 12),
        json_hash(agent.phenotype),
    )


def phenotype_vector(agent):
    return np.asarray(
        [getattr(agent.phenotype, field) for field in PHENOTYPE_FIELDS],
        dtype=float,
    )


def burden_vector(X):
    return np.mean(X, axis=1)


def run_command(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def environment_report():
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
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


def generate_stratified_population():
    """Generate 45 cases per domain x severity stratum."""
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


def split_strata(strata):
    """Return disjoint train/validation/test populations within every stratum."""
    rng = np.random.default_rng(SPLIT_SEED)
    train = []
    validation = []
    test = []
    manifest = {}

    n_train = int(N_PER_STRATUM * TRAIN_FRAC)
    n_validation = int(N_PER_STRATUM * VALIDATION_FRAC)
    n_test = N_PER_STRATUM - n_train - n_validation

    if (n_train, n_validation, n_test) != (27, 9, 9):
        raise RuntimeError("Unexpected stratified split sizes")

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
    train_ids = {a.agent_id for a in train}
    validation_ids = {a.agent_id for a in validation}
    test_ids = {a.agent_id for a in test}

    intersections = {
        "train_validation": sorted(train_ids & validation_ids),
        "train_test": sorted(train_ids & test_ids),
        "validation_test": sorted(validation_ids & test_ids),
    }

    duplicate_total = (
        len(train) - len(train_ids)
        + len(validation) - len(validation_ids)
        + len(test) - len(test_ids)
    )

    return {
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "duplicate_id_count": duplicate_total,
        "intersections": {k: v[:20] for k, v in intersections.items()},
        "status": "PASS"
        if duplicate_total == 0 and all(not x for x in intersections.values())
        else "FAIL",
    }


def safe_corr(X):
    """Correlation matrix with NaN for undefined constant-feature pairs."""
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    corr = np.full((n_features, n_features), np.nan, dtype=float)

    std = np.std(X, axis=0, ddof=1) if len(X) > 1 else np.zeros(n_features)

    for i in range(n_features):
        if std[i] > 1e-12:
            corr[i, i] = 1.0

        for j in range(i + 1, n_features):
            if std[i] <= 1e-12 or std[j] <= 1e-12:
                continue
            value = np.corrcoef(X[:, i], X[:, j])[0, 1]
            corr[i, j] = value
            corr[j, i] = value

    return corr


def bootstrap_mean_diff_ci(x, y, seed, n_boot=400):
    """Bootstrap 95% CI for mean(y)-mean(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    if len(x) == 0 or len(y) == 0:
        return None

    draws = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        draws[i] = np.mean(yb) - np.mean(xb)

    return {
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "estimate": float(np.mean(y) - np.mean(x)),
    }


def feature_metrics(reference, generated, seed_offset=0):
    """Compute univariate fidelity without inventing statistics for constants."""
    from scipy.stats import ks_2samp, wasserstein_distance

    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)

    rows = []

    for i, field in enumerate(PHENOTYPE_FIELDS):
        ref = reference[:, i]
        gen = generated[:, i]
        ref_sd = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
        gen_sd = float(np.std(gen, ddof=1)) if len(gen) > 1 else 0.0
        ref_mean = float(np.mean(ref))
        gen_mean = float(np.mean(gen))

        row = {
            "feature": field,
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
                float((gen_sd**2) / (ref_sd**2))
                if ref_sd > 1e-12
                else None
            ),
            "reference_nonzero_rate": float(np.mean(np.abs(ref) > 1e-12)),
            "generated_nonzero_rate": float(np.mean(np.abs(gen) > 1e-12)),
            "bootstrap_mean_difference_95ci": bootstrap_mean_diff_ci(
                ref,
                gen,
                seed=SEED_CI(seed_offset, i),
            ),
        }

        if ref_sd <= 1e-12:
            row.update(
                {
                    "reference_constant": True,
                    "ks_d": None,
                    "ks_pvalue": None,
                    "wasserstein": float(
                        wasserstein_distance(ref, gen)
                    ),
                    "constant_support_violation": bool(
                        np.any(np.abs(gen - ref_mean) > 1e-8)
                    ),
                }
            )
        else:
            ks = ks_2samp(ref, gen, alternative="two-sided", method="auto")
            row.update(
                {
                    "reference_constant": False,
                    "ks_d": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein": float(wasserstein_distance(ref, gen)),
                    "constant_support_violation": False,
                }
            )

        rows.append(row)

    return rows


def SEED_CI(offset, feature_index):
    return int(SPLIT_SEED + 10000 * offset + feature_index)


def correlation_metrics(reference, generated):
    ref_corr = safe_corr(reference)
    gen_corr = safe_corr(generated)
    n = len(PHENOTYPE_FIELDS)

    valid_pairs = []
    undefined_reference_pairs = []
    undefined_generated_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            ref_value = ref_corr[i, j]
            gen_value = gen_corr[i, j]

            if np.isfinite(ref_value) and np.isfinite(gen_value):
                valid_pairs.append(abs(float(ref_value - gen_value)))
            elif not np.isfinite(ref_value):
                undefined_reference_pairs.append((PHENOTYPE_FIELDS[i], PHENOTYPE_FIELDS[j]))
            elif not np.isfinite(gen_value):
                undefined_generated_pairs.append((PHENOTYPE_FIELDS[i], PHENOTYPE_FIELDS[j]))

    return {
        "valid_pair_count": len(valid_pairs),
        "mean_abs_correlation_error": (
            float(np.mean(valid_pairs)) if valid_pairs else None
        ),
        "max_abs_correlation_error": (
            float(np.max(valid_pairs)) if valid_pairs else None
        ),
        "undefined_reference_pairs": undefined_reference_pairs,
        "undefined_generated_pairs": undefined_generated_pairs,
        "reference_correlation_matrix": np.where(
            np.isfinite(ref_corr), ref_corr, np.nan
        ).round(6).tolist(),
        "generated_correlation_matrix": np.where(
            np.isfinite(gen_corr), gen_corr, np.nan
        ).round(6).tolist(),
    }


def nearest_neighbor_distances(reference, generated, baseline):
    """Compare generated->train NN distance with held-out-test->train baseline."""
    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    scale = np.std(reference, axis=0, ddof=1)
    active = scale > 1e-12

    # If all features are constant, novelty is undefined rather than zero.
    if not np.any(active):
        return {
            "status": "UNDEFINED_ALL_REFERENCE_FEATURES_CONSTANT",
            "active_feature_count": 0,
        }

    scale = scale[active]
    ref = reference[:, active] / scale
    gen = generated[:, active] / scale
    base = baseline[:, active] / scale

    def nn_mean(A, B):
        values = []
        for start in range(0, len(A), 256):
            chunk = A[start:start + 256]
            d2 = (
                np.sum(chunk[:, None, :] ** 2, axis=2)
                + np.sum(B[None, :, :] ** 2, axis=2)
                - 2 * chunk @ B.T
            )
            d2 = np.maximum(d2, 0.0)
            values.extend(np.sqrt(np.min(d2, axis=1)))
        return float(np.mean(values))

    generated_to_train = nn_mean(gen, ref)
    baseline_to_train = nn_mean(base, ref)

    return {
        "status": "PASS",
        "active_feature_count": int(np.sum(active)),
        "generated_to_train_mean_nn": generated_to_train,
        "test_to_train_mean_nn": baseline_to_train,
        "novelty_ratio_generated_over_test": (
            float(generated_to_train / baseline_to_train)
            if baseline_to_train > 1e-12
            else None
        ),
    }


def duplicate_metrics(generated):
    vectors = [tuple(np.round(phenotype_vector(a), 6)) for a in generated]
    unique = len(set(vectors))
    return {
        "n": len(vectors),
        "unique_vectors_1e-6": unique,
        "duplicate_vector_count_1e-6": len(vectors) - unique,
        "unique_fraction_1e-6": float(unique / len(vectors)) if vectors else None,
    }


def discriminator_auc(reference, generated, seed):
    """Five-fold logistic discriminator; ~0.5 means difficult to distinguish."""
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

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        clf,
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
        "interpretation": "lower_is_better_for_generatability; 0.5 is ideal",
    }


def conditional_generation_matrix(model, domain, severity, count, prefix):
    return model.sample(
        domain=domain,
        severity=severity,
        difficulty=severity,
        count=count,
        agent_id_prefix=prefix,
    )


def summarize_domain_condition(
    reference_agents,
    generated_agents,
    train_agents,
    test_agents,
    seed_offset,
):
    reference = np.stack([phenotype_vector(a) for a in reference_agents])
    generated = np.stack([phenotype_vector(a) for a in generated_agents])
    train = np.stack([phenotype_vector(a) for a in train_agents])
    baseline_test = np.stack([phenotype_vector(a) for a in test_agents])

    univariate = feature_metrics(reference, generated, seed_offset=seed_offset)
    correlation = correlation_metrics(reference, generated)
    novelty = nearest_neighbor_distances(train, generated, baseline_test)

    active_reference = [
        row for row in univariate
        if not row["reference_constant"]
    ]

    standardized_biases = [
        row["standardized_mean_bias"]
        for row in active_reference
        if row["standardized_mean_bias"] is not None
    ]
    variance_ratios = [
        row["variance_ratio"]
        for row in active_reference
        if row["variance_ratio"] is not None
    ]
    ks_ds = [
        row["ks_d"]
        for row in active_reference
        if row["ks_d"] is not None
    ]

    return {
        "n_reference": len(reference_agents),
        "n_generated": len(generated_agents),
        "univariate": univariate,
        "correlation": correlation,
        "novelty": novelty,
        "duplicates": duplicate_metrics(generated_agents),
        "summary": {
            "max_standardized_mean_bias": (
                float(max(standardized_biases))
                if standardized_biases else None
            ),
            "median_standardized_mean_bias": (
                float(np.median(standardized_biases))
                if standardized_biases else None
            ),
            "min_variance_ratio": (
                float(min(variance_ratios)) if variance_ratios else None
            ),
            "max_variance_ratio": (
                float(max(variance_ratios)) if variance_ratios else None
            ),
            "max_ks_d": float(max(ks_ds)) if ks_ds else None,
            "mean_abs_correlation_error": correlation[
                "mean_abs_correlation_error"
            ],
            "constant_support_violations": int(
                sum(
                    row["constant_support_violation"]
                    for row in univariate
                )
            ),
        },
    }


def screen_domain_condition(metrics):
    s = metrics["summary"]
    n = metrics["novelty"]
    checks = {}

    checks["standardized_mean_bias"] = (
        s["max_standardized_mean_bias"] is None
        or s["max_standardized_mean_bias"]
        <= SCREEN["max_abs_standardized_mean_bias"]
    )
    checks["variance_ratio"] = (
        s["min_variance_ratio"] is None
        or (
            s["min_variance_ratio"] >= SCREEN["min_variance_ratio"]
            and s["max_variance_ratio"] <= SCREEN["max_variance_ratio"]
        )
    )
    checks["ks"] = (
        s["max_ks_d"] is None
        or s["max_ks_d"] <= SCREEN["max_ks_d"]
    )
    checks["correlation"] = (
        s["mean_abs_correlation_error"] is None
        or s["mean_abs_correlation_error"]
        <= SCREEN["max_mean_abs_correlation_error"]
    )
    checks["constant_support"] = s["constant_support_violations"] == 0

    ratio = n.get("novelty_ratio_generated_over_test")
    checks["novelty"] = (
        ratio is None
        or (
            ratio >= SCREEN["min_novelty_ratio"]
            and ratio <= SCREEN["max_novelty_ratio"]
        )
    )

    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def build_condition_maps(agents):
    result = {}
    for agent in agents:
        key = f"{agent.domain.value}|{agent.severity:.1f}"
        result.setdefault(key, []).append(agent)
    return result


def severity_response_metrics(test_map, generated_map):
    rows = []

    for domain in ChallengeDomain:
        severities = []
        test_burdens = []
        generated_burdens = []

        for severity in SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"
            test_rows = test_map[key]
            generated_rows = generated_map[key]

            Xt = np.stack([phenotype_vector(a) for a in test_rows])
            Xg = np.stack([phenotype_vector(a) for a in generated_rows])

            severities.append(severity)
            test_burdens.append(float(np.mean(burden_vector(Xt))))
            generated_burdens.append(float(np.mean(burden_vector(Xg))))

        test_slope, test_intercept = np.polyfit(
            severities,
            test_burdens,
            deg=1,
        )
        gen_slope, gen_intercept = np.polyfit(
            severities,
            generated_burdens,
            deg=1,
        )

        rows.append({
            "domain": domain.value,
            "test_slope": float(test_slope),
            "generated_slope": float(gen_slope),
            "slope_difference": float(gen_slope - test_slope),
            "test_intercept": float(test_intercept),
            "generated_intercept": float(gen_intercept),
            "severities": severities,
            "test_mean_burden": test_burdens,
            "generated_mean_burden": generated_burdens,
        })

    return rows


def fit_models(train_agents):
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


def generate_for_model(model):
    generated = []

    for domain in ChallengeDomain:
        for severity in SEVERITIES:
            count = 9
            prefix = f"V2-{domain.value}-{severity:.1f}"
            generated.extend(
                conditional_generation_matrix(
                    model,
                    domain,
                    severity,
                    count,
                    prefix,
                )
            )

    return generated


def evaluate_model(model, model_seed, train, validation, test):
    test_map = build_condition_maps(test)
    validation_map = build_condition_maps(validation)
    generated = generate_for_model(model)
    generated_map = build_condition_maps(generated)

    by_condition = {}
    for domain in ChallengeDomain:
        for severity in SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"

            train_rows = train_by_condition(train, domain, severity)
            validation_rows = validation_map[key]
            test_rows = test_map[key]
            generated_rows = generated_map[key]

            metrics = summarize_domain_condition(
                validation_rows,
                generated_rows,
                train_rows,
                test_rows,
                seed_offset=model_seed * 1000 + int(severity * 10),
            )

            metrics["test_set_metrics"] = summarize_domain_condition(
                test_rows,
                generated_rows,
                train_rows,
                test_rows,
                seed_offset=model_seed * 2000 + int(severity * 10),
            )

            metrics["screen"] = screen_domain_condition(
                metrics["test_set_metrics"]
            )

            by_condition[key] = metrics

    # Aggregate across all severity levels for each domain using the final
    # held-out test distribution.
    test_domain_map = {}
    gen_domain_map = {}
    train_domain_map = {}
    for domain in ChallengeDomain:
        test_domain_map[domain.value] = [a for a in test if a.domain == domain]
        gen_domain_map[domain.value] = [a for a in generated if a.domain == domain]
        train_domain_map[domain.value] = [a for a in train if a.domain == domain]

    domain_aggregate = {}
    for domain in ChallengeDomain:
        ref = test_domain_map[domain.value]
        gen = gen_domain_map[domain.value]
        tr = train_domain_map[domain.value]
        base = np.stack([phenotype_vector(a) for a in ref])
        g = np.stack([phenotype_vector(a) for a in gen])

        domain_metrics = summarize_domain_condition(
            ref,
            gen,
            tr,
            ref,
            seed_offset=model_seed * 3000 + len(domain.value),
        )

        domain_metrics["discriminator"] = discriminator_auc(
            base,
            g,
            seed=model_seed,
        )
        domain_metrics["screen"] = screen_domain_condition(domain_metrics)
        domain_aggregate[domain.value] = domain_metrics

    severity_rows = severity_response_metrics(
        test_map,
        generated_map,
    )

    seed_unique = len({json_hash([agent_key(a) for a in generated])})

    return {
        "model_seed": model_seed,
        "generated_count": len(generated),
        "by_condition": by_condition,
        "domain_aggregate": domain_aggregate,
        "severity_response": severity_rows,
        "overall_generated_duplicates": duplicate_metrics(generated),
        "generated_ids_unique": len({a.agent_id for a in generated}) == len(generated),
        "generated_vectors_hash": json_hash([
            phenotype_vector(a).round(8).tolist()
            for a in generated
        ]),
        "generated_object_hash": json_hash([agent_key(a) for a in generated]),
        "seed_unique_hash_count": seed_unique,
    }


def train_by_condition(agents, domain, severity):
    return [
        a
        for a in agents
        if a.domain == domain and abs(float(a.severity) - severity) < 1e-12
    ]


def reproducibility_check(train_agents):
    from cardiagent.ml import AgentGeneratorModel

    def train_once():
        model = AgentGeneratorModel(seed=REPRO_SEED, latent_dim=12)
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
        return model, generated

    model_a, gen_a = train_once()
    model_b, gen_b = train_once()

    hash_a = json_hash([phenotype_vector(a).round(10).tolist() for a in gen_a])
    hash_b = json_hash([phenotype_vector(a).round(10).tolist() for a in gen_b])

    diff_seed_model = AgentGeneratorModel(seed=REPRO_SEED + 1, latent_dim=12)
    diff_seed_model.fit(
        train_agents,
        epochs=50,
        learning_rate=2e-3,
        beta=0.015,
        batch_size=BATCH_SIZE,
        verbose=False,
    )
    gen_c = diff_seed_model.sample(
        domain=ChallengeDomain.ISCHEMIC,
        severity=0.6,
        difficulty=0.6,
        count=32,
        agent_id_prefix="REPRO2",
    )
    hash_c = json_hash([phenotype_vector(a).round(10).tolist() for a in gen_c])

    return {
        "same_seed_identical_hash": hash_a == hash_b,
        "same_seed_hash_a": hash_a,
        "same_seed_hash_b": hash_b,
        "different_seed_hash": hash_c,
        "different_seed_differs": hash_c != hash_a,
        "status": "PASS" if hash_a == hash_b and hash_c != hash_a else "FAIL",
        "note": (
            "Tests stochastic-training reproducibility at fixed seed; model
            parameters and generated sample vectors must match after rounding."
        ).replace("\n", " "),
    }


def aggregate_screen(results):
    domain_statuses = []
    discriminator_failures = []
    condition_failures = []

    for model_result in results:
        for domain, metrics in model_result["domain_aggregate"].items():
            domain_statuses.append(metrics["screen"]["status"])

            auc = metrics["discriminator"].get("mean_auc")
            if auc is not None and auc > SCREEN["max_discriminator_auc"]:
                discriminator_failures.append({
                    "model_seed": model_result["model_seed"],
                    "domain": domain,
                    "mean_auc": auc,
                })

        for key, metrics in model_result["by_condition"].items():
            if metrics["screen"]["status"] == "FAIL":
                condition_failures.append({
                    "model_seed": model_result["model_seed"],
                    "condition": key,
                    "failed_checks": [
                        name
                        for name, passed in metrics["screen"]["checks"].items()
                        if not passed
                    ],
                })

    return {
        "domain_aggregate_pass_count": domain_statuses.count("PASS"),
        "domain_aggregate_total": len(domain_statuses),
        "domain_aggregate_all_pass": all(x == "PASS" for x in domain_statuses),
        "discriminator_failures": discriminator_failures,
        "condition_failure_count": len(condition_failures),
        "condition_failures": condition_failures[:100],
        "status": (
            "PASS"
            if all(x == "PASS" for x in domain_statuses)
            and not discriminator_failures
            and not condition_failures
            else "FAIL"
        ),
    }


def write_csv(report):
    path = OUT / "condition_metrics.csv"
    rows = []

    for model_result in report["models"]:
        seed = model_result["model_seed"]
        for condition, metrics in model_result["by_condition"].items():
            summary = metrics["test_set_metrics"]["summary"]
            novelty = metrics["test_set_metrics"]["novelty"]
            rows.append({
                "model_seed": seed,
                "condition": condition,
                "max_standardized_mean_bias": summary[
                    "max_standardized_mean_bias"
                ],
                "min_variance_ratio": summary["min_variance_ratio"],
                "max_variance_ratio": summary["max_variance_ratio"],
                "max_ks_d": summary["max_ks_d"],
                "mean_abs_correlation_error": summary[
                    "mean_abs_correlation_error"
                ],
                "constant_support_violations": summary[
                    "constant_support_violations"
                ],
                "novelty_ratio": novelty.get(
                    "novelty_ratio_generated_over_test"
                ),
                "screen_status": metrics["screen"]["status"],
            })

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report):
    path = OUT / "scientific_validation_report.md"
    overall = report["overall"]

    lines = [
        "# CardiAgent CVAE Scientific Validation v2",
        "",
        "## Scope",
        "",
        "This is an internal generative-algorithm validation using the deterministic CardiAgent distribution as the source population. It is not external biological validation.",
        "",
        "## Split",
        "",
        f"- train: {TRAIN_FRAC:.0%}",
        f"- validation: {VALIDATION_FRAC:.0%}",
        f"- held-out test: {TEST_FRAC:.0%}",
        f"- strata: {len(ChallengeDomain) * len(SEVERITIES)} domain × severity strata",
        f"- cases per stratum: {N_PER_STRATUM}",
        f"- model seeds: {', '.join(map(str, MODEL_SEEDS))}",
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
        f"Domain aggregate screens passed: {overall['aggregate_screen']['domain_aggregate_pass_count']} / {overall['aggregate_screen']['domain_aggregate_total']}",
        f"Condition screen failures: {overall['aggregate_screen']['condition_failure_count']}",
        f"Discriminator failures: {len(overall['aggregate_screen']['discriminator_failures'])}",
        "",
        "## Interpretation",
        "",
        overall["interpretation"],
        "",
        "The report intentionally retains failures and undefined statistics. A failed generative screen is evidence that the current ML generator does not reproduce the held-out distribution closely enough under the declared targets; it is not converted into a PASS by post-hoc tolerance changes.",
    ]

    path.write_text("\n".join(lines) + "\n")


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
            "train_fraction": TRAIN_FRAC,
            "validation_fraction": VALIDATION_FRAC,
            "test_fraction": TEST_FRAC,
            "ml_epochs": ML_EPOCHS,
            "batch_size": BATCH_SIZE,
            "screen": SCREEN,
        },
        "environment": environment_report(),
    }

    print("[1/8] Generating stratified population")
    strata = generate_stratified_population()
    train, validation, test, split_manifest = split_strata(strata)
    report["split_integrity"] = verify_split_integrity(train, validation, test)
    report["split_manifest_hash"] = json_hash(split_manifest)

    if report["split_integrity"]["status"] != "PASS":
        raise RuntimeError("Train/validation/test split integrity failed")

    # Persist only IDs for reproducibility; the held-out phenotype values are
    # not necessary to prove the split and keeping this artifact small matters.
    with open(OUT / "split_manifest.json", "w") as handle:
        json.dump(canonical(split_manifest), handle, indent=2)

    print("[2/8] Deterministic generator checks")
    deterministic_generator = ChallengeGenerator(seed=DATA_SEED)
    deterministic_probe_a = [
        deterministic_generator.generate(
            ChallengeDomain.ISCHEMIC,
            severity=0.6,
            difficulty=0.6,
        )
        for _ in range(30)
    ]
    deterministic_generator = ChallengeGenerator(seed=DATA_SEED)
    deterministic_probe_b = [
        deterministic_generator.generate(
            ChallengeDomain.ISCHEMIC,
            severity=0.6,
            difficulty=0.6,
        )
        for _ in range(30)
    ]
    report["deterministic_generator"] = {
        "status": "PASS"
        if json_hash([agent_key(a) for a in deterministic_probe_a])
        == json_hash([agent_key(a) for a in deterministic_probe_b])
        else "FAIL"
    }

    print("[3/8] CVAE training")
    models, training_summaries = fit_models(train)
    report["training"] = training_summaries

    print("[4/8] Same-seed reproducibility")
    report["ml_reproducibility"] = reproducibility_check(train)

    print("[5/8] Held-out validation/test evaluation")
    model_results = []
    for i, model in enumerate(models):
        print(f"  model seed={MODEL_SEEDS[i]}")
        model_results.append(
            evaluate_model(
                model,
                MODEL_SEEDS[i],
                train,
                validation,
                test,
            )
        )
    report["models"] = model_results

    print("[6/8] Aggregate screening")
    report["overall"] = {
        "aggregate_screen": aggregate_screen(model_results),
        "status": aggregate_screen(model_results)["status"],
        "interpretation": (
            "This run evaluates internal distributional generalization of the "
            "CVAE against held-out CardiAgent-generated cases. It does not "
            "establish external biological validity."
        ),
    }

    print("[7/8] Writing reports")
    report["elapsed_seconds"] = time.time() - start

    with open(OUT / "scientific_validation_report.json", "w") as handle:
        json.dump(canonical(report), handle, indent=2)

    write_csv(report)
    write_markdown(report)

    summary_lines = [
        "VIRELION CARDIAGENT CVAE SCIENTIFIC VALIDATION V2",
        "================================================",
        f"Git commit: {report['environment'].get('git_commit')}",
        f"Train/validation/test: {len(train)}/{len(validation)}/{len(test)}",
        f"Model seeds: {MODEL_SEEDS}",
        f"Overall screen: {report['overall']['status']}",
        f"ML reproducibility: {report['ml_reproducibility']['status']}",
        f"Condition failures: {report['overall']['aggregate_screen']['condition_failure_count']}",
        f"Discriminator failures: {len(report['overall']['aggregate_screen']['discriminator_failures'])}",
        f"Elapsed seconds: {report['elapsed_seconds']:.2f}",
        "",
        "IMPORTANT: this is internal computational generative validation, not external biological validation.",
    ]
    (OUT / "scientific_validation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
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
