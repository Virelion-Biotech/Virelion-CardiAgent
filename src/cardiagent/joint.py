"""Conditional Gaussian rank copula with within-support quantile interpolation.

A separate numerical family; no CVAE recovery, mechanism or patient realism claim.
Gaussian copulas do not universally capture nonlinear dependence, tail dependence,
structural constraints or privacy. All generated agents remain unqualified.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
from scipy.spatial import Delaunay
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from .conditional import FIELDS
from .ml import PHENOTYPE_FIELDS
from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile
from .serialization import finite_number, positive_integer, read_json, strict_json, write_json


class ConditionalCopulaGenerator:
    VERSION = "0.6-conditional-gaussian-rank-copula"

    def __init__(self, seed=0, shrinkage=0.02):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        finite_number(shrinkage, "shrinkage", low=0.001, high=1)
        self.seed = seed
        self.shrinkage = shrinkage
        self._rng = random.Random(seed)
        self._counter = 0
        self.groups = {}
        self.training_summary = {}
        self._supports = {}

    def _build_supports(self):
        supports = {}
        for domain in sorted({key[0] for key in self.groups}):
            keys = sorted(key for key in self.groups if key[0] == domain)
            points = np.array([key[1:] for key in keys])
            origin = points.mean(axis=0)
            _, singular, basis = np.linalg.svd(points - origin, full_matrices=False)
            rank = int(np.count_nonzero(singular > 1e-10))
            axes = basis[:rank].T
            projected = (points - origin) @ axes
            supports[domain] = (
                keys,
                origin,
                axes,
                projected,
                Delaunay(projected) if rank == 2 else None,
            )
        self._supports = supports

    def fit(self, agents):
        rows = [ChallengeAgent.from_dict(row.to_dict()) for row in agents]
        if len({row.agent_id for row in rows}) != len(rows):
            raise ValueError("Training IDs must be unique")
        grouped = {}
        for row in rows:
            difficulty = row.metadata.get(
                "requested_difficulty", row.metadata.get("difficulty", row.severity)
            )
            finite_number(difficulty, "training difficulty", low=0, high=1)
            key = (row.domain.value, row.severity, difficulty)
            grouped.setdefault(key, []).append(
                [
                    getattr(row.phenotype, f) if f in PHENOTYPE_FIELDS else getattr(row, f)
                    for f in FIELDS
                ]
            )
        if not grouped or len(grouped) > 1000 or any(len(rows) < 32 for rows in grouped.values()):
            raise ValueError("Require 1..1000 strata with at least 32 unique training rows each")
        groups = {}
        for key, values in grouped.items():
            matrix = np.array(values)
            n = len(matrix)
            scores = ndtri((rankdata(matrix, axis=0, method="average") - 0.5) / n)
            centered = scores - scores.mean(axis=0)
            norms = np.linalg.norm(centered, axis=0)
            active = norms > 1e-12
            standardized = np.divide(centered, norms, out=np.zeros_like(centered), where=active)
            correlation = standardized.T @ standardized
            np.fill_diagonal(correlation, 1)
            correlation = (1 - self.shrinkage) * correlation + self.shrinkage * np.eye(len(FIELDS))
            groups[key] = {
                "knots": np.sort(matrix, axis=0).T.tolist(),
                "correlation": correlation.tolist(),
            }
        self.groups = groups
        self._build_supports()
        self.training_summary = {
            "examples": len(rows),
            "strata": len(groups),
            "family": self.VERSION,
            "conditional_independence_assumed": False,
            "copula_assumption": "Gaussian normal-score dependence",
            "interpolation_assumption": "barycentric marginal quantiles and latent correlation within condition hull",
            "quality_status": "not_qualified",
            "patient_validated": False,
            "training_sha256": hashlib.sha256(
                strict_json([row.to_dict() for row in rows], sort_keys=True).encode()
            ).hexdigest(),
        }
        return self

    def _weights(self, domain, severity, difficulty):
        key = (ChallengeDomain(domain).value, severity, difficulty)
        if key in self.groups:
            return [(key, 1.0)]
        if key[0] not in self._supports:
            raise ValueError("Unseen domain; no extrapolation supported")
        keys, origin, axes, points, triangulation = self._supports[key[0]]
        offset = np.array([severity, difficulty]) - origin
        query = offset @ axes
        if np.linalg.norm(offset - query @ axes.T) > 1e-10:
            raise ValueError("Condition outside affine training support")
        if axes.shape[1] == 0:
            raise ValueError("Unseen condition; singleton training support")
        if axes.shape[1] == 1:
            order = np.argsort(points[:, 0])
            grid = points[order, 0]
            if query[0] < grid[0] or query[0] > grid[-1]:
                raise ValueError("Condition outside training hull")
            upper = int(np.searchsorted(grid, query[0], side="right"))
            upper = min(max(upper, 1), len(grid) - 1)
            lower = upper - 1
            weight = (query[0] - grid[lower]) / (grid[upper] - grid[lower])
            return [(keys[order[lower]], float(1 - weight)), (keys[order[upper]], float(weight))]
        simplex = int(triangulation.find_simplex(query, tol=1e-10))
        if simplex < 0:
            raise ValueError("Condition outside training hull")
        transform = triangulation.transform[simplex]
        weights = transform[:2] @ (query - transform[2])
        weights = np.r_[weights, 1 - weights.sum()]
        if np.min(weights) < -1e-10:
            raise ValueError("Invalid interpolation weights")
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        return [
            (keys[i], float(w))
            for i, w in zip(triangulation.simplices[simplex], weights, strict=True)
        ]

    def sample(self, *, domain, severity=0.5, difficulty=None, count=1, agent_id_prefix="J-CA"):
        finite_number(severity, "severity", low=0, high=1)
        difficulty = severity if difficulty is None else difficulty
        finite_number(difficulty, "difficulty", low=0, high=1)
        positive_integer(count, "count")
        if count > 100000:
            raise ValueError("Sample count exceeds 100000")
        if not isinstance(agent_id_prefix, str) or not agent_id_prefix.strip():
            raise ValueError("agent_id_prefix must be non-empty")
        weights = self._weights(domain, severity, difficulty)
        correlation = sum(
            weight * np.array(self.groups[key]["correlation"]) for key, weight in weights
        )
        factor = np.linalg.cholesky(correlation)
        normals = np.array([[self._rng.gauss(0, 1) for _ in FIELDS] for _ in range(count)])
        uniform = ndtr(normals @ factor.T)
        values = np.zeros_like(uniform)
        for key, weight in weights:
            knots = np.array(self.groups[key]["knots"])
            grid = (np.arange(knots.shape[1]) + 0.5) / knots.shape[1]
            for j, column in enumerate(knots):
                values[:, j] += weight * np.interp(uniform[:, j], grid, column)
        values = np.clip(values, 0, 1)
        result = []
        for vector in values:
            self._counter += 1
            result.append(
                ChallengeAgent(
                    agent_id=f"{agent_id_prefix}-{self.seed}-{self._counter}",
                    domain=ChallengeDomain(domain),
                    version=self.VERSION,
                    seed=self.seed,
                    severity=severity,
                    onset=float(vector[8]),
                    persistence=float(vector[9]),
                    heterogeneity=float(vector[10]),
                    phenotype=PhenotypeProfile(**dict(zip(PHENOTYPE_FIELDS, vector[:8]))),
                    metadata={
                        "generator": "conditional-gaussian-rank-copula",
                        "requested_domain": ChallengeDomain(domain).value,
                        "requested_severity": severity,
                        "requested_difficulty": difficulty,
                        "model_family": "conditional_gaussian_rank_copula",
                        "conditional_independence_assumed": False,
                        "copula_assumption": "Gaussian",
                        "representation": "phenotype-level",
                        "condition_interpolated": len(weights) > 1,
                        "support_weights": [
                            {"condition": list(key), "weight": w} for key, w in weights
                        ],
                        "quality_status": "not_qualified",
                        "patient_validated": False,
                        "temporal_profile_available": False,
                        "privacy_guarantee": False,
                    },
                )
            )
        return result

    def save(self, path):
        if not self.groups:
            raise RuntimeError("Cannot save untrained generator")
        return write_json(
            {
                "version": self.VERSION,
                "seed": self.seed,
                "shrinkage": self.shrinkage,
                "counter": self._counter,
                "random_state": self._rng.getstate(),
                "training_summary": self.training_summary,
                "groups": [
                    {"key": list(key), **value} for key, value in sorted(self.groups.items())
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = read_json(path)
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "version",
                "seed",
                "shrinkage",
                "counter",
                "random_state",
                "training_summary",
                "groups",
            }
            or payload["version"] != cls.VERSION
        ):
            raise ValueError("Invalid copula checkpoint version/fields")
        model = cls(payload["seed"], payload["shrinkage"])
        groups = {}
        if not isinstance(payload["groups"], list) or not 1 <= len(payload["groups"]) <= 1000:
            raise ValueError("Invalid checkpoint groups")
        for group in payload["groups"]:
            if not isinstance(group, dict) or set(group) != {"key", "knots", "correlation"}:
                raise ValueError("Invalid checkpoint group fields")
            raw = group["key"]
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError("Invalid condition key")
            finite_number(raw[1], "severity", low=0, high=1)
            finite_number(raw[2], "difficulty", low=0, high=1)
            key = (ChallengeDomain(raw[0]).value, *raw[1:])
            knots = group["knots"]
            if (
                key in groups
                or not isinstance(knots, list)
                or len(knots) != len(FIELDS)
                or any(not isinstance(k, list) for k in knots)
                or len({len(k) for k in knots}) != 1
                or len(knots[0]) < 32
            ):
                raise ValueError("Invalid checkpoint quantiles")
            for column in knots:
                for value in column:
                    finite_number(value, "quantile", low=0, high=1)
                if column != sorted(column):
                    raise ValueError("Checkpoint quantiles must be sorted")
            raw_correlation = group["correlation"]
            if (
                not isinstance(raw_correlation, list)
                or len(raw_correlation) != len(FIELDS)
                or any(
                    not isinstance(row, list) or len(row) != len(FIELDS) for row in raw_correlation
                )
            ):
                raise ValueError("Invalid correlation dimensions")
            for row in raw_correlation:
                for value in row:
                    finite_number(value, "correlation", low=-1, high=1)
            matrix = np.array(raw_correlation)
            if (
                not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                or not np.allclose(np.diag(matrix), 1, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(matrix).min() < model.shrinkage - 1e-10
            ):
                raise ValueError(
                    "Correlation must be symmetric, unit diagonal and shrinkage-positive definite"
                )
            groups[key] = {"knots": knots, "correlation": raw_correlation}
        summary = payload["training_summary"]
        if (
            not isinstance(summary, dict)
            or summary.get("family") != cls.VERSION
            or summary.get("quality_status") != "not_qualified"
            or summary.get("patient_validated") is not False
            or summary.get("conditional_independence_assumed") is not False
            or summary.get("strata") != len(groups)
            or summary.get("examples") != sum(len(g["knots"][0]) for g in groups.values())
        ):
            raise ValueError("Checkpoint summary conflicts with model scope or training sizes")
        model.groups = groups
        model.training_summary = summary
        model._build_supports()
        model._counter = positive_integer(payload["counter"], "counter", minimum=0)

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        model._rng.setstate(tuples(payload["random_state"]))
        return model
