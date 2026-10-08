"""CPU-only conditional marginal quantile baseline, not a neural generator.

Learns observed strata only. Preserves constant features and boundary atoms.
Assumes conditional feature independence; does not estimate a joint copula or
interpolate/extrapolate to unseen severity/difficulty settings.
"""

import random

from .ml import PHENOTYPE_FIELDS
from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile
from .serialization import finite_number, positive_integer, read_json, write_json

FIELDS = PHENOTYPE_FIELDS + ("onset", "persistence", "heterogeneity")


class ConditionalQuantileGenerator:
    VERSION = "0.5-conditional-marginal-quantile"

    def __init__(self, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0
        self.groups = {}
        self.training_summary = {}

    @staticmethod
    def _key(domain, severity, difficulty):
        return (ChallengeDomain(domain).value, severity, difficulty)

    def fit(self, agents):
        rows = list(agents)
        ids = [row.agent_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Training IDs must be unique")
        groups = {}
        for row in rows:
            difficulty = row.metadata.get(
                "requested_difficulty", row.metadata.get("difficulty", row.severity)
            )
            finite_number(difficulty, "training difficulty", low=0, high=1)
            key = self._key(row.domain, row.severity, difficulty)
            groups.setdefault(key, []).append(
                [
                    getattr(row.phenotype, f) if f in PHENOTYPE_FIELDS else getattr(row, f)
                    for f in FIELDS
                ]
            )
        if not groups or any(len(group) < 8 for group in groups.values()):
            raise ValueError("At least eight independent rows per observed stratum required")
        self.groups = {
            key: [sorted(column) for column in zip(*group)] for key, group in groups.items()
        }
        self.training_summary = {
            "examples": len(rows),
            "strata": len(groups),
            "family": self.VERSION,
            "conditional_independence_assumed": True,
            "quality_status": "not_qualified",
        }
        return self

    def sample(self, *, domain, severity=0.5, difficulty=None, count=1, agent_id_prefix="Q-CA"):
        finite_number(severity, "severity", low=0, high=1)
        difficulty = severity if difficulty is None else difficulty
        finite_number(difficulty, "difficulty", low=0, high=1)
        positive_integer(count, "count")
        if not isinstance(agent_id_prefix, str) or not agent_id_prefix.strip():
            raise ValueError("agent_id_prefix must be non-empty")
        key = self._key(domain, severity, difficulty)
        if key not in self.groups:
            raise ValueError(
                "Unseen domain/severity/difficulty stratum; no interpolation or extrapolation supported"
            )
        result = []
        for _ in range(count):
            vector = []
            for knots in self.groups[key]:
                # Midpoint empirical quantiles; end bins preserve boundary atoms.
                position = self._rng.random() * len(knots) - 0.5
                low = max(0, min(len(knots) - 1, int(position // 1)))
                high = max(0, min(len(knots) - 1, low + 1))
                value = (
                    knots[low]
                    if position <= 0
                    else knots[low] + min(1, position - low) * (knots[high] - knots[low])
                )
                vector.append(value)
            self._counter += 1
            result.append(
                ChallengeAgent(
                    agent_id=f"{agent_id_prefix}-{self.seed}-{self._counter}",
                    domain=ChallengeDomain(domain),
                    version=self.VERSION,
                    seed=self.seed,
                    severity=severity,
                    onset=vector[8],
                    persistence=vector[9],
                    heterogeneity=vector[10],
                    phenotype=PhenotypeProfile(**dict(zip(PHENOTYPE_FIELDS, vector[:8]))),
                    metadata={
                        "generator": "conditional-marginal-quantile",
                        "requested_domain": key[0],
                        "requested_severity": severity,
                        "requested_difficulty": difficulty,
                        "model_family": "conditional_marginal_quantile",
                        "conditional_independence_assumed": True,
                        "representation": "phenotype-level",
                        "quality_status": "not_qualified",
                        "temporal_profile_available": False,
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
                "counter": self._counter,
                "random_state": self._rng.getstate(),
                "training_summary": self.training_summary,
                "groups": [
                    {"key": list(key), "knots": knots} for key, knots in sorted(self.groups.items())
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = read_json(path)
        if payload.get("version") != cls.VERSION:
            raise ValueError("Unsupported quantile model version")
        model = cls(payload["seed"])
        groups = {}
        for group in payload["groups"]:
            raw = group["key"]
            finite_number(raw[1], "severity", low=0, high=1)
            finite_number(raw[2], "difficulty", low=0, high=1)
            key = cls._key(*raw)
            knots = group["knots"]
            if (
                key in groups
                or len(knots) != len(FIELDS)
                or len({len(k) for k in knots}) != 1
                or len(knots[0]) < 8
            ):
                raise ValueError("Invalid quantile model groups")
            for column in knots:
                for value in column:
                    finite_number(value, "quantile", low=0, high=1)
                if column != sorted(column):
                    raise ValueError("Quantiles must be sorted")
            groups[key] = knots
        if not groups:
            raise ValueError("Model groups must not be empty")
        model.groups = groups
        model._counter = positive_integer(payload["counter"], "counter", minimum=0)

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        model._rng.setstate(tuples(payload["random_state"]))
        model.training_summary = payload["training_summary"]
        return model
