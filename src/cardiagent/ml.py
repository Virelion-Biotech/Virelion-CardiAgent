"""Machine-learning generation of phenotype-level CardiAgent challenges.

This module adds a conditional variational autoencoder (CVAE) that learns the
empirical distribution of already-safe, phenotype-level ChallengeAgent objects
and samples new challenge agents from that learned distribution. It does not
model pathogens, sequences, protocols, doses, culture conditions, or other
operational biological parameters.

PyTorch is an optional dependency so the core deterministic generator remains
usable without ML dependencies.
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from .serialization import finite_number, positive_integer
from typing import Iterable, Sequence

from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile


PHENOTYPE_FIELDS: tuple[str, ...] = (
    "stress",
    "inflammation",
    "electrical_instability",
    "contractile_impairment",
    "viability_loss",
    "oxidative_stress",
    "metabolic_disruption",
    "remodeling_signal",
)

LATENT_DIM = 12
INPUT_DIM = 11


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "ML generation requires PyTorch. Install with: pip install 'virelion-cardiagent[ml]'"
        ) from exc
    return torch, nn


def _domain_index(domain: ChallengeDomain) -> int:
    return list(ChallengeDomain).index(domain)


def _feature_vector(agent: ChallengeAgent):
    torch, _ = _torch()
    values = [getattr(agent.phenotype, field) for field in PHENOTYPE_FIELDS]
    values.extend([agent.onset, agent.persistence, agent.heterogeneity])
    return torch.tensor(values, dtype=torch.float32)


class _ConditionalVAE:
    """Small conditional VAE used internally by AgentGeneratorModel."""

    def __init__(self, torch, nn, domain_count: int, latent_dim: int = LATENT_DIM):
        self.domain_count = domain_count
        self.latent_dim = latent_dim
        condition_dim = domain_count + 1
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, INPUT_DIM),
            nn.Sigmoid(),
        )
        self.torch = torch

    def parameters(self):
        for module in (self.encoder, self.mu, self.logvar, self.decoder):
            yield from module.parameters()

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "mu": self.mu.state_dict(),
            "logvar": self.logvar.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.mu.load_state_dict(state["mu"])
        self.logvar.load_state_dict(state["logvar"])
        self.decoder.load_state_dict(state["decoder"])

    def _condition(self, domain_ids, severity):
        one_hot = self.torch.nn.functional.one_hot(
            domain_ids, num_classes=self.domain_count
        ).float()
        return self.torch.cat([one_hot, severity[:, None]], dim=1)

    def encode(self, x, domain_ids, severity):
        condition = self._condition(domain_ids, severity)
        h = self.encoder(self.torch.cat([x, condition], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, domain_ids, severity):
        condition = self._condition(domain_ids, severity)
        return self.decoder(self.torch.cat([z, condition], dim=1))

    def forward(self, x, domain_ids, severity, generator=None):
        mu, logvar = self.encode(x, domain_ids, severity)
        std = self.torch.exp(0.5 * logvar)
        z = mu + self.torch.randn(std.shape, generator=generator) * std
        return self.decode(z, domain_ids, severity), mu, logvar


class AgentGeneratorModel:
    """Trainable ML model that produces new CardiAgent instances.

    The model learns a distribution from safe ChallengeAgent examples. Sampling
    is conditional on domain and severity; difficulty increases latent
    exploration and phenotype ambiguity. Returned objects enter the existing
    blinded CardiVex benchmark pipeline unchanged.
    """

    VERSION = "0.5-ml-cvae-support"

    def __init__(self, *, latent_dim: int = LATENT_DIM, seed: int = 0):
        torch, nn = _torch()
        positive_integer(latent_dim, "latent_dim")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.torch = torch
        self.seed = seed
        self.latent_dim = latent_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _ConditionalVAE(torch, nn, len(ChallengeDomain), latent_dim)
        self._rng = torch.Generator().manual_seed(seed)
        self._sample_counter = 0
        self.trained = False
        self.support_constants = {}
        self.checkpoint_version = self.VERSION
        self.training_summary: dict[str, float | int] = {}

    def fit(
        self,
        agents: Iterable[ChallengeAgent],
        *,
        epochs: int = 250,
        learning_rate: float = 2e-3,
        beta: float = 0.015,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> "AgentGeneratorModel":
        """Fit the model to phenotype-level challenge examples."""
        torch = self.torch
        rows = list(agents)
        if len(rows) < 8:
            raise ValueError(
                "At least 8 ChallengeAgent examples are required to train the ML generator"
            )
        positive_integer(epochs, "epochs")
        positive_integer(batch_size, "batch_size")
        finite_number(learning_rate, "learning_rate", low=0)
        finite_number(beta, "beta", low=0)
        if learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        self.trained = False
        self.checkpoint_version = self.VERSION
        x = torch.stack([_feature_vector(agent) for agent in rows])
        domains = torch.tensor([_domain_index(agent.domain) for agent in rows], dtype=torch.long)
        severity = torch.tensor([agent.severity for agent in rows], dtype=torch.float32)
        # Exact point masses cannot be represented by a sigmoid decoder.
        # Learn domain-level constants from training rows only, never from test data.
        self.support_constants = {}
        for domain_id in sorted(set(domains.tolist())):
            domain_rows = [row for row in rows if _domain_index(row.domain) == domain_id]
            fields = PHENOTYPE_FIELDS + ("onset", "persistence", "heterogeneity")
            columns = [
                [
                    getattr(row.phenotype, field)
                    if field in PHENOTYPE_FIELDS
                    else getattr(row, field)
                    for row in domain_rows
                ]
                for field in fields
            ]
            constants = [
                float(column[0]) if all(value == column[0] for value in column) else None
                for column in columns
            ]
            self.support_constants[domain_id] = constants
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        last_loss = 0.0

        for epoch in range(epochs):
            order = torch.randperm(len(rows), generator=self._rng)
            epoch_loss = 0.0
            for start in range(0, len(rows), batch_size):
                idx = order[start : start + batch_size]
                recon, mu, logvar = self.model.forward(
                    x[idx], domains[idx], severity[idx], generator=self._rng
                )
                reconstruction = torch.nn.functional.mse_loss(recon, x[idx])
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction + beta * kl
                if not torch.isfinite(loss):
                    raise RuntimeError("ML training produced non-finite loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
            last_loss = epoch_loss / len(rows)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0):
                print(f"epoch={epoch + 1} loss={last_loss:.6f}")

        self.trained = True
        self.training_summary = {
            "examples": len(rows),
            "epochs": epochs,
            "final_loss": last_loss,
            "learning_rate": learning_rate,
            "beta": beta,
        }
        return self

    def sample(
        self,
        *,
        domain: ChallengeDomain,
        severity: float = 0.5,
        difficulty: float = 0.75,
        count: int = 1,
        agent_id_prefix: str = "ML-CA",
    ) -> list[ChallengeAgent]:
        """Sample novel agents from the learned distribution."""
        torch = self.torch
        if not self.trained:
            raise RuntimeError("Train the model with .fit(...) before sampling")
        domain = ChallengeDomain(domain)
        finite_number(severity, "severity", low=0, high=1)
        finite_number(difficulty, "difficulty", low=0, high=1)
        if self.support_constants and _domain_index(domain) not in self.support_constants:
            raise ValueError("Requested domain was not observed during training")
        if not 0.0 <= severity <= 1.0 or not 0.0 <= difficulty <= 1.0:
            raise ValueError("severity and difficulty must be within [0, 1]")
        positive_integer(count, "count")
        if not isinstance(agent_id_prefix, str) or not agent_id_prefix.strip():
            raise ValueError("agent_id_prefix cannot be empty")
        if count < 1:
            raise ValueError("count must be positive")

        temperature = 0.65 + 0.95 * difficulty
        domain_id = torch.tensor([_domain_index(domain)] * count, dtype=torch.long)
        severity_tensor = torch.tensor([severity] * count, dtype=torch.float32)
        z = torch.randn(count, self.latent_dim, generator=self._rng) * temperature
        with torch.no_grad():
            decoded = self.model.decode(z, domain_id, severity_tensor).clamp(0.0, 1.0)
            for column, value in enumerate(self.support_constants.get(_domain_index(domain), [])):
                if value is not None:
                    decoded[:, column] = value

        agents: list[ChallengeAgent] = []
        for i, vector in enumerate(decoded.tolist(), start=1):
            for column, value in enumerate(self.support_constants.get(_domain_index(domain), [])):
                if value is not None:
                    vector[column] = value  # Preserve source precision, not float32 rounding.
            phenotype = PhenotypeProfile(**dict(zip(PHENOTYPE_FIELDS, vector[:8])))
            onset, persistence, heterogeneity = vector[8:11]
            overlap = min(
                1.0, 0.20 + 0.70 * difficulty + float(torch.rand(1, generator=self._rng)) * 0.10
            )
            neighbor = self._neighbor(domain, difficulty)
            metadata = {
                "requested_domain": domain.value,
                "requested_severity": severity,
                "requested_difficulty": difficulty,
                "generator": "virelion-cardiagent-ml",
                "generator_version": self.checkpoint_version,
                "support_mode": "training_domain_constants"
                if self.support_constants
                else "legacy_unconstrained",
                "quality_status": "not_qualified",
                "representation": "phenotype-level",
                "model_family": "conditional_variational_autoencoder",
                "latent_dim": self.latent_dim,
                "sampling_temperature": temperature,
                "difficulty": difficulty,
                "ml_generated": True,
                "phenotype_overlap": overlap,
                "overlap_reference": neighbor.value,
                "benchmark_intent": "independent_detection_and_characterization",
                "expected_observables": list(PHENOTYPE_FIELDS),
            }
            agents.append(
                ChallengeAgent(
                    agent_id=f"{agent_id_prefix}-{self.seed:06d}-{self._sample_counter + i:05d}",
                    domain=domain,
                    version=self.checkpoint_version,
                    seed=self.seed,
                    severity=severity,
                    onset=float(onset),
                    persistence=float(persistence),
                    heterogeneity=float(heterogeneity),
                    phenotype=phenotype,
                    metadata=metadata,
                )
            )
        self._sample_counter += count
        return agents

    @staticmethod
    def _neighbor(domain: ChallengeDomain, difficulty: float) -> ChallengeDomain:
        neighbors = {
            ChallengeDomain.ISCHEMIC: (ChallengeDomain.METABOLIC, ChallengeDomain.TOXIC_INJURY),
            ChallengeDomain.INFLAMMATORY: (
                ChallengeDomain.VIRAL_LIKE,
                ChallengeDomain.TOXIC_INJURY,
            ),
            ChallengeDomain.ELECTROPHYSIOLOGIC: (
                ChallengeDomain.GENETIC_SUSCEPTIBILITY,
                ChallengeDomain.METABOLIC,
            ),
            ChallengeDomain.TOXIC_INJURY: (ChallengeDomain.ISCHEMIC, ChallengeDomain.INFLAMMATORY),
            ChallengeDomain.VIRAL_LIKE: (ChallengeDomain.INFLAMMATORY, ChallengeDomain.METABOLIC),
            ChallengeDomain.METABOLIC: (
                ChallengeDomain.ISCHEMIC,
                ChallengeDomain.ELECTROPHYSIOLOGIC,
            ),
            ChallengeDomain.GENETIC_SUSCEPTIBILITY: (
                ChallengeDomain.ELECTROPHYSIOLOGIC,
                ChallengeDomain.METABOLIC,
            ),
        }
        options = neighbors[domain]
        return options[1 if difficulty >= 0.5 else 0]

    def save(self, path: str | Path) -> None:
        """Save model weights and training metadata."""
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.checkpoint_version,
            "checkpoint_format": 2,
            "support_constants": self.support_constants,
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "training_summary": self.training_summary,
            "state_dict": self.model.state_dict(),
            "rng_state": self._rng.get_state(),
            "sample_counter": self._sample_counter,
        }
        fd, temporary = tempfile.mkstemp(
            prefix="cardiagent-model-", suffix=".pt", dir=destination.parent
        )
        os.close(fd)
        try:
            self.torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "AgentGeneratorModel":
        """Load a previously trained model."""
        torch, _ = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("version") not in {cls.VERSION, "0.3-ml-cvae"}
            or payload.get("checkpoint_format", 1) not in {1, 2}
        ):
            raise ValueError("Unsupported checkpoint version/format")
        model = cls(latent_dim=payload["latent_dim"], seed=payload.get("seed", 0))
        model.checkpoint_version = payload["version"]
        constants = payload.get("support_constants", {})
        if not isinstance(constants, dict):
            raise ValueError("Invalid support constants")
        for domain_id, values in constants.items():
            if (
                isinstance(domain_id, bool)
                or not isinstance(domain_id, int)
                or not 0 <= domain_id < len(ChallengeDomain)
                or not isinstance(values, list)
                or len(values) != INPUT_DIM
            ):
                raise ValueError("Invalid support constants")
            for value in values:
                if value is not None:
                    finite_number(value, "support constant", low=0, high=1)
        if payload["version"] == cls.VERSION and payload.get("checkpoint_format") != 2:
            raise ValueError("Support-aware checkpoint requires format 2")
        if payload["version"] == cls.VERSION and not constants:
            raise ValueError("Support-aware checkpoint requires learned constants")
        model.support_constants = constants
        model.model.load_state_dict(payload["state_dict"])
        model.training_summary = dict(payload.get("training_summary", {}))
        if any(not torch.isfinite(p).all() for p in model.model.parameters()):
            raise ValueError("Checkpoint contains non-finite model weights")
        if "rng_state" in payload:
            model._rng.set_state(payload["rng_state"])
        model._sample_counter = positive_integer(
            payload.get("sample_counter", 0), "sample_counter", minimum=0
        )
        model.trained = True
        return model


def train_agent_model(
    agents: Sequence[ChallengeAgent],
    *,
    epochs: int = 250,
    seed: int = 0,
    **kwargs,
) -> AgentGeneratorModel:
    """Convenience function for training a CardiAgent ML generator."""
    return AgentGeneratorModel(seed=seed).fit(agents, epochs=epochs, **kwargs)


def generate_ml_agents(
    model: AgentGeneratorModel,
    *,
    domain: ChallengeDomain,
    severity: float,
    difficulty: float = 0.75,
    count: int = 16,
) -> list[ChallengeAgent]:
    """Generate a batch ready for ``build_blind_benchmark``."""
    return model.sample(
        domain=domain,
        severity=severity,
        difficulty=difficulty,
        count=count,
    )


def generate_ml_benchmark(
    model: AgentGeneratorModel,
    *,
    benchmark_id: str,
    domains: Sequence[ChallengeDomain] | None = None,
    severity: float = 0.6,
    difficulty: float = 0.8,
    per_domain: int = 16,
    seed: int = 0,
):
    """Generate ML-created cases and immediately package them as a blind benchmark."""
    from .benchmark import build_blind_benchmark

    selected = tuple(domains or tuple(ChallengeDomain))
    agents: list[ChallengeAgent] = []
    for domain in selected:
        agents.extend(
            model.sample(
                domain=domain,
                severity=severity,
                difficulty=difficulty,
                count=per_domain,
                agent_id_prefix=f"ML-{domain.value}",
            )
        )
    return build_blind_benchmark(agents, benchmark_id=benchmark_id, seed=seed)
