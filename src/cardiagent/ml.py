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

from dataclasses import replace
from pathlib import Path
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
INPUT_DIM = 11  # 8 phenotype values + onset + persistence + heterogeneity


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ML generation requires PyTorch. Install with: "
            "pip install 'virelion-cardiagent[ml]'"
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
    """Small conditional VAE used internally by :class:`AgentGeneratorModel`."""

    def __init__(self, torch, nn, domain_count: int, latent_dim: int = LATENT_DIM):
        self.domain_count = domain_count
        self.latent_dim = latent_dim
        condition_dim = domain_count + 1  # one-hot domain + severity

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
        self.nn = nn

    def parameters(self):
        modules = [self.encoder, self.mu, self.logvar, self.decoder]
        for module in modules:
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
        domain_one_hot = self.torch.nn.functional.one_hot(
            domain_ids, num_classes=self.domain_count
        ).float()
        return self.torch.cat([domain_one_hot, severity[:, None]], dim=1)

    def encode(self, x, domain_ids, severity):
        condition = self._condition(domain_ids, severity)
        h = self.encoder(self.torch.cat([x, condition], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, domain_ids, severity):
        condition = self._condition(domain_ids, severity)
        return self.decoder(self.torch.cat([z, condition], dim=1))

    def forward(self, x, domain_ids, severity):
        mu, logvar = self.encode(x, domain_ids, severity)
        std = self.torch.exp(0.5 * logvar)
        eps = self.torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z, domain_ids, severity), mu, logvar


class AgentGeneratorModel:
    """Trainable ML model that produces new CardiAgent instances.

    The model learns a distribution from a collection of safe
    ``ChallengeAgent`` examples. Sampling is conditional on challenge domain
    and severity, with ``difficulty`` controlling latent sampling temperature
    and cross-domain phenotype mixing. The returned objects are ordinary
    ``ChallengeAgent`` instances and can therefore enter the existing blinded
    CardiVex benchmark pipeline.
    """

    VERSION = "0.3-ml-cvae"

    def __init__(self, *, latent_dim: int = LATENT_DIM, seed: int = 0):
        torch, nn = _torch()
        self.torch = torch
        self.nn = nn
        self.seed = seed
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.model = _ConditionalVAE(torch, nn, len(ChallengeDomain), latent_dim)
        self.trained = False
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
            raise ValueError("At least 8 ChallengeAgent examples are required to train the ML generator")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        x = torch.stack([_feature_vector(agent) for agent in rows])
        domains = torch.tensor([_domain_index(agent.domain) for agent in rows], dtype=torch.long)
        severity = torch.tensor([agent.severity for agent in rows], dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        last_loss = 0.0
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.mu.train()
        self.model.logvar.train()

        for epoch in range(epochs):
            order = torch.randperm(len(rows))
            epoch_loss = 0.0
            for start in range(0, len(rows), batch_size):
                idx = order[start:start + batch_size]
                recon, mu, logvar = self.model.forward(x[idx], domains[idx], severity[idx])
                reconstruction = torch.nn.functional.mse_loss(recon, x[idx])
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction + beta * kl
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
        for name, value in (("severity", severity), ("difficulty", difficulty)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if count < 1:
            raise ValueError("count must be positive")

        self.model.encoder.eval()
        self.model.decoder.eval()
        domain_id = torch.tensor([_domain_index(domain)] * count, dtype=torch.long)
        severity_tensor = torch.tensor([severity] * count, dtype=torch.float32)

        # Higher difficulty explores farther from the learned center and
        # slightly increases ambiguity through latent temperature.
        temperature = 0.65 + 0.95 * difficulty
        z = torch.randn(count, self.latent_dim) * temperature
        with torch.no_grad():
            decoded = self.model.decode(z, domain_id, severity_tensor).clamp(0.0, 1.0)

        agents: list[ChallengeAgent] = []
        for i, vector in enumerate(decoded.tolist(), start=1):
            phenotype = PhenotypeProfile(**dict(zip(PHENOTYPE_FIELDS, vector[:8])))
            onset, persistence, heterogeneity = vector[8:11]
            overlap = min(1.0, 0.20 + 0.70 * difficulty + float(torch.rand(1)) * 0.10)
            neighbor = self._neighbor(domain, difficulty)
            metadata = {
                "generator": "virelion-cardiagent-ml",
                "generator_version": self.VERSION,
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
                    agent_id=f"{agent_id_prefix}-{self.seed:06d}-{i:05d}",
                    domain=domain,
                    version=self.VERSION,
                    seed=self.seed,
                    severity=severity,
                    onset=float(onset),
                    persistence=float(persistence),
                    heterogeneity=float(heterogeneity),
                    phenotype=phenotype,
                    metadata=metadata,
                )
            )
        return agents

    @staticmethod
    def _neighbor(domain: ChallengeDomain, difficulty: float) -> ChallengeDomain:
        neighbors = {
            ChallengeDomain.ISCHEMIC: (ChallengeDomain.METABOLIC, ChallengeDomain.TOXIC_INJURY),
            ChallengeDomain.INFLAMMATORY: (ChallengeDomain.VIRAL_LIKE, ChallengeDomain.TOXIC_INJURY),
            ChallengeDomain.ELECTROPHYSIOLOGIC: (ChallengeDomain.GENETIC_SUSCEPTIBILITY, ChallengeDomain.METABOLIC),
            ChallengeDomain.TOXIC_INJURY: (ChallengeDomain.ISCHEMIC, ChallengeDomain.INFLAMMATORY),
            ChallengeDomain.VIRAL_LIKE: (ChallengeDomain.INFLAMMATORY, ChallengeDomain.METABOLIC),
            ChallengeDomain.METABOLIC: (ChallengeDomain.ISCHEMIC, ChallengeDomain.ELECTROPHYSIOLOGIC),
            ChallengeDomain.GENETIC_SUSCEPTIBILITY: (ChallengeDomain.ELECTROPHYSIOLOGIC, ChallengeDomain.METABOLIC),
        }
        options = neighbors[domain]
        return options[1 if difficulty >= 0.5 else 0]

    def save(self, path: str | Path) -> None:
        """Save model weights and metadata."""
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model")
        payload = {
            "version": self.VERSION,
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "training_summary": self.training_summary,
            "state_dict": self.model.state_dict(),
        }
        self.torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "AgentGeneratorModel":
        """Load a previously trained model."""
        torch, _ = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(latent_dim=int(payload["latent_dim"]), seed=int(payload.get("seed", 0)))
        model.model.load_state_dict(payload["state_dict"])
        model.training_summary = dict(payload.get("training_summary", {}))
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
