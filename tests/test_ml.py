import pytest


torch = pytest.importorskip("torch")

from cardiagent import AgentGeneratorModel, ChallengeDomain, ChallengeGenerator


def _training_set():
    agents = []
    for seed in range(8):
        generator = ChallengeGenerator(seed=seed)
        for domain in ChallengeDomain:
            agents.append(
                generator.generate(
                    domain,
                    severity=0.2 + 0.1 * (seed % 6),
                    difficulty=0.3 + 0.1 * (seed % 5),
                )
            )
    return agents


def test_ml_model_trains_and_generates_agents():
    model = AgentGeneratorModel(seed=7)
    model.fit(_training_set(), epochs=2, batch_size=32)
    generated = model.sample(
        domain=ChallengeDomain.INFLAMMATORY,
        severity=0.8,
        difficulty=0.9,
        count=4,
    )

    assert len(generated) == 4
    assert all(agent.metadata["ml_generated"] for agent in generated)
    assert all(agent.version == AgentGeneratorModel.VERSION for agent in generated)
    assert all(0.0 <= agent.phenotype.inflammation <= 1.0 for agent in generated)
