import pytest


torch = pytest.importorskip("torch")

from cardiagent import AgentGeneratorModel, ChallengeDomain, ChallengeGenerator  # noqa: E402


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


def test_ml_random_state_isolated_and_checkpoint_resumes_sequence(tmp_path):
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    model = AgentGeneratorModel(seed=7).fit(_training_set(), epochs=1)
    assert torch.equal(before, torch.random.get_rng_state())
    first = model.sample(domain=ChallengeDomain.ISCHEMIC, count=2)
    checkpoint = tmp_path / "nested" / "model.pt"
    model.save(checkpoint)
    expected = model.sample(domain=ChallengeDomain.ISCHEMIC, count=2)
    restored = AgentGeneratorModel.load(checkpoint)
    observed = restored.sample(domain=ChallengeDomain.ISCHEMIC, count=2)
    assert [a.to_dict() for a in expected] == [a.to_dict() for a in observed]
    assert not {a.agent_id for a in first} & {a.agent_id for a in observed}


@pytest.mark.parametrize(
    "field,value", [("beta", float("nan")), ("learning_rate", float("inf")), ("epochs", True)]
)
def test_ml_invalid_training_controls_rejected(field, value):
    with pytest.raises(ValueError):
        AgentGeneratorModel(seed=7).fit(_training_set(), **{field: value})
