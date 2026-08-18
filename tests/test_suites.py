from cardiagent import ChallengeDomain, build_all_suites, build_suite


def test_all_canonical_suites_build_with_expected_case_count():
    suites = build_all_suites(per_domain=2, seed=13)
    assert set(suites) == {"baseline", "cross_domain", "overlap", "temporal", "observation", "stress"}
    assert all(benchmark.cases for benchmark in suites.values())
    assert all(len(benchmark.cases) == 2 * len(ChallengeDomain) for benchmark in suites.values())


def test_suite_is_reproducible_and_has_suite_metadata():
    a = build_suite("overlap", per_domain=2, seed=21)
    b = build_suite("overlap", per_domain=2, seed=21)

    assert a.public_json() == b.public_json()
    assert a.cases[0].presentation["observation_characteristics"]["measurement_noise"] >= 0.0


def test_observation_suite_increases_partial_observation():
    benchmark = build_suite("observation", domains=[ChallengeDomain.ISCHEMIC], per_domain=3, seed=3)
    rates = [case.presentation["observation_characteristics"]["measurement_noise"] for case in benchmark.cases]
    missingness = [case.presentation["observation_characteristics"]["heterogeneity"] for case in benchmark.cases]
    assert min(rates) >= 0.15
    assert min(missingness) >= 0.0
