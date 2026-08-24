from cardiagent import ChallengeDomain, available_suites, build_suite


def test_all_benchmark_suites_are_available_and_nonempty():
    names = available_suites()
    assert names == ("baseline", "difficulty", "severity", "overlap")
    for name in names:
        suite = build_suite(name)
        assert suite.name == name
        assert suite.version
        assert suite.case_count > 0
        assert all(challenge.domain in ChallengeDomain for challenge in suite.challenges)


def test_baseline_suite_is_reproducible():
    first = build_suite("baseline", seed=123)
    second = build_suite("baseline", seed=123)
    assert [case.to_dict() for case in first.challenges] == [case.to_dict() for case in second.challenges]


def test_difficulty_suite_has_multiple_difficulty_levels():
    suite = build_suite("difficulty", seed=456)
    levels = {round(float(case.metadata["difficulty"]), 3) for case in suite.challenges}
    assert len(levels) >= 3
