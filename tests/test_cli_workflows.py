import json
from importlib.resources import files

import jsonschema
import pytest

from cardiagent.cli import main


def load(path):
    return json.loads(path.read_text())


def test_core_cli_roundtrip(tmp_path, capsys):
    case, population = tmp_path / 'case.json', tmp_path / 'population.json'
    main(['doctor'])
    assert json.loads(capsys.readouterr().out)['status'] == 'ok'
    main(['generate', '--domain', 'ischemic', '--output', str(case)])
    main(['manifest', '--domain', 'ischemic', '--count', '4', '--output', str(population)])
    public, truth = tmp_path / 'public.json', tmp_path / 'truth.json'
    main(['blind', '--input', str(population), '--public', str(public), '--truth', str(truth)])
    assert len(load(public)['cases']) == 4
    outcomes = tmp_path / 'outcomes.json'
    outcomes.write_text(json.dumps([{'case_id': load(public)['cases'][0]['case_id'], 'predicted_domain': None}]))
    adapted = tmp_path / 'adapted.json'
    main(['adapt', '--input', str(population), '--outcomes', str(outcomes), '--count', '3', '--output', str(adapted)])
    assert load(adapted)['challenge_count'] == 3
    assert all('temporal_profile' not in c['metadata'] for c in load(adapted)['challenges'])
    for blind in (False, True):
        handoff = tmp_path / 'handoff.json'
        main(['handoff', '--input', str(case), '--output', str(handoff)] + (['--blind'] if blind else []))
        if blind:
            schema = json.loads(files('cardiagent').joinpath('schemas/cardi-vex-blind-handoff.schema.json').read_text())
            jsonschema.validate(load(handoff), schema)
    main(['benchmark', '--output-dir', str(tmp_path / 'benchmarks')])
    assert len(list((tmp_path / 'benchmarks').glob('*.json'))) >= 8


@pytest.mark.parametrize('args', [
    ['generate', '--domain', 'ischemic', '--severity', 'nan'],
    ['manifest', '--domain', 'ischemic', '--count', '0'],
    ['benchmark', '--suite', 'all', '--seed', '1'],
    ['generate', '--domain', 'ischemic', '--count', '2'],
])
def test_cli_errors(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_ml_cli_roundtrip(tmp_path):
    pytest.importorskip('torch')
    population, model, samples = [tmp_path / name for name in ('population.json', 'model.pt', 'samples.json')]
    main(['manifest', '--domain', 'ischemic', '--count', '12', '--output', str(population)])
    main(['train', '--input', str(population), '--output', str(model), '--epochs', '2'])
    main(['sample', '--model', str(model), '--domain', 'ischemic', '--count', '5', '--output', str(samples)])
    assert load(samples)['challenge_count'] == 5
    assert len({a['agent_id'] for a in load(samples)['challenges']}) == 5
