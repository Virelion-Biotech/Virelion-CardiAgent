import io
import json

from cardiagent.hearttwin_adapter import main


def test_adapter_env_and_stdin(monkeypatch, capsys):
    payload = {'entity_id': 'test', 'observations': [{'modality': 'structural', 'values': {'unrelated': 1}}], 'context': {'cardiagent': {'domain': 'ischemic', 'count': 2}}}
    monkeypatch.setenv('HEARTTWIN_PAYLOAD', json.dumps(payload))
    assert main() == 0
    assert len(json.loads(capsys.readouterr().out)['challenges']) == 2
    monkeypatch.setenv('HEARTTWIN_PAYLOAD_STDIN', '1')
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(payload)))
    assert main() == 0
    assert json.loads(capsys.readouterr().out)['entity_id'] == 'test'
    monkeypatch.delenv('HEARTTWIN_PAYLOAD_STDIN')
    payload['context']['cardiagent']['count'] = 1.5
    monkeypatch.setenv('HEARTTWIN_PAYLOAD', json.dumps(payload))
    assert main() == 1
    assert capsys.readouterr().out == ''
    monkeypatch.delenv('HEARTTWIN_PAYLOAD')
    assert main() == 1
