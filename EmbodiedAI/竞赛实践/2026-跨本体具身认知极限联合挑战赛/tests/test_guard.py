"""CPU-only regression checks for credential hygiene and safe technical resume."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import subprocess

import run_guard
from run_guard import api_identity, runtime_identity, verify_resume

environment = {'VLM_BASE_URL': 'https://unit.test/compatible-mode/v1',
               'VLM_MODEL': 'fixture-model', 'VLM_API_KEY': 'UNIT_TEST_ONLY'}
with patch.dict(os.environ, environment):
    identity = api_identity()
    assert identity['model'] == 'fixture-model'
    assert 'UNIT_TEST_ONLY' not in json.dumps(identity)
    assert 'unit.test' not in json.dumps(identity)
    for base in ('https://your-provider.example/v1', 'http://unit.test/v1',
                 'https://user:password@unit.test/v1', 'https://unit.test/v1?key=private'):
        with patch.dict(os.environ, {'VLM_BASE_URL': base}):
            try:
                api_identity()
                raise AssertionError('unsafe or placeholder endpoint accepted')
            except ValueError:
                pass
with tempfile.TemporaryDirectory() as folder:
    work = Path(folder)
    config = work / 'runner.json'
    config.write_text('{}')
    (work / 'run-inputs.json').write_text(json.dumps({
        'identity': identity, 'runner_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}))
    with patch('run_guard.run_identity', return_value=identity):
        verify_resume(work, work, config)
        config.write_text('{"changed":true}')
        try:
            verify_resume(work, work, config)
            raise AssertionError('changed Runner config accepted')
        except ValueError:
            pass
    with patch('run_guard.run_identity', return_value={'api': {'model': 'changed'}}):
        try:
            verify_resume(work, work, config)
            raise AssertionError('changed code/model identity accepted')
        except ValueError:
            pass
print('PASS: API credential hygiene and rejection of changed resume inputs')

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    module = root / 'robolab/__init__.py'
    module.parent.mkdir()
    module.write_text('# fixture\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '-q')
    git('add', 'robolab')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    commit = git('rev-parse', 'HEAD')
    with patch.object(run_guard, 'RUNTIME_COMMIT', commit), \
         patch.object(run_guard.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(module))), \
         patch.object(run_guard.importlib.metadata, 'version', side_effect=run_guard.PACKAGES.__getitem__), \
         patch.dict(os.environ, {'OMNI_KIT_ACCEPT_EULA': 'Y'}):
        assert runtime_identity(root)['commit'] == commit
        module.write_text('# changed fixture\n')
        try:
            runtime_identity(root)
            raise AssertionError('modified Runtime accepted')
        except ValueError:
            pass
print('PASS: pinned Runtime checkout and rejection of local code changes (temporary Git fixture)')
