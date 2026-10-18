"""Check declared inputs before starting simulation; never store credentials."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

RUNTIME_COMMIT = '0aef241fb088ca21bb4ebd24448940ed56620d17'
PACKAGES = {'robolab': '0.2.1', 'isaacsim': '5.1.0.0',
            'isaaclab': '2.3.2.post1', 'numpy': '1.26.0',
            'torch': '2.7.0+cu128', 'requests': '2.32.3'}


def api_identity():
    for name in ('VLM_BASE_URL', 'VLM_MODEL', 'VLM_API_KEY'):
        value = os.environ.get(name, '').strip()
        if not value or value == 'REPLACE_WITH_YOUR_KEY':
            raise ValueError('Set ' + name + ' in the environment')
    base = os.environ['VLM_BASE_URL'].rstrip('/')
    url = urlsplit(base)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.endswith('/v1')
            or url.hostname.endswith(('.example', '.invalid'))):
        raise ValueError('VLM_BASE_URL must be your HTTPS /v1 endpoint, without credentials/query')
    return {'model': os.environ['VLM_MODEL'],
            'endpoint_sha256': hashlib.sha256(base.encode()).hexdigest()}


def runtime_identity(source):
    source = Path(source).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != source:
        raise ValueError('runtime-source must be the RoboLab Git checkout root')
    if git('rev-parse', 'HEAD') != RUNTIME_COMMIT:
        raise ValueError('Runtime commit mismatch; see REPRODUCE.md')
    if git('status', '--porcelain', '--untracked-files=no') or git('status', '--porcelain', '--untracked-files=all', '--',
           'robolab', 'policies', 'dashboard', 'pyproject.toml'):
        raise ValueError('Runtime code has local modifications')
    spec = importlib.util.find_spec('robolab')
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(source):
        raise ValueError('Install RoboLab editable from --runtime-source in this environment')
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Python 3.11 required')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise ValueError('Runtime package versions differ from declared historical versions')
    if os.environ.get('OMNI_KIT_ACCEPT_EULA') != 'Y':
        raise ValueError('Read and accept the NVIDIA license before setting OMNI_KIT_ACCEPT_EULA=Y')
    return {'commit': RUNTIME_COMMIT, 'packages': versions}


def run_identity(runtime_source):
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'code').glob('*.py'))
    files += sorted((root / 'configs').glob('*')) + [root / 'requirements.txt']
    return {'api': api_identity(), 'runtime': runtime_identity(runtime_source),
            'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in files if p.is_file()}}


def verify_resume(work, runtime_source, runner_config):
    recorded = json.loads((Path(work) / 'run-inputs.json').read_text())
    if recorded['identity'] != run_identity(runtime_source):
        raise ValueError('Code/runtime/API identity changed; start a new run instead of resuming')
    digest = hashlib.sha256(Path(runner_config).read_bytes()).hexdigest()
    if recorded['runner_config_sha256'] != digest:
        raise ValueError('Runner configuration changed; start a new run')
