"""CPU-only integrity and public-SDK checks. Never starts a simulator or API."""
import argparse
import ast
import hashlib
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,help='extracted official competition package')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Missing or unsafe manifest file: '+name)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Checksum mismatch: '+name)
    for source in root.rglob('*.py'):
        if '.git' not in source.parts:ast.parse(source.read_text(),filename=str(source.relative_to(root)))
    print('PASS: file checksums and Python syntax')
    sys.path.insert(0,str(root/'code'))
    runpy.run_path(str(root/'tests/test_startup.py'),run_name='__main__')
    runpy.run_path(str(root/'tests/test_guard.py'),run_name='__main__')
    if a.runner:
        sdk=a.runner.resolve()/'competition_runner/runner'
        if not (sdk/'robolab_runner/policy/handler.py').is_file():
            p.error('--runner must point to the extracted official package')
        sys.path.insert(0,str(sdk))
        runpy.run_path(str(root/'tests/test_policy.py'),run_name='__main__')
    else:print('SDK policy tests skipped; supply --runner to run them')
    print('No GPU, simulator or external API was used.')


if __name__=='__main__':main()
