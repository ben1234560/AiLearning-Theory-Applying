import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--runner',required=True);p.add_argument('--port',type=int,default=5562);p.add_argument('--cache',required=True);a=p.parse_args()
os.environ['PYTHONPATH']=str(Path(a.runner).resolve()/'competition_runner/runner')
os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('policy.py')),'--endpoint',f'tcp://127.0.0.1:{a.port}','--cache',a.cache])
