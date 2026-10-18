"""Restore baseline in an empty work directory from the declared official archive."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ARCHIVE_SHA256='9166b1ea123a0eae25d71bb174c8f70c4b8e46ba403b70e9d324f6a5422bcbc4'
RUNNER_NAME='robolab-benchmark30-dual-runner-rgb3-20260818-v2'

def main():
    from configure_runtime import configure
    from run_guard import run_identity
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True);p.add_argument('--work',required=True)
    p.add_argument('--runtime-source',required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--devices',nargs='+',default=['cuda:0'],help='GPU devices assigned cyclically to workers')
    p.add_argument('--max-steps',type=int,default=7200);p.add_argument('--run-id',default='baseline-v4')
    p.add_argument('--port-base',type=int,default=6062);p.add_argument('--task');p.add_argument('--embodiment',default='r1_pro_official')
    p.add_argument('--cache');p.add_argument('--replay-only',action='store_true')
    a=p.parse_args(); work=Path(a.work).resolve();source=Path(__file__).resolve().parent
    if a.workers<1 or a.max_steps<1 or not 1<=a.port_base<=65535-a.workers:
        p.error('positive workers/steps and a valid port range required')
    if Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):
        p.error('run-id must be a single directory name')
    if any(not d.startswith('cuda:') or not d[5:].isdigit() for d in a.devices):
        p.error('devices must be cuda:N')
    if work.exists() and any(work.iterdir()):raise ValueError('work directory must be empty')
    identity=run_identity(a.runtime_source)
    h=hashlib.sha256()
    with open(a.archive,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=ARCHIVE_SHA256:raise ValueError('official archive SHA256 mismatch')
    configure()
    work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        for info in z.infolist():
            if info.filename.startswith('__MACOSX'):continue
            target=(work/info.filename).resolve()
            if not target.is_relative_to(work):raise ValueError('unsafe archive member')
            f=Path(z.extract(info,work)); mode=info.external_attr>>16
            if mode:f.chmod(mode & 0o777)
    runner=work/RUNNER_NAME
    config=json.loads((runner/'config/runner.example.json').read_text())
    config['runtime'].update(python=sys.executable,isaacSimRoot=str(Path(next(iter(importlib.util.find_spec('isaacsim').submodule_search_locations))).resolve()),robolabSourceRoot=str(Path(a.runtime_source).resolve()))
    workers=1 if a.task else a.workers; ports=list(range(a.port_base,a.port_base+workers))
    config['policy'].update(port=ports[0],actionHorizon=1,executeSteps=1)
    config['run'].update(minimumFreeGiB=20,spaceSaving=True,maxSteps=a.max_steps,workers=workers,devices=[a.devices[i%len(a.devices)] for i in range(workers)],policyPorts=ports)
    (runner/'config/runner.json').write_text(json.dumps(config,indent=2))
    (work/'run-inputs.json').write_text(json.dumps({'identity':identity,'runner_config_sha256':hashlib.sha256((runner/'config/runner.json').read_bytes()).hexdigest()},indent=2))
    env=os.environ.copy();env['OMNI_KIT_ACCEPT_EULA']='Y';env['OMP_NUM_THREADS']='4'
    if a.replay_only:env['REPLAY_ONLY']='1'
    cache=work/'decision-cache'
    if a.cache:shutil.copytree(a.cache,cache)
    else:cache.mkdir()
    def run(cmd,name):
        with (work/name).open('w') as log:subprocess.run(cmd,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    run(['./run.sh','check-env','--config','config/runner.json','--shallow'],'environment.log')
    processes=[];logs=[]
    try:
        for port in ports:
            log=(work/f'policy-{port}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,str(source/'launch_policy.py'),'--runner',str(runner),'--port',str(port),'--cache',str(cache)],env=env,stdout=log,stderr=subprocess.STDOUT))
        time.sleep(3)
        if any(proc.poll() is not None for proc in processes):raise RuntimeError('policy failed to start')
        selection=['--task',a.task,'--embodiment',a.embodiment] if a.task else ['--all']
        run(['./run.sh','execute','--config','config/runner.json',*selection,'--formal','--headless','--run-id',a.run_id],'execute.log')
        args=['./submit.sh','--config','config/runner.json','--run','output/'+a.run_id]
        if a.task:args+=['--demo','--task',a.task,'--embodiment',a.embodiment]
        run(args,'package.log')
        (work/'REPRODUCE_SUCCESS.json').write_text(json.dumps({'run_id':a.run_id,'task':a.task,'formal_submission':not bool(a.task),'archive_sha256':h.hexdigest(),'max_steps':a.max_steps},indent=2))
    finally:
        for proc in processes:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()
        for log in logs:log.close()

if __name__=='__main__':main()
