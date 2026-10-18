"""Retry only invalid cells in an existing v4 official run, preserving attempt logs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from configure_runtime import configure
from reproduce import RUNNER_NAME
from run_guard import verify_resume


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True)
    a=p.parse_args()
    work=Path(a.work).resolve();runner=work/RUNNER_NAME;code=Path(__file__).resolve().parent
    config=json.loads((runner/'config/runner.json').read_text())
    verify_resume(work,config['runtime']['robolabSourceRoot'],runner/'config/runner.json')
    manifest=json.loads((runner/'output/baseline-v4/run_manifest.json').read_text())
    assert manifest['policy']=={'policyId':'rgb3-qwen-closed-loop','policyVersion':'v4'}
    assert manifest['execution']['maxSteps']==7200
    assert manifest['execution']['actionHorizon']==manifest['execution']['executePrefixSteps']==1
    summary=json.loads((runner/'output/baseline-v4/run_summary.json').read_text())
    assert summary['status']!='RUNNING','do not overlap official matrix schedulers'
    assert summary['status']!='COMPLETED','already complete; package existing outputs instead'
    configure()
    old_logs=list(work.glob('*.log'))
    if old_logs:
        archive=work/'attempt-logs'/time.strftime('%Y%m%d-%H%M%S')
        archive.mkdir(parents=True,exist_ok=False)
        for log in old_logs:shutil.copy2(log,archive/log.name)
    env=os.environ.copy();env.update(OMNI_KIT_ACCEPT_EULA='Y',OMP_NUM_THREADS='4')
    def run(cmd,log):
        with (work/log).open('w') as f:
            subprocess.run(cmd,cwd=runner,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    run(['./run.sh','check-env','--config','config/runner.json','--shallow'],'environment-retry.log')
    processes=[];logs=[]
    try:
        for port in config['run']['policyPorts']:
            log=(work/f'policy-retry-{port}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,str(code/'launch_policy.py'),
                '--runner',str(runner),'--port',str(port),'--cache',str(work/'decision-cache')],
                env=env,stdout=log,stderr=subprocess.STDOUT))
        time.sleep(3)
        assert all(proc.poll() is None for proc in processes),'policy startup failed'
        run(['./run.sh','execute','--config','config/runner.json','--all','--formal',
            '--headless','--run-id','baseline-v4','--resume','--retry-failed'],'execute-retry.log')
        run(['./submit.sh','--config','config/runner.json','--run','output/baseline-v4'],'package-retry.log')
        (work/'RESUME_SUCCESS.json').write_text(json.dumps({'run_id':'baseline-v4','formal_submission':True}))
    finally:
        for proc in processes:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()
        for log in logs:log.close()


if __name__=='__main__':main()
