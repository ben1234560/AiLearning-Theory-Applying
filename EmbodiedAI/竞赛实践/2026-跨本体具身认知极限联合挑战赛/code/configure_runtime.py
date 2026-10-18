"""Pin Isaac Sim 5.1 to one renderer GPU per independent runner process."""
import hashlib
import importlib.metadata
import importlib.util
import sysconfig
import shutil
from pathlib import Path


def configure():
    dist=importlib.metadata.distribution('isaacsim')
    if dist.version!='5.1.0.0':raise RuntimeError(f'Unsupported Isaac Sim version: {dist.version}')
    root=Path(next(iter(importlib.util.find_spec('isaacsim').submodule_search_locations)))
    files=list(root.glob('exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py'))
    if len(files)!=1:raise RuntimeError(f'Cannot uniquely locate SimulationApp: {files}')
    path=Path(files[0]);source=path.read_text()
    old='"multi_gpu": True,';new='"multi_gpu": False,'
    if source.count(old)==1:source=source.replace(old,new,1);path.write_text(source)
    elif source.count(new)!=1:raise RuntimeError('Unexpected SimulationApp default configuration')
    lab_root=Path(next(iter(importlib.util.find_spec('isaaclab').submodule_search_locations)))
    launchers=list(lab_root.glob('source/isaaclab/isaaclab/app/app_launcher.py'))
    if not launchers:launchers=list(lab_root.glob('app/app_launcher.py'))
    if len(launchers)!=1:raise RuntimeError('Cannot locate Isaac Lab AppLauncher')
    launcher=launchers[0];text=launcher.read_text()
    old_line='launcher_args["active_gpu"] = self.device_id'
    new_line='launcher_args["active_gpu"] = int(os.environ.get("BASELINE_RENDER_GPU", self.device_id))'
    if text.count(old_line)==1:launcher.write_text(text.replace(old_line,new_line,1))
    elif text.count(new_line)!=1:raise RuntimeError('Unexpected AppLauncher renderer configuration')
    site=Path(sysconfig.get_path('purelib'))
    shutil.copyfile(Path(__file__).with_name('gpu_isolation.py'),site/'baseline_gpu_isolation.py')
    (site/'baseline_gpu_isolation.pth').write_text('import baseline_gpu_isolation\n')
    print('Isaac Sim per-process multi_gpu=False',hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)

if __name__=='__main__':configure()
