"""Exercise renderer configuration in temporary files, never installed packages."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import configure_runtime

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);sim=root/'sim';lab=root/'lab';site=root/'site';site.mkdir()
    app=sim/'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py'
    app.parent.mkdir(parents=True);app.write_text('CONFIG = {"multi_gpu": True,}\n')
    launcher=lab/'source/isaaclab/isaaclab/app/app_launcher.py'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('import os\nlauncher_args["active_gpu"] = self.device_id\n')
    def spec(name):
        return SimpleNamespace(submodule_search_locations=[str({'isaacsim':sim,'isaaclab':lab}[name])])
    with patch.object(configure_runtime.importlib.metadata,'distribution',return_value=SimpleNamespace(version='5.1.0.0')),patch.object(configure_runtime.importlib.util,'find_spec',side_effect=spec),patch.object(configure_runtime.sysconfig,'get_path',return_value=str(site)):
        configure_runtime.configure();configure_runtime.configure()
    hook=(site/'baseline_gpu_isolation.pth').read_text()
    assert hook=='import baseline_gpu_isolation\n'
    compile(hook,'temporary startup hook','exec')
    assert '"multi_gpu": False,' in app.read_text()
    assert 'BASELINE_RENDER_GPU' in launcher.read_text()
    assert (site/'baseline_gpu_isolation.py').is_file()
    with patch.dict(os.environ,{'ROBOLAB_RUNNER_DISTRIBUTION_ROOT':'/test/official'}),patch.object(sys,'argv',['fixture.py','record','--contract','fixture','--device','cuda:2']):
        runpy.run_path(str(Path(configure_runtime.__file__).with_name('gpu_isolation.py')))
        assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
        assert os.environ['BASELINE_RENDER_GPU']=='2'
        assert sys.argv[-1]=='cuda:0'
print('PASS: temporary startup hook, idempotence and CUDA/renderer mapping')
