"""Map each official record subprocess to a single logical CUDA GPU.

Loaded at interpreter startup by a .pth file installed by configure_runtime.py.
This runs before torch/Isaac imports and leaves all task, observation and policy
logic untouched. Vulkan uses the physical renderer index; CUDA uses local 0.
"""
import os
import sys

if os.environ.get('ROBOLAB_RUNNER_DISTRIBUTION_ROOT') and 'record' in sys.argv and '--contract' in sys.argv and '--device' in sys.argv:
    index=sys.argv.index('--device')+1
    device=sys.argv[index]
    if device.startswith('cuda:'):
        physical=int(device.split(':')[1])
        os.environ['CUDA_VISIBLE_DEVICES']=str(physical)
        os.environ['BASELINE_RENDER_GPU']=str(physical)
        sys.argv[index]='cuda:0'
