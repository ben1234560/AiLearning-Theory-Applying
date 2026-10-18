"""Optional CPU-only VLM integration probe. Network requires an explicit --call."""
import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--call', action='store_true', help='Allow up to 3 billable API attempts')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
    from run_guard import api_identity
    identity = api_identity()
    if not args.call:
        print('Environment format OK. No network request; add --call for the actual API probe.')
        return
    sys.path.insert(0, str(args.runner.resolve() / 'competition_runner/runner'))
    import numpy as np
    from policy import VisionPolicy
    state = np.zeros(40, np.float32)
    state[[3, 10, 18, 14, 22]] = 1
    obs = {'proprio_obs/canonical_observation40': state}
    for name in ('head_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'):
        obs['image_obs/' + name] = np.zeros((16, 16, 3), np.uint8)
    with tempfile.TemporaryDirectory() as cache:
        policy = VisionPolicy(cache)
        policy.reset(episode_id='api-probe', payload={})
        try:
            action = policy.infer_v3(obs, step=120, episode_id='api-probe',
                                    effective_instruction='Synthetic blank images: stay still and keep grippers open.',
                                    requested_action_horizon=1, execute_prefix_steps=1)
            if action.shape != (1, 26) or not np.isfinite(action).all():
                raise ValueError('Invalid action')
        except Exception:
            # Provider errors may contain private endpoints; never echo the raw response/traceback.
            raise SystemExit('FAIL: check account/model access and support for the documented request parameters.') from None
    print(json.dumps({'api_probe': 'pass', 'model': identity['model'], 'gpu_used': False}))


if __name__ == '__main__':
    main()
