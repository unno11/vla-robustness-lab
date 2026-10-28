"""Bounded simulator diagnostic; no policy or benchmark score is produced."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--libero-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--gpu', type=int)
parser.add_argument('--mode', choices=['camera', 'physics'], default='camera')
args = parser.parse_args()
if args.mode == 'camera' and args.gpu is None:
    parser.error('--gpu is required for camera mode')
root = args.libero_root.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ['LIBERO_CONFIG_PATH'] = str(out / 'libero-config')
os.environ['MUJOCO_GL'] = 'egl' if args.mode == 'camera' else 'disable'
if args.mode == 'camera':
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(root))
report = {'kind': 'simulator_smoke_only', 'passed': False, 'closed': False,
          'seed': 0, 'initialization': 'seeded_reset_not_benchmark_init_state',
          'steps_budget': 5, 'steps_completed': 0, 'policy_used': False,
          'training_performed': False, 'render_backend': os.environ['MUJOCO_GL'],
          'gpu': args.gpu if args.mode == 'camera' else None, 'mode': args.mode,
          'camera_validated': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_sha': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
env = None
try:
    import numpy as np
    import yaml
    from PIL import Image
    config = out / 'libero-config'
    config.mkdir()
    base = root / 'libero/libero'
    dataset = out / 'unused-datasets'
    dataset.mkdir()
    (config / 'config.yaml').write_text(yaml.safe_dump({
        'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
        'init_states': str(base / 'init_files'), 'assets': str(base / 'assets'),
        'datasets': str(dataset)}))
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv, ControlEnv
    task = 'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate'
    bddl = base / 'bddl_files/libero_spatial' / (task + '.bddl')
    report.update({'suite': 'libero_spatial', 'task': task,
                   'bddl_sha256': hashlib.sha256(bddl.read_bytes()).hexdigest()})
    np.random.seed(0)
    if args.mode == 'camera':
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                camera_widths=256, render_gpu_device_id=args.gpu)
    else:
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                         has_renderer=False, has_offscreen_renderer=False)
    env.seed(0)
    obs = env.reset()
    report['native_success_at_reset'] = bool(env.check_success())
    report['observations'] = {}
    for key in (['agentview_image', 'robot0_eye_in_hand_image'] if args.mode == 'camera' else []):
        arr = obs[key]
        assert arr.shape == (256, 256, 3) and arr.dtype == np.uint8
        assert float(arr.std()) > 1, 'blank image'
        np.save(out / (key + '.npy'), arr)
        # Raw array saved above. Flip only this human-facing preview vertically.
        Image.fromarray(arr[::-1]).save(out / (key + '.png'))
        report['observations'][key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype),
            'std': float(arr.std()), 'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
            'preview_transform': 'vertical_flip_only'}
    report['camera_validated'] = args.mode == 'camera'
    # Zero arm motion and open-gripper command: diagnostic, never a policy result.
    for _ in range(5):
        obs, reward, done, info = env.step([0., 0., 0., 0., 0., 0., -1.])
        report['steps_completed'] += 1
        if done:
            break
    report['native_success_after_steps'] = bool(env.check_success())
    report['diagnostic_completed'] = True
except Exception:
    report['error'] = traceback.format_exc()
finally:
    if env is not None:
        try:
            env.close()
            report['closed'] = True
        except Exception:
            report['close_error'] = traceback.format_exc()
    report['passed'] = bool(report.get('diagnostic_completed') and report['closed'])
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if not report['passed']:
    raise SystemExit(1)
