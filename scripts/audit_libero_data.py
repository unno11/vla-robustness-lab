"""CPU-only structural/read audit for pinned original LIBERO and OFT RLDS data."""
import argparse
import ast
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import google_crc32c
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import h5py
import numpy as np
from PIL import Image


def example_class():
    """Minimal standard tf.train.Example wire schema; no TensorFlow execution."""
    fd = descriptor_pb2.FileDescriptorProto(name='audit_example.proto', package='tensorflow', syntax='proto3')
    def field(m, name, number, kind, label=1, type_name=None, oneof=None):
        f = m.field.add(name=name, number=number, type=kind, label=label)
        if type_name:
            f.type_name = type_name
        if oneof is not None:
            f.oneof_index = oneof
    for name, kind in [('BytesList', 12), ('FloatList', 2), ('Int64List', 3)]:
        m = fd.message_type.add(name=name)
        field(m, 'value', 1, kind, 3)
    m = fd.message_type.add(name='Feature')
    m.oneof_decl.add(name='kind')
    for i, name in enumerate(['BytesList', 'FloatList', 'Int64List'], 1):
        field(m, ['bytes_list', 'float_list', 'int64_list'][i-1], i, 11, type_name='.tensorflow.'+name, oneof=0)
    m = fd.message_type.add(name='Features')
    entry = m.nested_type.add(name='FeatureEntry')
    entry.options.map_entry = True
    field(entry, 'key', 1, 9)
    field(entry, 'value', 2, 11, type_name='.tensorflow.Feature')
    field(m, 'feature', 1, 11, 3, '.tensorflow.Features.FeatureEntry')
    m = fd.message_type.add(name='Example')
    field(m, 'features', 1, 11, type_name='.tensorflow.Features')
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('tensorflow.Example'))


def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def records(path):
    with path.open('rb') as f:
        while True:
            header = f.read(12)
            if not header:
                return
            assert len(header) == 12, 'truncated TFRecord header'
            length, crc = struct.unpack('<QI', header)
            assert crc == masked_crc(header[:8]), 'TFRecord length CRC mismatch'
            assert length <= 512 * 1024**2, 'unexpected record >512MiB'
            data = f.read(length)
            footer = f.read(4)
            assert len(data) == length and len(footer) == 4, 'truncated TFRecord payload'
            assert struct.unpack('<I', footer)[0] == masked_crc(data), 'TFRecord payload CRC mismatch'
            yield data


def audit_hdf5(root, source):
    tree = ast.parse((source / 'libero/libero/benchmark/libero_suite_task_map.py').read_text())
    task_map = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in node.targets))
    report = {'tasks': [], 'errors': [], 'episode_count': 0, 'step_count': 0, 'image_frames_read': 0,
              'scope': 'All task files and episode schemas; all numeric actions/states/rewards/dones read; first and last image per camera per episode decoded. Intermediate image pixels not exhaustively read.'}
    for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10']:
        found = {p.name.removesuffix('_demo.hdf5') for p in (root / suite).glob('*.hdf5')}
        expected = set(task_map[suite])
        if found != expected:
            report['errors'].append({'suite': suite, 'missing': sorted(expected-found), 'unexpected': sorted(found-expected)})
        for task in sorted(expected):
            entry = {'suite': suite, 'task': task, 'episodes': 0, 'steps': 0, 'reward_positive_at_end': 0, 'image_shapes': []}
            try:
                with h5py.File(root / suite / (task + '_demo.hdf5'), 'r') as f:
                    data = f['data']
                    demos = sorted(k for k in data if k.startswith('demo_'))
                    assert len(demos) == int(data.attrs['num_demos']), 'num_demos mismatch'
                    for name in demos:
                        g = data[name]
                        actions = g['actions'][:]
                        n = len(actions)
                        assert n > 0 and actions.shape == (n, 7) and np.isfinite(actions).all()
                        for key in ['states', 'robot_states', 'obs/ee_pos', 'obs/ee_ori', 'obs/ee_states', 'obs/gripper_states', 'obs/joint_states', 'rewards', 'dones']:
                            arr = g[key][:]
                            assert arr.shape[0] == n and np.isfinite(arr).all(), (name, key)
                        for key in ['obs/agentview_rgb', 'obs/eye_in_hand_rgb']:
                            images = g[key]
                            assert images.shape == (n, 128, 128, 3) and images.dtype == np.uint8
                            for index in sorted({0, n-1}):
                                img = images[index]
                                assert img.shape == (128, 128, 3)
                                report['image_frames_read'] += 1
                            if list(images.shape[1:]) not in entry['image_shapes']:
                                entry['image_shapes'].append(list(images.shape[1:]))
                        entry['reward_positive_at_end'] += int(g['rewards'][-1] > 0)
                        entry['episodes'] += 1
                        entry['steps'] += n
                    assert entry['steps'] == int(data.attrs['total']), 'total steps mismatch'
            except Exception:
                report['errors'].append({'suite': suite, 'task': task, 'error': traceback.format_exc()})
            report['tasks'].append(entry)
            report['episode_count'] += entry['episodes']
            report['step_count'] += entry['steps']
            print('HDF5', suite, task, entry['episodes'], flush=True)
    report['passed'] = not report['errors'] and len(report['tasks']) == 130
    return report


def audit_rlds(root):
    Example = example_class()
    report = {'suites': [], 'errors': [], 'scope': 'All TFRecord framing and CRC32C, all Example records and numeric/sequence fields; first/last JPEG images of first episode in each suite. Not a TensorFlow/TFDS training-loader integration test.'}
    for suite in ['libero_spatial_no_noops', 'libero_object_no_noops', 'libero_goal_no_noops', 'libero_10_no_noops']:
        entry = {'suite': suite, 'episodes': 0, 'steps': 0, 'shards': 0, 'jpeg_frames_decoded': 0}
        try:
            base = root / suite / '1.0.0'
            info = json.loads((base / 'dataset_info.json').read_text())
            expected = list(map(int, info['splits'][0]['shardLengths']))
            paths = sorted(base.glob('*.tfrecord-*'))
            assert len(paths) == len(expected), 'shard count mismatch'
            for shard, path in enumerate(paths):
                count = 0
                for payload in records(path):
                    fields = Example.FromString(payload).features.feature
                    def values(key, kind):
                        assert key in fields and fields[key].WhichOneof('kind') == kind, key
                        return getattr(fields[key], kind).value
                    actions = values('steps/action', 'float_list')
                    n, remainder = divmod(len(actions), 7)
                    assert n > 0 and remainder == 0 and np.isfinite(actions).all()
                    for key, dim in [('state', 8), ('joint_state', 7)]:
                        arr = values('steps/observation/' + key, 'float_list')
                        assert len(arr) == n * dim and np.isfinite(arr).all()
                    for key in ['reward', 'discount']:
                        arr = values('steps/' + key, 'float_list')
                        assert len(arr) == n and np.isfinite(arr).all()
                    for key in ['is_first', 'is_last', 'is_terminal']:
                        arr = values('steps/' + key, 'int64_list')
                        assert len(arr) == n and set(arr).issubset({0, 1})
                    first = values('steps/is_first', 'int64_list')
                    last = values('steps/is_last', 'int64_list')
                    assert first[0] == 1 and sum(first) == 1 and last[-1] == 1 and sum(last) == 1
                    language = values('steps/language_instruction', 'bytes_list')
                    assert len(language) == n and all(x.decode('utf-8').strip() for x in language)
                    for camera in ['image', 'wrist_image']:
                        imgs = values('steps/observation/' + camera, 'bytes_list')
                        assert len(imgs) == n
                        if entry['episodes'] == 0:
                            for index in sorted({0, n-1}):
                                with Image.open(io.BytesIO(imgs[index])) as image:
                                    image.load()
                                    assert image.format == 'JPEG' and image.size == (256, 256) and image.mode == 'RGB'
                                entry['jpeg_frames_decoded'] += 1
                    entry['episodes'] += 1
                    entry['steps'] += n
                    count += 1
                assert count == expected[shard], ('record count', path.name, count, expected[shard])
                entry['shards'] += 1
                print('RLDS', suite, path.name, count, flush=True)
            assert entry['episodes'] == sum(expected)
        except Exception:
            report['errors'].append({'suite': suite, 'error': traceback.format_exc()})
        report['suites'].append(entry)
    report['passed'] = not report['errors'] and len(report['suites']) == 4
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--rlds', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), 'Preserve prior audit evidence; use a new output file'
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_used': False, 'training_performed': False, 'policy_rollout_completed': False}
    report['hdf5'] = audit_hdf5(args.original, args.source)
    args.output.write_text(json.dumps(report, indent=2))
    report['rlds'] = audit_rlds(args.rlds)
    report['passed'] = report['hdf5']['passed'] and report['rlds']['passed']
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2))
    print('AUDIT_PASSED', report['passed'], flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
