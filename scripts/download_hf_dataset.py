"""Download a pinned public HF dataset manifest with streaming hash verification (Linux)."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo, revision = manifest['id'], manifest['sha']
    assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    assert 1 <= args.workers <= 4
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    job_lock = args.state.with_suffix('.lock').open('w')
    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = manifest['siblings']
    total = sum(f['size'] for f in files)
    assert shutil.disk_usage(target).free > total * 1.1, 'Insufficient space for full dataset plus margin'
    state = dict(repo_id=repo, revision=revision, expected_files=len(files),
                 expected_bytes=total, verified_files=0, verified_bytes=0,
                 errors=[], active={}, phase='downloading', complete=False)
    lock = threading.Lock()
    stop = threading.Event()

    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp = args.state.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(args.state)

    def valid(path, entry):
        if not path.is_file() or path.stat().st_size != entry['size']:
            return False
        digest = hashlib.sha256() if entry.get('lfs') else hashlib.sha1()
        if not entry.get('lfs'):
            digest.update(b'blob ' + str(entry['size']).encode() + b'\0')
        with path.open('rb') as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
        expected = entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']
        return digest.hexdigest() == expected

    def heartbeat():
        while not stop.wait(10):
            with lock:
                for name, entry in state['active'].items():
                    part = target / (name + '.partial')
                    entry['partial_bytes'] = part.stat().st_size if part.exists() else 0
                save()

    def work(entry):
        name = entry['rfilename']
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert destination.resolve().is_relative_to(target)
        part = destination.with_name(destination.name + '.partial')
        with lock:
            state['active'][name] = dict(expected_bytes=entry['size'], partial_bytes=0)
            save()
        try:
            if not valid(destination, entry):
                if not valid(part, entry):
                    url = args.endpoint.rstrip('/') + '/datasets/' + repo + '/resolve/' + revision + '/' + name
                    result = subprocess.run([
                        'curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                        '--connect-timeout', '15', '--max-time', '21600', '--retry', '4',
                        '--retry-all-errors', '--retry-delay', '5', '--continue-at', '-',
                        url, '-o', str(part)], capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-800:])
                if not valid(part, entry):
                    raise RuntimeError('Upstream size or hash mismatch; partial retained, not accepted')
                part.replace(destination)
            with lock:
                state['verified_files'] += 1
                state['verified_bytes'] += entry['size']
        except Exception as error:
            with lock:
                state['errors'].append(dict(file=name, error=str(error)))
        finally:
            with lock:
                state['active'].pop(name, None)
                save()

    save()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(work, files))
    stop.set()
    thread.join()
    state['complete'] = state['verified_files'] == len(files) and not state['errors']
    state['phase'] = 'verified_download' if state['complete'] else 'incomplete_with_errors'
    save()
    if state['complete']:
        (target / 'DOWNLOAD_VERIFIED.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    return 0 if state['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
