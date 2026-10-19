"""Recover root space by archiving only eight explicitly retired containers to NVMe.

Keep complete writable-layer tar, base image, original inspect JSON and logs before removing the
stopped container. Model/evidence bind mounts and images are never removed.
"""
import hashlib
import json
import shutil
import subprocess
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'root-recovery-r82' / 'snapshot-archives-r83'
NAMES = ['qwen38-dev-r1', 'qwen38-long-native-d-r8', 'qwen38-long-native-m-r8',
         'qwen38-long-yarn-d-r8', 'qwen38-long-yarn-m-r8', 'qwen38-dev-base-fp32-head',
         'qwen38-fp64-projection-d-r13-v2', 'qwen38-fp64-projection-m-r13-v2']
ROOT.mkdir(exist_ok=False)
assert shutil.disk_usage(ROOT).free > 500 * 1024**3
records = []
for name in NAMES:
    info = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert info['Name'] == '/' + name
    assert not info['State']['Running'] and info['State']['Status'] == 'exited'
    assert info['State']['Pid'] == 0
    output = ROOT / name
    output.mkdir(exist_ok=False)
    (output / 'inspect.json').write_text(json.dumps(info, indent=2))
    with (output / 'container.log').open('xb') as f:
        subprocess.run(['docker', 'logs', name], stdout=f, stderr=subprocess.STDOUT, check=True)
    # Docker's merged-rootfs mount failed after ENOSPC. Read the stopped
    # snapshot's existing upperdir, archive every changed file and whiteout,
    # retain its base image, then let Docker remove the stopped container.
    # Never modify containerd metadata or any snapshot path directly.
    mounts = subprocess.check_output(['ctr', '-n', 'moby', 'snapshots', 'mounts',
                                     str(output / 'unused-mount-target'), info['Id']], text=True)
    match = re.search(r'upperdir=([^,]+)', mounts)
    assert match, mounts
    upper = Path(match.group(1))
    assert upper.is_dir() and upper.is_relative_to(
        '/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots')
    assert upper.name == 'fs' and upper.parent.name.isdigit()
    (output / 'snapshot-mount-command.txt').write_text(mounts)
    archive = output / 'writable-layer.tar'
    assert not archive.exists()
    print('archiving stopped writable snapshot', name, upper, flush=True)
    subprocess.run(['tar', '--xattrs', '--acls', '--numeric-owner', '-cpf', str(archive),
                    '-C', str(upper), '.'], check=True)
    hasher = hashlib.sha256()
    with archive.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    count = archive.stat().st_size
    assert count > 0
    subprocess.run(['tar', '-tf', str(archive)], stdout=subprocess.DEVNULL, check=True)
    current = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert current['Id'] == info['Id'] and current['Image'] == info['Image']
    assert current['State'] == info['State'], 'container changed during archive'
    record = {'name': name, 'original_id': info['Id'], 'image': info['Image'],
              'writable_layer_archive': str(archive), 'base_image_retained': True, 'bytes': count, 'sha256': hasher.hexdigest(),
              'inspect_sha256': hashlib.sha256((output / 'inspect.json').read_bytes()).hexdigest(),
              'state_preserved': info['State'], 'model_and_evidence_mounts_preserved': True,
              'image_removed': False, 'unix_s': time.time()}
    (output / 'verified-before-removal.json').write_text(json.dumps(record, indent=2))
    # No force or volume removal; Docker refuses removal if it became running.
    subprocess.run(['docker', 'rm', name], check=True)
    records.append(record)
    (ROOT / 'completed.json').write_text(json.dumps(records, indent=2))
    print('archived and removed stopped metadata', name,
          'root_free_bytes', shutil.disk_usage('/').free, flush=True)
print('finished; images, model files, raw results and active workers preserved', flush=True)
