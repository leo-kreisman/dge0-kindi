#!/usr/bin/env python3
"""Download and reconstruct the pinned Edge0 model; Python 3 and curl only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'https://github.com/leo-kreisman/dge0-kindi/releases/download/model-edge0-35b-a3b-int4-v1'
BLOCK = 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(BLOCK), b''):
            h.update(b)
    return h.hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name or '\\' in name:
        raise ValueError('Unsafe manifest filename')
    return name


def fetch(name, folder):
    safe_name(name)
    path = folder / name
    if path.exists():
        return path
    partial = folder / (name + '.download')
    subprocess.run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
                    '--retry', '5', '--connect-timeout', '30', '--continue-at', '-',
                    '--output', str(partial), BASE + '/' + name], check=True)
    partial.replace(path)
    return path


def reconstruct(folder, offline=False):
    manifest = folder / 'model-manifest.json'
    if not offline:
        fetch(manifest.name, folder)
    m = json.loads(manifest.read_text())
    model = folder / 'Edge0-35B-A3B-preview'
    model.mkdir(exist_ok=True)
    for item in m['files']:
        name = safe_name(item['name'])
        target = model / name
        if target.exists():
            if target.stat().st_size != item['size'] or digest(target) != item['sha256']:
                raise RuntimeError('Existing output differs; not overwriting: ' + name)
            print('Already verified:', name, flush=True)
            continue
        temporary = model / (name + '.assembling')
        # Exclusive creation avoids overwriting another running assembler.
        with temporary.open('xb') as out:
            h = hashlib.sha256()
            size = 0
            try:
                for part in item['parts']:
                    part_name = safe_name(part['name'])
                    source = folder / part_name if offline else fetch(part_name, folder)
                    if source.stat().st_size != part['size']:
                        raise RuntimeError('Part size mismatch: ' + part_name)
                    ph = hashlib.sha256()
                    with source.open('rb') as f:
                        for block in iter(lambda: f.read(BLOCK), b''):
                            ph.update(block)
                            h.update(block)
                            size += len(block)
                            out.write(block)
                    if ph.hexdigest() != part['sha256']:
                        raise RuntimeError('Part checksum mismatch: ' + part_name)
                if size != item['size'] or h.hexdigest() != item['sha256']:
                    raise RuntimeError('Reconstructed checksum mismatch: ' + name)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.rename(target)
        print('Reconstructed and verified:', name, flush=True)
    print('Model folder:', model.resolve())
    print('Keep the complete folder, including the adapters and tokenizer. No inference was started.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offline', action='store_true', help='Assemble already downloaded release assets')
    ap.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    reconstruct(args.directory, args.offline)


if __name__ == '__main__':
    main()
