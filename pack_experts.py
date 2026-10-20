#!/usr/bin/env python3
"""Build optional, byte-exact expert sidecars. No tensor conversion or training.

Run on the destination Mac after downloading the native model. Source size and
mtime are bound into the index; replacing/touching source shards invalidates it.
Memory is bounded to one expert (32 MiB maximum). Existing outputs are refused.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time
import zlib

MAX_EXPERT = 32 * 1024**2
PARTS = [(p, q) for p in ('up_proj', 'gate_proj', 'down_proj')
         for q in ('weight', 'scales', 'biases')]


def read_at(fd, length, offset):
    out = bytearray()
    while len(out) < length:
        data = os.pread(fd, length - len(out), offset + len(out))
        if not data:
            raise ValueError('truncated source shard')
        out.extend(data)
    return out


def stat_key(path):
    st = path.stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def headers(model):
    index = model / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    names = sorted(set(mapping.values())) if mapping else [p.name for p in model.glob('*.safetensors')]
    tensors = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError('unsafe source shard name')
        path = model / name
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if not 0 < size <= 32 * 1024**2:
                raise ValueError('invalid safetensors header length')
            data = json.loads(f.read(size))
        for key, info in data.items():
            if key == '__metadata__' or (mapping and mapping.get(key) != name):
                continue
            if not re.fullmatch(r'(language_model\.)?model\.layers\.\d+\.mlp\.(experts|switch_mlp)\.(up_proj|gate_proj|down_proj)\.(weight|scales|biases)', key):
                continue
            start, end = info['data_offsets']
            if not 0 <= start < end <= path.stat().st_size - size - 8:
                raise ValueError('invalid tensor bounds')
            tensors[key] = dict(info, shard=name, begin=8+size+start, end=8+size+end)
    return tensors


def pack(model, output, layers=None, rate_mib=20):
    model, output = Path(model).resolve(), Path(output).resolve()
    tensors = headers(model)
    prefixes = sorted({k.rsplit('.', 2)[0] for k in tensors})
    if layers is not None:
        prefixes = [p for p in prefixes if int(re.search(r'\.layers\.(\d+)\.', p)[1]) in layers]
    if not prefixes:
        raise ValueError('no requested expert tensors found')
    output.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        li = int(re.search(r'\.layers\.(\d+)\.', prefix)[1])
        meta = [tensors[prefix+'.'+p+'.'+q] for p,q in PARTS]
        E = meta[0]['shape'][0]
        lengths = []
        sources = []
        for i, t in enumerate(meta):
            width = 4 if i % 3 == 0 else 2
            if (t['dtype'] != ('U32' if width == 4 else 'BF16') or
                len(t['shape']) != 3 or t['shape'][0] != E or E <= 0):
                raise ValueError('unsupported expert tensor layout')
            length = (t['end'] - t['begin']) // E
            if length * E != t['end'] - t['begin'] or length != t['shape'][1] * t['shape'][2] * width:
                raise ValueError('expert shape/length mismatch')
            lengths.append(length)
            sources.append(dict(shard=t['shard'], begin=t['begin'], end=t['end'],
                                **stat_key(model / t['shard'])))
        total = sum(lengths)
        if not 0 < total <= MAX_EXPERT:
            raise ValueError('expert exceeds bounded staging buffer')
        if shutil.disk_usage(output).free < E * total + 5 * 1024**3:
            raise ValueError('insufficient disk headroom (5 GiB reserve)')
        data_path, index_path = output/f'layer-{li}.bin', output/f'layer-{li}.json'
        if data_path.exists() or index_path.exists():
            raise FileExistsError(f'refusing to replace {data_path} or its index')
        started = time.monotonic()
        checksums = []
        with data_path.open('xb') as out:
            for e in range(E):
                crc = 0
                for t, length in zip(meta, lengths):
                    with (model / t['shard']).open('rb') as src:
                        block = read_at(src.fileno(), length, t['begin'] + e * length)
                    crc = zlib.crc32(block, crc)
                    out.write(block)
                checksums.append(crc)
                if rate_mib:
                    delay = (e+1) * total / (rate_mib * 1024**2) - (time.monotonic()-started)
                    if delay > 0:
                        time.sleep(delay)
            out.flush()
            os.fsync(out.fileno())
        for source in sources:
            if stat_key(model/source['shard']) != {k: source[k] for k in ('size', 'mtime_ns')}:
                raise ValueError('source changed while packing; incomplete sidecar must be removed')
        record = {'format': 'edge0-experts-v1', 'layer': li, 'experts': E,
                  'lengths': lengths, 'sources': sources, 'crc32': checksums,
                  'data_bytes': E * total}
        with index_path.open('x') as out:
            json.dump(record, out, indent=2)
            out.write('\n')
        print(f'Packed layer {li}: {E} experts, {total} bytes/expert', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--layers', help='Optional comma-separated layers for a limited I/O experiment')
    ap.add_argument('--rate-mib', type=float, default=20)
    args = ap.parse_args()
    if not 0 < args.rate_mib <= 100:
        ap.error('rate-mib must be >0 and <=100')
    pack(args.model_dir, args.output,
         set(map(int,args.layers.split(','))) if args.layers else None, args.rate_mib)


if __name__ == '__main__':
    main()
