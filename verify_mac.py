#!/usr/bin/env python3
"""Bounded, offline Mac inference check; no downloads, training or service edits.

Runs one child at a time. Parent watchdog covers model loading and inference.
It samples RSS; it is NOT a proof of a hard instantaneous memory ceiling.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3


def timing_metrics(start, token_times, finished):
    """Decode excludes prefill and the first token; end-to-end includes both."""
    count = len(token_times)
    decode = token_times[-1] - token_times[0] if count > 1 else 0
    wall = finished - start
    return {
        'output_tokens': count,
        'wall_seconds': wall,
        'first_token_seconds': token_times[0] - start if count else None,
        'decode_tokens_per_second': (count - 1) / decode if decode > 0 else None,
        'end_to_end_tokens_per_second': count / wall if wall > 0 else None,
        'token_elapsed_seconds': [t - start for t in token_times],
        'short_sample': count < 32,
    }


def evaluate(args):
    lib = C.CDLL(str(Path(args.dylib).resolve()))
    ptr = C.c_void_p
    sizep = C.POINTER(C.c_size_t)
    lib.e0_engine_create.argtypes = [C.c_char_p] * 3
    lib.e0_engine_create.restype = ptr
    lib.e0_engine_destroy.argtypes = [ptr]
    lib.e0_turn_reset.argtypes = [ptr]
    lib.e0_generate_begin.argtypes = [ptr, C.POINTER(C.c_int32), C.c_uint32, C.c_char_p]
    lib.e0_generate_begin.restype = ptr
    for name in ['e0_stats', 'e0_last_error']:
        getattr(lib, name).argtypes = [ptr, ptr, C.c_size_t, sizep]
    class Token(C.Structure):
        _fields_ = [('size', C.c_uint32), ('token', C.c_int32), ('flags', C.c_uint32)]
    lib.e0_next.argtypes = [ptr, ptr, C.POINTER(Token)]
    lib.e0_tok_create.argtypes = [C.c_char_p]
    lib.e0_tok_create.restype = ptr
    lib.e0_tok_destroy.argtypes = [ptr]
    lib.e0_tok_tokenize.argtypes = [ptr, C.c_char_p, ptr, C.c_size_t, sizep]

    def read_json(fn, *prefix):
        n = C.c_size_t()
        code = fn(*prefix, None, 0, C.byref(n))
        if code < 0:
            raise RuntimeError(f'JSON size query failed: {code}')
        if n.value > 16 * 1024 * 1024:
            raise RuntimeError('oversize ABI response')
        buf = C.create_string_buffer(n.value + 1)
        if fn(*prefix, buf, n.value, C.byref(n)) < 0:
            raise RuntimeError('JSON read failed')
        return json.loads(buf.raw[:n.value])

    tk = lib.e0_tok_create(os.fsencode(args.model_dir))
    if not tk:
        raise RuntimeError('tokenizer could not load')
    # 2200 shared tokens exercise the existing 2048-token prefill boundary.
    # Separate tail tokens exercise divergence; no prompt text leaves this host.
    try:
        text = ('# repository fixture: def add(a, b): return a + b\n' *
                (args.prompt_tokens // 8 + 400))
        ids = read_json(lib.e0_tok_tokenize, tk,
                        json.dumps({'text': text}).encode())['tokens'][0]
        shared = ids[:args.prompt_tokens]
        tails = [read_json(lib.e0_tok_tokenize, tk,
                          json.dumps({'text': t}).encode())['tokens'][0]
                 for t in ['\n# Write a test for add:\n', '\n# Explain add briefly:\n']]
    finally:
        lib.e0_tok_destroy(tk)
    if len(shared) < 2100:
        raise RuntimeError('fixture did not produce enough tokens')
    cfg = {'forward': args.tier.removeprefix('edge0-'), 'lora': True,
           'model_identity': str(Path(args.model_dir).resolve()),
           'expert_io': {'batch_materialize': args.child != 'baseline',
                         'overlap_prefill': args.overlap_prefill and args.child != 'baseline',
                         'packed_dir': args.packed_experts_dir if args.child in ('packed','adaptive','profiled') else '',
                         'adaptive_cache': args.child in ('adaptive','profiled'),
                         'profile_path': args.profile_path if args.child in ('adaptive','profiled') else ''},
           'prefix_cache': {'enabled': args.child in ('cached', 'packed', 'adaptive', 'profiled'),
                            'budget_bytes': 512 * 1024 ** 2,
                            'min_prefix_tokens': 128, 'checkpoint_every': 1024}}
    h = lib.e0_engine_create(args.tier.encode(), os.fsencode(args.model_dir),
                             json.dumps(cfg).encode())
    if not h:
        raise RuntimeError(read_json(lib.e0_last_error, None))
    rows = []
    cases = [('initial', shared + tails[0]), ('repeat', shared + tails[0]),
             ('edited_tail', shared + tails[1]),
             ('changed_prefix', [shared[1]] + shared[1:] + tails[0])]
    try:
        for label, prompt in cases:
            if lib.e0_turn_reset(h) < 0:
                raise RuntimeError('turn reset failed')
            start = time.monotonic()
            before_stats = read_json(lib.e0_stats, h)
            buf = (C.c_int32 * len(prompt))(*prompt)
            params = {'temperature': 0, 'max_new_tokens': args.max_new_tokens}
            stream = lib.e0_generate_begin(h, buf, len(prompt), json.dumps(params).encode())
            if not stream:
                raise RuntimeError(read_json(lib.e0_last_error, h))
            out = []
            token_times = []
            while True:
                if time.monotonic() - start > args.timeout:
                    raise RuntimeError('request timed out')
                t = Token(C.sizeof(Token), 0, 0)
                code = lib.e0_next(h, stream, C.byref(t))
                if code < 0:
                    raise RuntimeError(read_json(lib.e0_last_error, h))
                if t.flags & 1:
                    break
                if code == 1:
                    token_times.append(time.monotonic())
                    out.append(t.token)
                    if len(out) > args.max_new_tokens:
                        raise RuntimeError('engine exceeded generation limit')
                elif code == 2:
                    time.sleep(0.001)
                else:
                    raise RuntimeError(f'unexpected stream status: {code}')
            finished = time.monotonic()
            stats = read_json(lib.e0_stats, h)
            if args.child != 'reference' and 'joint_budget' not in stats:
                raise RuntimeError('unpatched dylib: joint budget telemetry absent')
            rows.append({'case': label, 'tokens': out,
                         **timing_metrics(start, token_times, finished),
                         'expert_io_delta': {k: v-before_stats.get('expert_io',{}).get(k,0)
                                             for k,v in stats.get('expert_io',{}).items()},
                         'stats': stats})
    finally:
        lib.e0_engine_destroy(h)
    Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')


def supervised(args, mode, folder):
    out = folder / (mode + '.json')
    log = folder / (mode + '.log')
    cmd = [sys.executable, __file__, '--child', mode, '--dylib', args.reference_dylib if mode == 'reference' else args.dylib,
           '--model-dir', args.model_dir, '--tier', args.tier, '--output', str(out),
           '--max-new-tokens', str(args.max_new_tokens), '--timeout', str(args.timeout)]
    cmd += ['--prompt-tokens',str(args.prompt_tokens)]
    if args.packed_experts_dir:
        cmd += ['--packed-experts-dir',args.packed_experts_dir]
    if args.profile_restart and mode in ('adaptive','profiled'):
        cmd += ['--profile-path', str(folder / 'expert-profile.json')]
    if args.overlap_prefill:
        cmd += ['--overlap-prefill']
    peak = 0
    began = time.monotonic()
    with log.open('w') as stream:
        p = subprocess.Popen(cmd, stdout=stream, stderr=stream, start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic() - began > args.timeout:
                    raise RuntimeError(f'{mode}: timeout; see {log}')
                rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(p.pid)],
                                     capture_output=True, text=True, check=False).stdout.strip()
                if rss:
                    peak = max(peak, int(rss) * 1024)
                    if peak >= int(7.5 * GIB):
                        raise RuntimeError(f'{mode}: watchdog reached 7.5 GiB RSS')
                time.sleep(0.25)
            if p.returncode:
                raise RuntimeError(f'{mode}: child exited {p.returncode}; see {log}')
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    return {'sampled_rss_peak_bytes': peak, 'rows': json.loads(out.read_text())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dylib', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--reference-dylib', help='Optional unmodified engine for full output parity')
    ap.add_argument('--tier', choices=['edge0-35b', 'edge0-8b'], default='edge0-35b')
    ap.add_argument('--output', default='edge0-kindi-mac-check')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--max-new-tokens', type=int, default=4,
                    help='4 for smoke verification; 64 or more for speed measurement (max 256)')
    ap.add_argument('--prompt-tokens',type=int,default=2200,
                    help='Shared synthetic prompt length; 2100..131072. Larger contexts may fail the memory guard.')
    ap.add_argument('--packed-experts-dir',default='',help='Optional complete sidecar directory from pack_experts.py')
    ap.add_argument('--overlap-prefill',action='store_true',
                    help='Opt-in 35B resident-work overlap with bounded cold-expert graph evaluation')
    ap.add_argument('--adaptive-cache',action='store_true',help='Compare adaptive residency against unchanged LRU arms')
    ap.add_argument('--profile-restart',action='store_true',help='Also save a fresh adaptive profile and test a new worker loading it')
    ap.add_argument('--profile-path',default='',help=argparse.SUPPRESS)
    ap.add_argument('--trials',type=int,default=2,help='Alternate mode order across independent trials (1..4)')
    ap.add_argument('--child', choices=['reference', 'baseline', 'io', 'cached', 'packed', 'adaptive', 'profiled'], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not 1 <= args.max_new_tokens <= 256 or args.timeout <= 0:
        ap.error('max-new-tokens must be 1..256 and timeout must be positive')
    if not 2100 <= args.prompt_tokens <= 131072 or not 1 <= args.trials <= 4:
        ap.error('prompt-tokens must be 2100..131072 and trials 1..4')
    if args.profile_restart:
        args.adaptive_cache = True
    if args.child == 'profiled' and not Path(args.profile_path).is_file():
        ap.error('profiled arm requires a previously saved profile')
    if sys.platform != 'darwin':
        ap.error('Metal verification must run on the target Mac; Linux results would not validate it')
    if args.child:
        evaluate(args)
        return
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    modes = (['reference'] if args.reference_dylib else []) + ['baseline', 'io', 'cached']
    if args.packed_experts_dir:
        modes.append('packed')
    if args.adaptive_cache and not args.profile_restart:
        modes.append('adaptive')
    result = {'status': 'incomplete', 'trials': [], 'prompt_tokens': args.prompt_tokens,
              'overlap_prefill':args.overlap_prefill, 'adaptive_cache':args.adaptive_cache,
              'profile_restart':args.profile_restart,
              'note': 'Alternating mode order; no OS page-cache flush; sampled memory, not a hard ceiling.'}
    try:
        for trial in range(args.trials):
            trial_dir = folder / f'trial-{trial+1}'
            trial_dir.mkdir()
            run = {}
            result['trials'].append(run)
            ordered = modes if trial % 2 == 0 else list(reversed(modes))
            # Dependent restart test follows its fresh producer; never reuse another trial's file.
            if args.profile_restart:
                ordered = ordered + ['adaptive']
            if args.profile_restart:
                ordered = ordered + ['profiled']
            for mode in ordered:
                print(f'Trial {trial+1}: {mode}, one worker, 7.5 GiB RSS watchdog', flush=True)
                run[mode] = supervised(args, mode, trial_dir)
                for row in run[mode]['rows']:
                    rate = row['decode_tokens_per_second']
                    ttft = row['first_token_seconds']
                    speed = 'n/a' if rate is None else f'{rate:.2f}'
                    latency = 'n/a' if ttft is None else f'{ttft:.2f}s'
                    print(f"{mode:9} {row['case']:14} decode={speed} tokens/s "
                          f"TTFT={latency} total={row['wall_seconds']:.2f}s "
                          f"tokens={row['output_tokens']}"
                          + (' [short sample]' if row['short_sample'] else ''), flush=True)
            run['checks'] = check_modes(run)
        result['status'] = 'pass' if all(all(t['checks'].values()) for t in result['trials']) else 'fail'
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status':result['status'], 'checks':[t['checks'] for t in result['trials']]}, indent=2))
    if result['status'] != 'pass':
        raise SystemExit(1)


def check_modes(run):
    """Require every case in every arm; missing rows must never pass via zip()."""
    cases = ['initial', 'repeat', 'edited_tail', 'changed_prefix']
    modes = [k for k in run if k != 'checks']
    shape_ok = all([r['case'] for r in run[m]['rows']] == cases for m in modes)
    checks = {'complete_cases':shape_ok}
    if not shape_ok:
        return checks
    base = run['baseline']['rows']
    checks['token_parity'] = all(bool(a['tokens']) and a['tokens'] == b['tokens']
        for mode in modes for a,b in zip(base,run[mode]['rows']))
    for mode in ('cached', 'packed', 'adaptive', 'profiled'):
        if mode not in run:
            continue
        rows = run[mode]['rows']
        checks[mode+'_repeat_hit'] = rows[1]['stats']['prefix_cache']['last_cached_tokens'] > 0
        checks[mode+'_edited_prefix_hit'] = rows[2]['stats']['prefix_cache']['last_cached_tokens'] > 0
        checks[mode+'_changed_prefix_miss'] = rows[3]['stats']['prefix_cache']['last_cached_tokens'] == 0
    rows = [r for m in modes if m != 'reference' for r in run[m]['rows']]
    checks['observed_memory_ok'] = all(0 < r['stats']['memory']['resident_peak_bytes'] <= 8*GIB and
        0 < r['stats']['memory']['footprint_bytes'] <= 8*GIB for r in rows)
    checks['managed_budget_ok'] = all(r['stats']['joint_budget']['managed_peak_bytes'] <=
        r['stats']['joint_budget']['managed_limit_bytes'] for r in rows)
    if 'packed' in run:
        checks['packed_path_exercised'] = any(r['stats']['expert_io']['packed_read_calls'] > 0
                                             for r in run['packed']['rows'])
    for mode in ('adaptive','profiled'):
        if mode not in run:
            continue
        stats=[r['stats'].get('expert_residency',{}) for r in run[mode]['rows']]
        checks[mode+'_observed'] = all(s.get('enabled',False) for s in stats) and stats[-1].get('observations',0)>0
        checks[mode+'_profile_state'] = all((s.get('loaded_entries',-1)>0 if mode=='profiled' else s.get('loaded_entries',-1)==0) for s in stats)
    return checks

if __name__ == '__main__':
    main()
