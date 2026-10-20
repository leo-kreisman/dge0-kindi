#!/usr/bin/env python3
"""Apply the experimental Mac runtime patch to its pinned Edge0 revision."""
import argparse
from pathlib import Path
import subprocess

BASE = '85b2a03b2a05f66a6526ff904e820dfb3509a7d9'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkout', type=Path)
    args = ap.parse_args()
    root = args.checkout.resolve()
    patch = Path(__file__).with_name('edge0-kindi.patch').resolve()
    def git(*argv, check=True):
        return subprocess.run(['git', '-C', str(root), *argv], text=True,
                              capture_output=True, check=check)
    if git('rev-parse', 'HEAD').stdout.strip() != BASE:
        ap.error(f'expected Edge0 revision {BASE}; use a separate checkout/worktree')
    if git('apply', '--reverse', '--check', str(patch), check=False).returncode == 0:
        print('Patch already applied.')
        return
    check = git('apply', '--check', str(patch), check=False)
    if check.returncode:
        raise SystemExit('Patch does not apply cleanly; no files changed:\n' + check.stderr)
    git('apply', str(patch))
    print('Applied experimental Mac runtime patch. No build or inference started.')


if __name__ == '__main__':
    main()
