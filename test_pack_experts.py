#!/usr/bin/env python3
"""Cross-language sidecar tests using the production native reader."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from pack_experts import pack, PARTS


class PackedTests(unittest.TestCase):
    reader = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model, self.out = root/'model', root/'packed'
        self.model.mkdir()
        # Two shards, mixed BF16/U32, unordered tensor names, exact real prefix.
        for shard in range(2):
            data, hdr = bytearray(), {}
            for i,(p,q) in enumerate(PARTS):
                if i % 2 != shard:
                    continue
                length = 8*16*16*(4 if q == 'weight' else 2)
                start = len(data)
                data.extend(bytes([i+1])*length)
                hdr[f'language_model.model.layers.0.mlp.switch_mlp.{p}.{q}'] = {
                    'shape':[8,16,16], 'dtype':'U32' if q == 'weight' else 'BF16',
                    'data_offsets':[start,len(data)]}
            raw=json.dumps(hdr).encode()
            raw+=b' '*((-len(raw))%8)
            (self.model/f'model-{shard}.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+data)
        pack(self.model,self.out,rate_mib=0)

    def native(self, success=True):
        p=subprocess.run([self.reader,str(self.model),str(self.out),'0'],capture_output=True,text=True)
        if success:
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertTrue(json.loads(p.stdout)['byte_parity'])
        else:
            self.assertNotEqual(p.returncode,0)
        return p

    def test_byte_exact(self): self.native()

    def test_corruption_rejected(self):
        p=self.out/'layer-0.bin'
        with p.open('r+b') as f:
            f.seek(10); f.write(b'\xff')
        self.assertIn('checksum',self.native(False).stderr)

    def test_truncation_rejected(self):
        with (self.out/'layer-0.bin').open('r+b') as f: f.truncate(10)
        self.native(False)

    def test_stale_source_rejected(self):
        p=self.model/'model-0.safetensors'; st=p.stat()
        os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+1000))
        self.assertIn('source changed',self.native(False).stderr)

    def test_layout_rejected(self):
        p=self.out/'layer-0.json'; j=json.loads(p.read_text())
        j['lengths'][0]=2**63; p.write_text(json.dumps(j))
        self.native(False)

    def test_existing_refused(self):
        with self.assertRaises(FileExistsError):pack(self.model,self.out,rate_mib=0)


if __name__=='__main__':
    PackedTests.reader=str(Path(sys.argv.pop(1)).resolve())
    unittest.main()
