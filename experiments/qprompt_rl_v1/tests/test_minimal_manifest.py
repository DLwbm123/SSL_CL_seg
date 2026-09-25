"""Generated fixtures only. No real HDF5, SSH, GPU or optimizer is used."""
import csv
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1] / 'migration/minimal_manifest.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.code, self.data, self.assets = [self.root/x for x in ('src_code','src_data','src_assets')]
        for p in (self.code,self.data,self.assets): p.mkdir()
        (self.code/'README.md').write_text('synthetic code fixture\n')
        (self.code/'module.py').write_text('VALUE = 1\n')
        subprocess.run(['git','init','-q',str(self.code)], check=True)
        subprocess.run(['git','-C',str(self.code),'add','.'], check=True)
        subprocess.run(['git','-C',str(self.code),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
        commit=subprocess.check_output(['git','-C',str(self.code),'rev-parse','HEAD'],text=True).strip()
        rows=[]
        for dom,(nl,nu) in M.COUNTS.items():
            for role,n in [('train_labeled',nl),('train_unlabeled',nu),('val',2),('test',2)]:
                for i in range(n):
                    cid=f'{dom}_{role}_{i}'; image=f'images/{cid}.h5'
                    p=self.data/image;p.parent.mkdir(exist_ok=True);p.write_bytes(('generated-image:'+cid).encode())
                    label='' if role=='train_unlabeled' else f'labels/{cid}.h5'
                    if label:
                        q=self.data/label;q.parent.mkdir(exist_ok=True);q.write_bytes(('generated-label:'+cid).encode())
                    rows.append(dict(dataset='fundus',case_id=cid,patient_id=cid,site_or_vendor=dom,
                                     primary_20pct_split=role,split_seed='0',image_h5_relpath=image,image_sha256=M.sha256(p),
                                     label_h5_relpath=label,label_sha256=M.sha256(self.data/label) if label else ''))
        self.rows=rows
        (self.data/M.MANIFEST).parent.mkdir(parents=True);(self.data/M.SPLIT).parent.mkdir(parents=True)
        self.cfg=dict(study_id='SSLCL_QPROMPT_RL_V1',phase='M1',roots=dict(code=str(self.code),data=str(self.data),assets=str(self.assets)),code_commit=commit,
                      code_files=[dict(path='README.md',reason='fixture'),dict(path='module.py',reason='fixture')],asset_files=[])
        self.config=self.root/'cfg.json'; self.freeze_rows()
    def tearDown(self): self.tmp.cleanup()
    def freeze_rows(self):
        with (self.data/M.MANIFEST).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        (self.data/M.SPLIT).write_text(json.dumps(dict(seed=0,records=self.rows)))
        self.cfg.update(manifest_sha256=M.sha256(self.data/M.MANIFEST),split_sha256=M.sha256(self.data/M.SPLIT));self.save_cfg()
    def save_cfg(self): self.config.write_text(json.dumps(self.cfg))
    def make(self,phase='M1'):
        self.cfg['phase']=phase;self.save_cfg();return M.build(self.config,self.root/('out_'+phase))
    def stage(self,m):
        dest=self.root/'staging';dest.mkdir()
        for x in m['files']:
            p=dest/x['target_relative'];p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(m['roots'][x['root_id']])/x['source_relative'],p)
        return dest
    def test_m1_role_minimality(self):
        m=self.make();roles={(x['domain'],x['role']) for x in m['files'] if x['domain']}
        self.assertEqual(roles,{(d,r) for d in M.DOMAINS[1:] for r in ('train_labeled','val')})
        self.assertEqual(m['records']['RIM_ONE_r3/train_labeled'],16)
    def test_split_metadata_and_payload_roots(self):
        metadata=self.root/'src_metadata';metadata.mkdir()
        for relative in (M.MANIFEST,M.SPLIT):
            destination=metadata/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(self.data/relative,destination)
        self.cfg['metadata_root']=str(metadata)
        m=self.make();stage=self.stage(m)
        self.assertEqual(M.verify(m,stage)['status'],'VERIFIED_BYTES_ONLY')
        self.assertTrue((stage/'data'/M.MANIFEST).is_file())
        self.assertTrue((stage/'data'/M.SPLIT).is_file())
        self.assertEqual({x['root_id'] for x in m['files'] if x['role']=='metadata'},{'metadata'})
    def test_m2_never_copies_source_u_or_test(self):
        m=self.make('M2')
        for x in m['files']:
            self.assertNotEqual(x['role'],'test');self.assertFalse(x['domain']=='REFUGE' and x['role']=='train_unlabeled')
            if x['role']=='train_unlabeled':self.assertIn('/images/',x['target_relative'])
    def test_valid_stage_verify_and_idempotent_promote(self):
        m=self.make();s=self.stage(m);self.assertEqual(M.verify(m,s)['status'],'VERIFIED_BYTES_ONLY')
        d=self.root/'final';d.mkdir();a=M.promote(m,s,d);b=M.promote(m,s,d)
        self.assertEqual(a['copied'],m['file_count']);self.assertEqual(b['copied'],0)
        self.assertTrue((self.code/'module.py').exists())
    def test_payload_hash_tamper(self):
        r=next(r for r in self.rows if r['site_or_vendor']=='RIM_ONE_r3' and r['primary_20pct_split']=='train_labeled')
        (self.data/r['image_h5_relpath']).write_bytes(b'changed')
        with self.assertRaises(M.Refusal):self.make()
    def test_u_gt_metadata_refused_even_when_not_copied(self):
        r=next(r for r in self.rows if r['primary_20pct_split']=='train_unlabeled');r['label_h5_relpath']='forbidden.h5';self.freeze_rows()
        with self.assertRaises(M.Refusal):self.make()
    def test_path_traversal_refused(self):
        self.cfg['code_files']=[dict(path='../outside.py',reason='bad')]
        with self.assertRaises(M.Refusal):self.make()
    def test_sensitive_path_refused(self):
        self.cfg['code_files']=[dict(path='.env',reason='bad')]
        with self.assertRaises(M.Refusal):self.make()
    def test_directory_entry_refused(self):
        self.cfg['code_files']=[dict(path='.',reason='bad')]
        with self.assertRaises(M.Refusal):self.make()
    def test_symlink_payload_refused(self):
        r=next(r for r in self.rows if r['site_or_vendor']=='RIM_ONE_r3' and r['primary_20pct_split']=='train_labeled')
        p=self.data/r['image_h5_relpath'];p.unlink();p.symlink_to(self.code/'README.md')
        with self.assertRaises(M.Refusal):self.make()
    def test_extra_file_refused(self):
        m=self.make();s=self.stage(m);(s/'data/extra.h5').write_bytes(b'extra')
        with self.assertRaises(M.Refusal):M.verify(m,s)
    def test_target_conflict_never_overwritten(self):
        m=self.make();s=self.stage(m);d=self.root/'final';(d/'code').mkdir(parents=True)
        p=d/'code/module.py';p.write_bytes(b'keep me')
        with self.assertRaises(M.Refusal):M.promote(m,s,d)
        self.assertEqual(p.read_bytes(),b'keep me')
    def test_dirty_code_refused(self):
        (self.code/'module.py').write_text('modified')
        with self.assertRaises(M.Refusal):self.make()
    def test_manifest_digest_binding(self):
        m=self.make();p=self.root/'out_M1/MIGRATION_MANIFEST.private.json'
        self.assertEqual(M.load_manifest(p,M.sha256(p))['file_count'],m['file_count'])
        with self.assertRaises(M.Refusal):M.load_manifest(p,'0'*64)
    def test_canonical_count_mismatch(self):
        self.rows.pop(0);self.freeze_rows()
        with self.assertRaises(M.Refusal):self.make()
    def test_target_hash_tamper(self):
        m=self.make();s=self.stage(m);(s/'code/module.py').write_bytes(b'malicious')
        with self.assertRaises(M.Refusal):M.verify(m,s)

if __name__=='__main__':unittest.main(verbosity=2)
