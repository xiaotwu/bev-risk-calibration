"""CPU data-contract tests; fixtures are not detector evidence."""
import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/layergate'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
records=load('gate_records');analysis=load('analyze_gates')

class Tensor(np.ndarray):
    def __new__(cls,x):return np.asarray(x).view(cls)
    def detach(self):return self
    def cpu(self):return self
    def float(self):return self
    def new_tensor(self,x):return Tensor(x)
    def all(self,dim=None,**kw):return super().all(axis=dim,**kw)

def row(token,scene,w,n):
    r=dict(sample_token=token,scene_token=scene,lidar_point_count=n,lidar_in_range_point_count=n,
           point_count_definition='model_input_before_voxelization',in_range_definition='xyz_lower_inclusive_upper_exclusive',point_cloud_range=[0,0,0,10,10,10])
    r.update({f'gate_layer_{i+1}':v for i,v in enumerate(w)});return r

class RecordTests(unittest.TestCase):
    def test_exact_coverage(self):
        with TemporaryDirectory() as d:
            r=records.GateRecorder(Path(d)/'g.jsonl',['a','b'],2,[0,0,0,1,1,1])
            r.write('a','scene',[.2,.4],8,4);r.write('b','scene',[.3,.5],9,5)
            self.assertEqual(r.finalize()['sample_count'],2)
            self.assertEqual(json.loads((Path(d)/'g.jsonl.expected_tokens.json').read_text()),['a','b'])
    def test_duplicate_unknown_and_nan_rejected(self):
        with TemporaryDirectory() as d:
            r=records.GateRecorder(Path(d)/'g.jsonl',['a','b'],2,[0,0,0,1,1,1])
            r.write('a','scene',[.2,.4],8,4)
            for token,w in [('a',[.2,.4]),('x',[.2,.4]),('b',[float('nan'),.5])]:
                with self.assertRaises(ValueError):r.write(token,'scene',w,8,4)
            with self.assertRaises(ValueError):r.finalize()
            self.assertFalse((Path(d)/'g.jsonl.coverage.json').exists())
    def test_capture_counts_and_tokens(self):
        with TemporaryDirectory() as d:
            r=records.GateRecorder(Path(d)/'g.jsonl',['a'],2,[0,0,0,1,1,1])
            transformer=SimpleNamespace(encoder_gate_mode='layergate',last_encoder_gates=Tensor([[.2,.4]]))
            model=SimpleNamespace(pts_bbox_head=SimpleNamespace(transformer=transformer))
            r.capture(model,[dict(sample_idx='a',scene_token='s')],[[Tensor([[0,0,0],[1,0,0],[-1,0,0],[.5,.5,.5]])]])
            r.finalize();data=json.loads((Path(d)/'g.jsonl').read_text())
            self.assertEqual(data['lidar_point_count'],4);self.assertEqual(data['lidar_in_range_point_count'],2)
    def test_actual_detector_callback_preserves_output(self):
        source=ast.parse((ROOT/'projects/mmdet3d_plugin/bevformer/detectors/bevformer.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='simple_test');fn.decorator_list=[]
        env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'simple_test','exec'),env)
        calls=[];m=SimpleNamespace(extract_feat=lambda **kw:['feature'],simple_test_pts=lambda *args,**kw:('bev',['boxes']))
        expected=env['simple_test'](m,[{'sample_idx':'a'}],points=['p'])
        m.layergate_recorder=SimpleNamespace(capture=lambda *args:calls.append(args))
        observed=env['simple_test'](m,[{'sample_idx':'a'}],points=['p'])
        self.assertEqual(expected,observed);self.assertEqual(len(calls),1);self.assertEqual(calls[0][2],['p'])
    def test_statistics_and_undefined_correlation(self):
        rows=[row('a','s1',[.2,.5],1),row('b','s1',[.4,.5],2),row('c','s2',[.8,.5],3)]
        r=analysis.analyze(rows,['a','b','c'],[.3,.5]);a=r['layers'][0]
        self.assertAlmostEqual(a['mean'],1.4/3);self.assertAlmostEqual(a['scene_weighted_mean'],.55)
        self.assertAlmostEqual(a['density_spearman'],1)
        self.assertIsNone(r['layers'][1]['density_spearman']);self.assertIsNone(r['pearson_matrix'][1][1])
    def test_analysis_refuses_incomplete_or_confounded_logs(self):
        rows=[row('a','s',[.2,.5],2)]
        with self.assertRaises(ValueError):analysis.analyze(rows,['a','b'],[.3,.5])
        with self.assertRaises(ValueError):analysis.analyze(rows+rows,['a'],[.3,.5])
        second=row('b','s',[.4,.5],4);second['point_cloud_range']=[0,0,0,20,20,20]
        with self.assertRaises(ValueError):analysis.analyze(rows+[second],['a','b'],[.3,.5])

if __name__=='__main__':unittest.main()
