"""CPU tests of actual source methods with lightweight attention substitutes.

These tests DO NOT execute MMCV attention/CUDA, model construction or nuScenes.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import unittest
import warnings
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
MODULE=ROOT/'projects/mmdet3d_plugin/bevformer/modules'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mod=load(MODULE/'layer_gate.py','gate')
converter=load(ROOT/'tools/layergate/convert_checkpoint.py','converter')

def method(source,cls,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=copy.deepcopy(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name))
    fn.decorator_list=[]
    env={'torch':torch,'np':np,'copy':copy,'warnings':warnings,'fuse':mod.fuse}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'source_method','exec'),env)
    return env[name]

class Attention(nn.Module):
    def __init__(self,scale):super().__init__();self.scale=scale
    def forward(self,q,*args,**kw):return q*self.scale+torch.sin(q)
class FFN(nn.Module):
    def forward(self,q,identity=None):return q+q.tanh()*.1

def layer(forward,alpha):
    m=nn.Module();m.forward=forward.__get__(m)
    m.operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')
    m.attentions=nn.ModuleList([Attention(.8),Attention(1.1)])
    m.norms=nn.ModuleList([nn.LayerNorm(8) for _ in range(3)])
    m.ffns=nn.ModuleList([FFN()]);m.num_attn=2;m.pre_norm=False
    m.cross_model_weights_logit=nn.Parameter(torch.tensor(alpha))
    m.lidar_cross_attn_layer=Attention(.3)
    return m

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(MODULE/'encoder.py').read_text()
        old=subprocess.check_output(['git','show','3b84adad901a6d24075bc1c05cd51a0f316c4187:projects/mmdet3d_plugin/bevformer/modules/encoder.py'],cwd=ROOT,text=True)
        cls.old=staticmethod(method(old,'MM_BEVFormerLayer','forward'))
        cls.new=staticmethod(method(cls.source,'MM_BEVFormerLayer','forward'))
    def setUp(self):torch.manual_seed(44)
    def inputs(self):
        return dict(spatial_shapes=torch.tensor([[2,3]]),level_start_index=torch.tensor([0]),
            bev_h=2,bev_w=3,ref_2d=torch.zeros(3,6,1,2),lidar_bev_tokens=torch.randn(3,6,8))
    def test_actual_layer_legacy_and_external_parity(self):
        for prev in [None,torch.randn(3,6,8)]:
            for alpha in [-1.4,.2,1.1]:
                old,new=layer(self.old,alpha),layer(self.new,alpha)
                q=torch.randn(3,6,8);kw=self.inputs();kw['prev_bev']=prev
                a=old(q,**kw);b=new(q,**kw)
                c=new(q,fusion_camera_weight=torch.full((3,),torch.sigmoid(torch.tensor(alpha)).item()),**kw)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(a,c,rtol=0,atol=0)
    def test_actual_layer_per_sample_weights(self):
        m=layer(self.new,0.);q=torch.randn(3,6,8);kw=self.inputs();w=torch.tensor([0.,.4,1.])
        batched=m(q,fusion_camera_weight=w,**kw)
        for b in range(3):
            single={k:(v[b:b+1] if k in ('ref_2d','lidar_bev_tokens') else v) for k,v in kw.items()}
            torch.testing.assert_close(batched[b:b+1],m(q[b:b+1],fusion_camera_weight=w[b:b+1],**single),rtol=0,atol=0)
    def test_encoder_routes_each_column(self):
        forward=method(self.source,'BEVFormerEncoder','forward')
        getrefs=method(self.source,'BEVFormerEncoder','get_reference_points')
        e=nn.Module();e.layers=nn.ModuleList([layer(self.new,x) for x in [-1.,0.,.5,1.]])
        e.pc_range=[-1,-1,-1,1,1,1];e.num_points_in_pillar=4;e.return_intermediate=False
        e.get_reference_points=getrefs;e.point_sampling=lambda *args:(None,None)
        q=torch.randn(6,3,8);pos=torch.zeros_like(q);kw=self.inputs()
        weights=torch.rand(3,4)
        actual=forward(e,q,None,None,bev_pos=pos,img_metas=[{}]*3,shift=torch.zeros(3,2),
            encoder_fusion_weights=weights,**{k:v for k,v in kw.items() if k!='ref_2d'})
        expected=q.permute(1,0,2).contiguous()
        for i,l in enumerate(e.layers):expected=l(expected,fusion_camera_weight=weights[:,i],**kw)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def test_transformer_gate_selection_and_reset(self):
        source=(MODULE/'transformer.py').read_text();t=nn.Module()
        t.encoder=nn.Module();t.encoder.layers=nn.ModuleList([layer(self.new,x) for x in [-1.,0.,.5,1.]])
        for name in ['_fixed_encoder_logits','_encoder_weights','initialize_layer_gate_from_encoder']:
            setattr(t,name,method(source,'PerceptionTransformer',name).__get__(t))
        t.layer_gate=mod.LayerGate(256,64,torch.zeros(4))
        c=[torch.randn(3,6,256,2,3)];l=torch.randn(3,64,4,4)
        t.initialize_layer_gate_from_encoder()
        t.encoder_gate_mode='fixed_refactored';a=t._encoder_weights(c,l)
        t.encoder_gate_mode='layergate';b=t._encoder_weights(c,l)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(t.last_encoder_gates.requires_grad)
        self.assertIsNone(t._encoder_weights(c,None));self.assertIsNone(t.last_encoder_gates)
        t.encoder_gate_mode='fixed_legacy';self.assertIsNone(t._encoder_weights(c,l))
    def test_conversion_preserves_legacy_values_and_uses_logits(self):
        state={converter.PREFIX+f'encoder.layers.{i}.cross_model_weights_logit':torch.tensor(float(i)-2) for i in range(4)}
        state['some_head.weight']=torch.randn(2,3)
        checkpoint={'state_dict':state,'optimizer':{'fake':1}}
        out=converter.convert(checkpoint)
        for k,v in state.items():torch.testing.assert_close(v,out['state_dict'][k],rtol=0,atol=0)
        self.assertNotIn('optimizer',out)
        self.assertEqual(out['layergate_conversion']['added_parameters'],29380)
        g=mod.LayerGate(256,64,torch.zeros(4))
        prefix=converter.PREFIX+'layer_gate.'
        g.load_state_dict({k[len(prefix):]:v for k,v in out['state_dict'].items() if k.startswith(prefix)},strict=True)
        torch.testing.assert_close(g.output.bias,torch.tensor([-2.,-1.,0.,1.]),rtol=0,atol=0)
        with self.assertRaises(ValueError):converter.convert(out)
        with self.assertRaises(ValueError):converter.convert({'state_dict':{}})
    def test_init_weights_restores_zero_output_after_global_xavier(self):
        tree=ast.parse((MODULE/'transformer.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PerceptionTransformer')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_weights')
        # The reset must occur after the global initialization loop.
        self.assertIn('initialize_layer_gate_from_encoder',ast.unparse(fn.body[-1]))

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main(verbosity=2)
