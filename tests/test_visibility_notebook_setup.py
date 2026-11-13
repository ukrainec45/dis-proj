"""Verify the actual notebook initializer without Torch or network downloads."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('problem',[None,'missing','shape','unexpected'])
def test_complete_camp_checkpoint_guard_and_factory_restoration(monkeypatch,problem):
    notebook=json.loads(Path('notebooks/anyvisloc_visibility_quick.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(notebook['cells'][12]['source']))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='camp_from_complete_checkpoint')
    expected={'weight':np.zeros((2,3)),'bias':np.zeros(2)}
    checkpoint={f'model_1.convnext.{k}':np.ones_like(v) for k,v in expected.items()}
    if problem=='missing': checkpoint.pop('model_1.convnext.bias')
    if problem=='shape': checkpoint['model_1.convnext.bias']=np.ones(3)
    if problem=='unexpected': checkpoint['model_1.convnext.extra']=np.ones(1)
    calls=[]
    class Model:
        def state_dict(self): return expected
        def load_state_dict(self,state,strict):
            assert strict
            self.loaded=state
    def factory(name,*args,**kwargs):
        assert kwargs['pretrained'] is False
        calls.append(name)
        return Model()
    module=SimpleNamespace(create_model=factory)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(load=lambda *a,**kw:checkpoint))
    namespace=dict(contextmanager=contextmanager,sys=SimpleNamespace(path=[]),
                   importlib=SimpleNamespace(import_module=lambda name:module))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook-initializer','exec'),namespace)
    if problem is None:
        with namespace['camp_from_complete_checkpoint'](Path('/fixture'),Path('/fixture/camp.pth')):
            model=module.create_model('convnext_base',pretrained=True)
            np.testing.assert_array_equal(model.loaded['weight'],np.ones((2,3)))
    else:
        with pytest.raises(RuntimeError,match='CAMP checkpoint'):
            with namespace['camp_from_complete_checkpoint'](Path('/fixture'),Path('/fixture/camp.pth')):
                module.create_model('convnext_base',pretrained=True)
    assert calls==['convnext_base'] and module.create_model is factory
