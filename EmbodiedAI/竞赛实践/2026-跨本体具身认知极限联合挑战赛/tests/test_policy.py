"""Contract smoke tests without API calls or simulation."""
import tempfile
from pathlib import Path
import numpy as np
from policy import VisionPolicy, qdelta

with tempfile.TemporaryDirectory() as cache:
    p=VisionPolicy(cache)
    p.reset(episode_id='test',payload={})
    s=np.zeros(40,np.float32);s[[3,10,18]]=1;s[[14,22]]=1
    obs={'proprio_obs/canonical_observation40':s}
    kwargs=dict(episode_id='test',effective_instruction='pick up a cup',requested_action_horizon=1,execute_prefix_steps=1)
    a=p.infer_v3(obs,step=0,**kwargs)
    assert a.shape==(1,26) and a.dtype==np.float32
    assert np.array_equal(a[0,[3,10,18]],[1,1,1])
    assert np.count_nonzero(a)==3
    # Antipodal quaternions describe the same physical orientation.
    assert np.allclose(qdelta(np.array([1.,0,0,0]),np.array([-1.,0,0,0])),[1,0,0,0])
    p.plan=lambda *args,**kw:dict(reason='test',waist_dz=0,left_dxyz=[.1,0,0],right_dxyz=[0,0,0],left_rotvec=[0,0,0],right_rotvec=[0,0,0],left_open=0,right_open=1,base_velocity=[0,0,0])
    first=p.infer_v3(obs,step=120,**kwargs)
    assert first[0,7]>0 and first[0,14]<0
    assert 0 < first[:,7].sum() <= .100001
    assert -1.000001 <= first[:,14].sum() < 0
    p.on_handshake(episode_id='test',payload={'action_dim':26,'requested_action_horizon':1,'execute_prefix_steps':1})
    # Once measured hand reaches target, position correction must vanish.
    s[7]=.1;s[14]=0
    reached=p.infer_v3(obs,step=121,**kwargs)
    assert abs(reached[0,7])<1e-6 and abs(reached[0,14])<1e-6
    assert np.isfinite(reached).all()
    p.episodes['test']['target'][23:26]=[.15,-.1,.25]
    s[23:26]=[.01,.02,-.01]
    base=p.infer_v3(obs,step=122,**kwargs)
    assert np.allclose(base[0,23:26]+s[23:26],[.15,-.1,.25])
print('PASS: Action26 contract, quaternion equivalence, measured-target feedback')

# Repeated feedback calls must preserve the desired decoded velocity as measurements change.
with tempfile.TemporaryDirectory() as cache:
    p=VisionPolicy(cache);p.reset(episode_id='feedback',payload={})
    s=np.zeros(40,np.float32);s[[3,10,18]]=1
    target=s.copy();target[23:26]=[.15,-.1,.25]
    p.episodes['feedback']['target']=target;p.episodes['feedback']['next']=1000
    for step in range(120,128):
        s[23:26]=np.array([.01,.02,-.01])*(step-119)
        a=p.infer_v3({'proprio_obs/canonical_observation40':s},step=step,episode_id='feedback',effective_instruction='test',requested_action_horizon=1,execute_prefix_steps=1)
        assert a.shape==(1,26)
        assert np.allclose(a[0,23:26]+s[23:26],target[23:26],atol=1e-6)
print('PASS: successive measured states preserve a fixed decoded velocity target')

# Exercise the real HTTP request builder with an in-process fake response.
import json
import os
from unittest.mock import patch
from types import SimpleNamespace

with tempfile.TemporaryDirectory() as cache:
    p=VisionPolicy(cache)
    state=np.zeros(40,np.float32);state[[3,10,18]]=1
    obs={'image_obs/'+name:np.zeros((16,16,3),dtype=np.uint8)
         for name in ['head_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']}
    plan=dict(stage='align',reason='fixture',waist_dz=0,left_dxyz=[0,0,0],right_dxyz=[0,0,0],left_rotvec=[0,0,0],right_rotvec=[0,0,0],left_open=1,right_open=1,base_velocity=[0,0,0])
    response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':json.dumps(plan)}}]})
    with patch.dict(os.environ,{'VLM_BASE_URL':'https://example.invalid/v1','VLM_MODEL':'fixture-model','VLM_API_KEY':'UNIT_TEST_ONLY'}),patch('policy.requests.post',return_value=response) as post:
        first=p.plan(obs,state,'pick up a cup',[],120)
        second=p.plan(obs,state,'pick up a cup',[],120)
        assert first==second==plan and post.call_count==1
        assert post.call_args.args[0]=='https://example.invalid/v1/chat/completions'
        body=post.call_args.kwargs['json']
        assert body['temperature']==0 and body['model']=='fixture-model'
        assert sum(x['type']=='image_url' for x in body['messages'][1]['content'])==3
        assert all('UNIT_TEST_ONLY' not in f.read_text() for f in Path(cache).glob('*.json'))
print('PASS: RGB3 request, environment credentials and response-cache reuse (mock HTTP only)')
