"""RGB3 closed-loop baseline. Only public observations enter the policy."""
import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from robolab_runner.policy.handler import PolicyHandler, serve_policy

SYSTEM = '''You control a mobile dual-arm robot through three current RGB cameras and measured state. Work toward the task, making incremental observable progress. Images are head, left wrist, right wrist. Never assume an object is held or a door locked without visual evidence. Avoid collisions, pushing through resistance, dropping objects, and repeated failed pushes. If blocked, retreat and reobserve.
Coordinates: right-handed +X forward, +Y left, +Z up, meters. Waist pose is relative to Base; hand poses are relative to waist (Floating). Quaternion WXYZ. State indices: waist xyz 0:3, quat 3:7; left hand xyz 7:10 quat 10:14 opening14; right hand xyz15:18 quat18:22 opening22; measured base vx,vy,wz 23:26. Opening 0 closed,1 open. State is NOT world position or joint angles. Wrench data unavailable. Head image is egocentric: head-image left is robot left (+Y), head-image right is robot right (-Y); positive base yaw turns left and negative yaw turns right. Use the HEAD image as the primary reference for base turning. Wrist cameras rotate with the hands: wrist-image left is NOT necessarily robot left. Do not choose base turn direction from wrist-image sides alone.
Choose the next 2 seconds of motion based on images, current state and previous plan. Locate the source object or mechanism before moving toward the destination. When it is not visible, scan in place toward relevant visible work surfaces; do not repeatedly drive backward into unseen space. For navigation keep arms safe. Near targets, stop base and visually align the appropriate hand, open gripper, approach then close and lift; refine across observations. Move only when justified by current images. You may adjust waist height gently. Do not use task IDs or imagined world coordinates.
Use the last observations and measured changes to choose one stage: search, approach, align, manipulate, verify, or recover. If the source/handle is already near the robot and visible, stop navigation and make a small hand alignment action instead of continuing to scan. Use the supplied measured_progress_since_last_plan; a nonzero hand delta means the hand moved even if image differences are subtle. Do not call it stuck merely because the target is not yet reached. Do not infer world heading from the waist quaternion. Compare the newest measured state to previous states: commanded motion is not evidence of achieved motion. Repeating a plan without visible progress calls for a different direction, view, or smaller motion. A gripper may close only when its fingers visually straddle the object or handle; otherwise align first. During manipulation keep the base stopped unless repositioning is necessary. Complete intermediate task stages rather than simply facing the target. These rules apply to any natural-language task and both embodiments.
Return one JSON object only, with all keys:
{"stage":"search|approach|align|manipulate|verify|recover", "reason":"brief visual evidence, actual progress since last plan, and next subgoal", "waist_dz":0.0, "left_dxyz":[0,0,0], "right_dxyz":[0,0,0], "left_rotvec":[0,0,0], "right_rotvec":[0,0,0], "left_open":1.0,"right_open":1.0,"base_velocity":[0,0,0]}
Hand displacements are relative to CURRENT hand position, in Floating coordinates, each component within +-0.12m. Rotvec is relative rotation about Floating axes in radians, components within +-0.3. Waist dz within +-0.03m. Base velocity [vx,vy,wz] is absolute desired m/s,m/s,rad/s within +-0.15,+-0.1,+-0.25. Set unchanged grippers to current values. Use cautious small steps for manipulation, larger steps for clear navigation. Do not claim success based solely on commanded actions.'''

def mul(a,b):
    w,x,y,z=a; W,X,Y,Z=b
    return np.array([w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W])

def quat(v):
    angle=np.linalg.norm(v)
    return np.r_[np.cos(angle/2),v*np.sin(angle/2)/angle] if angle>1e-9 else np.array([1.,0,0,0])

def qdelta(now,target,horizon=1):
    now=now/np.linalg.norm(now); target=target/np.linalg.norm(target)
    q=mul(target,now*np.array([1,-1,-1,-1])); q=q/np.linalg.norm(q)
    if q[0]<0:q=-q
    angle=2*np.arccos(np.clip(q[0],-1,1))
    return quat(q[1:]/max(np.linalg.norm(q[1:]),1e-9)*min(angle/horizon,0.015))

class VisionPolicy(PolicyHandler):
    policy_id='rgb3-qwen-closed-loop'
    policy_version='v4'
    def __init__(self, cache):
        super().__init__(); self.cache=Path(cache); self.cache.mkdir(parents=True,exist_ok=True); self.episodes={}
    def on_handshake(self,*,episode_id,payload):
        if payload['action_dim']!=26:raise ValueError('Action26 required')
        if payload['requested_action_horizon']!=1 or payload['execute_prefix_steps']!=1:raise ValueError('H=X=1 required')
    def reset(self,*,episode_id,payload):self.episodes[episode_id]={'next':120,'target':None,'memory':[],'last_step':0,'yaw':0.,'plan_yaw':0.}
    def end_episode(self,*,episode_id,payload):self.episodes.pop(episode_id,None)
    def infer_v3(self,observations,*,step,episode_id,effective_instruction,requested_action_horizon,execute_prefix_steps):
        s=np.asarray(observations['proprio_obs/canonical_observation40'],dtype=np.float32)
        a=np.zeros(26,np.float32);a[[3,10,18]]=1
        if not np.any(s):return a[None,:]  # Official doctor synthetic zero probe.
        if s.shape!=(40,) or not np.isfinite(s).all():raise ValueError('invalid state')
        e=self.episodes[episode_id]
        e['yaw']+=float(s[25])*max(0,step-e['last_step'])/120.
        e['last_step']=step
        if e['target'] is None:e['target']=s[:26].copy();e['target'][23:26]=0
        if step>=e['next']:
            plan=self.plan(observations,s,effective_instruction,e['memory'],step,base_yaw_delta=e['yaw']-e['plan_yaw'])
            e['plan_yaw']=e['yaw']
            t=s[:26].copy()
            t[2]+=np.clip(float(plan['waist_dz']),-.03,.03)
            for hand,i,g in [('left',7,14),('right',15,22)]:
                d=np.asarray(plan[hand+'_dxyz'],float); r=np.asarray(plan[hand+'_rotvec'],float)
                if d.shape!=(3,) or r.shape!=(3,) or not np.isfinite(np.r_[d,r]).all():raise ValueError('invalid pose plan')
                t[i:i+3]+=np.clip(d,-.12,.12)
                t[i+3:i+7]=mul(quat(np.clip(r,-.3,.3)),s[i+3:i+7])
                t[g]=np.clip(float(plan[hand+'_open']),0,1)
            v=np.asarray(plan['base_velocity'],float)
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('invalid velocity')
            t[23:26]=np.clip(v,[-.15,-.1,-.25],[.15,.1,.25])
            if not np.isfinite(t).all():raise ValueError('invalid target')
            e['target']=t;e['next']=step+240
            e['memory']=(e['memory']+[{'step':step,'plan':plan,'state':s[:26].tolist()}])[-4:]
        t=e['target']
        for i in [0,7,15]:
            a[i:i+3]=np.clip((t[i:i+3]-s[i:i+3])/8,-.003,.003)
            a[i+3:i+7]=qdelta(s[i+3:i+7],t[i+3:i+7],8)
        a[[14,22]]=np.clip((t[[14,22]]-s[[14,22]])/8,-.02,.02)
        # Base slots encode a velocity target minus measured velocity, not acceleration.
        # The official swerve controller already slew-limits acceleration internally.
        a[23:26]=t[23:26]-s[23:26]
        return a[None,:]
    def plan(self,obs,state,instruction,memory,step,base_yaw_delta=0.):
        previous=np.asarray(memory[-1]['state'],float) if memory else state
        def named(s):
            return {'waist_in_base_xyz':s[0:3].tolist(),'waist_in_base_quat_wxyz':s[3:7].tolist(),
                    'left_in_floating_xyz':s[7:10].tolist(),'left_quat_wxyz':s[10:14].tolist(),'left_open':float(s[14]),
                    'right_in_floating_xyz':s[15:18].tolist(),'right_quat_wxyz':s[18:22].tolist(),'right_open':float(s[22]),
                    'measured_base_velocity_m_s_rad_s':s[23:26].tolist()}
        progress={'left_hand_delta_m':(state[7:10]-previous[7:10]).tolist(),
                  'right_hand_delta_m':(state[15:18]-previous[15:18]).tolist(),
                  'gripper_opening_delta':(state[[14,22]]-previous[[14,22]]).tolist(),
                  'approx_base_rotation_rad':base_yaw_delta,
                  'note':'Rotation integrated from measured angular velocity; drift possible. Waist quaternion is relative to base, NOT world heading. Joint angles and object contact/weight are unavailable.'}
        history=[{'step':m['step'],'plan':m['plan'],'measured_state':named(np.asarray(m['state']))} for m in memory]
        content=[{'type':'text','text':json.dumps({'instruction':instruction,'measured_state':named(state),'measured_progress_since_last_plan':progress,'previous_plans':history,'step':step})}]
        for name in ['head_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']:
            im=Image.fromarray(obs['image_obs/'+name]); im.thumbnail((640,360)); b=io.BytesIO();im.save(b,format='PNG')
            content.extend([{'type':'text','text':name},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()}}])
        body={'model':os.environ.get('VLM_MODEL','qwen3.8-omni-flash'),'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],'temperature':0,'reasoning_effort':'low','max_tokens':1400,'response_format':{'type':'json_object'}}
        key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(); path=self.cache/(key+'.json')
        request_path=self.cache/(key+'.request.json')
        if not request_path.exists():request_path.write_text(json.dumps(body,ensure_ascii=False))
        if path.exists():return json.loads(path.read_text())['plan']
        if os.environ.get('REPLAY_ONLY')=='1':raise RuntimeError('Missing recorded model response: '+key)
        last=None
        for attempt in range(3):
            try:
                r=requests.post(os.environ['VLM_BASE_URL'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+os.environ['VLM_API_KEY']},json=body,timeout=70);r.raise_for_status();data=r.json()
                plan=json.loads(data['choices'][0]['message']['content'])
                if not isinstance(plan,dict):raise ValueError('plan must be one JSON object')
                for field in ['waist_dz','left_dxyz','right_dxyz','left_rotvec','right_rotvec','left_open','right_open','base_velocity']:
                    if field not in plan:raise ValueError('missing plan field: '+field)
                    if not np.isfinite(np.asarray(plan[field],dtype=float)).all():raise ValueError('nonfinite plan')
                path.write_text(json.dumps({'plan':plan,'response':data,'request_sha256':key},ensure_ascii=False))
                return plan
            except (requests.RequestException,ValueError,KeyError) as ex:
                last=ex;time.sleep(attempt+1)
        raise RuntimeError('VLM request failed') from last

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--endpoint',default='tcp://127.0.0.1:5562');p.add_argument('--cache',required=True);args=p.parse_args()
    serve_policy(VisionPolicy(args.cache),args.endpoint,max_message_bytes=67108864)
