"""Wide vector probes; previous experiment artifacts are read-only inputs."""
import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MAIN=ROOT.parent
sys.path.append(str(MAIN))
import torch
from torch import nn
from common import WIDTH,POSITIONS,SEED,read_jsonl,atomic_json
RUN=ROOT/'run';RUN.mkdir(exist_ok=True)
(RUN/'probes').mkdir(exist_ok=True)
ACT=MAIN/'activations'
def event(stage,**kw):
    r={'stage':stage,'epoch':time.time(),**kw};atomic_json(RUN/'status.json',r);print(json.dumps(r),flush=True)
def configurations():
    return [{'id':f'{kind}_w{width}','kind':kind,'hidden':width,'hidden_layers':2 if kind=='silu2' else 1}
            for width in [6144,8192,10240,12288] for kind in ['silu','gelu','silu2','swiglu']]
class WideProbe(nn.Module):
    def __init__(self,config,scale=1.0,width=WIDTH):
        super().__init__();self.config=config;self.register_buffer('scale',torch.tensor(float(scale)))
        h=config['hidden'];self.kind=config['kind']
        if self.kind=='affine':
            self.linear=nn.Linear(width,width);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias);return
        self.down=nn.Linear(width,h)
        if self.kind=='swiglu':self.gate=nn.Linear(width,h)
        if self.kind in ['silu2','linear2']:self.middle=nn.Linear(h,h)
        self.up=nn.Linear(h,width);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):
        x=x.float();z=x/self.scale
        if self.kind=='affine':return x+self.scale*self.linear(z).float()
        a=self.down(z)
        if self.kind=='gelu':a=torch.nn.functional.gelu(a)
        elif self.kind not in ['linear1','linear2']:a=torch.nn.functional.silu(a)
        if self.kind=='swiglu':a=a*self.gate(z)
        if self.kind=='silu2':a=torch.nn.functional.silu(self.middle(a))
        if self.kind=='linear2':a=self.middle(a)
        return x+self.scale*self.up(a).float()
def save_torch(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def load_probe(path):
    c=torch.load(path,map_location='cpu',weights_only=True);p=WideProbe(c['config'],c['scale'])
    p.load_state_dict(c['state_dict']);return p.cuda().eval().requires_grad_(False),c
class ExitNorm(nn.Module):
    def __init__(self,p,n):super().__init__();self.probe=p;self.norm=n
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):a=self.probe(x)
        return self.norm(a.to(x.dtype))
