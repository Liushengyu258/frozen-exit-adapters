import hashlib,platform,subprocess
from core import *
from judge import self_test

def main():
    self_test();torch.set_num_threads(4);torch.manual_seed(1)
    checks={};x=torch.randn(9,8,device='cuda')
    for kind in ['affine','silu','silu2','linear1','linear2','gelu','swiglu']:
        p=WideProbe({'kind':kind,'hidden':16},2.,width=8).cuda()
        assert torch.equal(p(x),x),kind
        output=p.linear if kind=='affine' else p.up
        torch.nn.init.normal_(output.weight,std=.1)
        if kind in ['affine','linear1','linear2']:
            a=torch.randn_like(x);b=torch.randn_like(x)
            assert torch.allclose(p(a+b)-p(a)-p(b)+p(torch.zeros_like(x)),torch.zeros_like(x),atol=2e-6),kind
        with torch.autocast('cuda',dtype=torch.bfloat16):y=p(x)
        y.square().mean().backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters())
        checks[kind]={'identity_at_init':True,'finite_gradients':True,'parameters_small_test':sum(v.numel() for v in p.parameters())}
    import transformers
    plan=json.loads((ROOT/'plan.json').read_text());assert len(plan['jobs'])==24;assert len(set(j['name'] for j in plan['jobs']))==24
    assert torch.cuda.get_device_properties(0).total_memory>70*2**30
    files={str(f.relative_to(MAIN)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'plan.json',ROOT/'PROTOCOL.md',MAIN/'evaluation.jsonl',MAIN/'prompts.jsonl',MAIN/'run/collection.jsonl',MAIN/'judge.py',MAIN/'code_worker.py']}
    atomic_json(RUN/'engineering_verification.json',{'passed':True,'checks':checks,'hashes':files,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0)})
    atomic_json(RUN/'source_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
    (RUN/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('SUPPLEMENT_VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
