"""Fixed updates, independent sampling RNG, durable per-job recovery."""
import gc,math,hashlib,shutil
import numpy as np
from safetensors.torch import load_file
from core import *

def main():
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    settings=json.loads((ROOT/'plan.json').read_text());stop=json.loads((RUN/'budget.json').read_text())['training_deadline_epoch']
    records=read_jsonl(MAIN/'run/collection.jsonl');rows=read_jsonl(MAIN/'prompts.jsonl');shape=tuple(json.loads((ACT/'layout.json').read_text())['shape'])
    counts=torch.zeros(shape[0],dtype=torch.long,device='cuda')
    for r in records:counts[r['index']]=r['count']
    qs={s:torch.tensor([r['index'] for r in records if r['split']==s],device='cuda') for s in ['train','validation','test']}
    atomic_json(RUN/'split_hashes.json',{s:hashlib.sha256(json.dumps(q.tolist()).encode()).hexdigest() for s,q in qs.items()})
    weight=load_file(str(ACT/'norm.safetensors'))['weight'].float().cuda();eps=json.loads((ACT/'norm.json').read_text())['eps']
    def norm(x):return x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*(1+weight)
    cache={}
    def layer(i):
        if i not in cache:
            for k in list(cache):
                if k!=64:del cache[k]
            mm=np.memmap(ACT/f'layer_{i:02d}.bf16',mode='r',dtype=np.uint16,shape=shape)
            cache[i]=torch.from_numpy(np.array(mm)).view(torch.bfloat16).reshape(-1,WIDTH).cuda()
        return cache[i]
    y=layer(64);path=RUN/'probe_results.jsonl';results=read_jsonl(path) if path.exists() else [];done={r['name'] for r in results if r['completed']}
    for job in settings['jobs']:
        name=job['name']
        if name in done:continue
        if time.time()>stop-120:break
        if shutil.disk_usage(RUN).free<8*2**30:raise RuntimeError('Disk headroom below 8GiB')
        x=layer(job['source']);torch.manual_seed(job['seed']);rng=torch.Generator(device='cuda').manual_seed(job['seed']+100000)
        scale=x[qs['train'][:512]*POSITIONS].float().square().mean().sqrt().item();p=WideProbe(job['config'],scale).cuda()
        opt=torch.optim.AdamW(p.parameters(),lr=settings['lr'],weight_decay=0,fused=True)
        step=0;best=float('inf');best_step=0;curve=[];active=0.;resume=RUN/'active.pt';bestfile=RUN/'probes'/f'{name}.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=True)
            if saved['name']!=name:raise RuntimeError('Resume job mismatch: '+saved['name'])
            p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['sampling_rng']);step=saved['step'];best=saved['best'];best_step=saved['best_step'];curve=saved['curve'];active=saved['active_seconds'];del saved
        def state():return {k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
        def evaluate():
            details=[]
            with torch.no_grad():
                for start in range(0,len(qs['validation']),8):
                    group=qs['validation'][start:start+8].tolist();ix=torch.cat([torch.arange(q*POSITIONS,q*POSITIONS+int(counts[q]),device='cuda') for q in group])
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=p(x[ix])
                    err=(norm(pred)-norm(y[ix])).square().mean(-1);at=0
                    for q in group:
                        c=int(counts[q]);details.append((rows[q]['domain'],err[at:at+c].mean().item()));at+=c
            return {d:float(np.mean([v for domain,v in details if d=='all' or domain==d])) for d in ['all','math','code','control']}
        start=time.time();losses=[]
        while True:
            if not curve or step%settings['validation_every_steps']==0 or time.time()>stop-90:
                scores=evaluate();v=scores['all']
                point={'step':step,'active_seconds':active,'validation_mse':v,'validation':scores,'train_mse':float(np.mean(losses)) if losses else None,'lr':opt.param_groups[0]['lr']}
                if not curve or curve[-1]['step']!=step:curve.append(point)
                losses=[]
                if v<best:
                    best=v;best_step=step;save_torch(bestfile,{**job,'scale':scale,'state_dict':state(),'best_step':step,'validation':scores,'storage_dtype':'float32'})
                save_torch(resume,{'name':name,'model':state(),'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'step':step,'best':best,'best_step':best_step,'curve':curve,'active_seconds':active})
                atomic_json(RUN/(name+'_curve.json'),curve)
                event('fixed_step_training',name=name,step=step,target_steps=settings['steps'],jobs_completed=len(done),best_validation_mse=best,free_disk_gib=shutil.disk_usage(RUN).free/2**30)
                if step>=settings['steps'] or time.time()>stop-90:break
            t=time.time();q=qs['train'][torch.randint(len(qs['train']),(settings['batch_size'],),device='cuda',generator=rng)]
            ix=q*POSITIONS+(torch.rand(len(q),device='cuda',generator=rng)*counts[q]).long()
            lr=settings['lr']*(.01+.99*.5*(1+math.cos(math.pi*step/settings['steps'])))*min(1,(step+1)/settings['warmup_steps'])
            for g in opt.param_groups:g['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=p(x[ix])
            loss=(norm(pred)-norm(y[ix])).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss {name} {step}')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();losses.append(loss.item());step+=1;active+=time.time()-t
        completed=step==settings['steps']
        if completed:
            ck=torch.load(bestfile,map_location='cpu',weights_only=True)
            row={**job,'completed':True,'steps':step,'best_step':best_step,'best_at_last':best_step==step,'validation':ck['validation'],'parameters':sum(v.numel() for v in p.parameters()),'active_seconds':active,'elapsed_seconds_this_process':time.time()-start,'curve':curve}
            with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            done.add(name);results.append(row);resume.unlink(missing_ok=True);del ck
        del p,opt;gc.collect();torch.cuda.empty_cache()
        if not completed:break
    atomic_json(RUN/'training_summary.json',{'completed':len(done)==len(settings['jobs']),'completed_jobs':len(done),'planned_jobs':len(settings['jobs']),'fixed_steps':settings['steps']})
    event('training_finished',completed=len(done),planned=len(settings['jobs']))
if __name__=='__main__':main()
