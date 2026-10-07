"""64 wide FFNs, fair timed fitting followed by validation-selected refinement.
No patience early stopping. Fixed saved-vector supervision only.
"""
import gc,math,hashlib
import numpy as np
from safetensors.torch import load_file
from core import *
def main():
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    budget=json.loads((RUN/'budget.json').read_text());stop=budget['training_deadline_epoch']
    records=read_jsonl(MAIN/'run/collection.jsonl');rows=read_jsonl(MAIN/'prompts.jsonl')
    shape=tuple(json.loads((ACT/'layout.json').read_text())['shape'])
    counts=torch.zeros(shape[0],dtype=torch.long,device='cuda')
    for r in records:counts[r['index']]=r['count']
    qs={s:torch.tensor([r['index'] for r in records if r['split']==s],device='cuda') for s in ['train','validation','test']}
    splits={s:hashlib.sha256(json.dumps(q.tolist()).encode()).hexdigest() for s,q in qs.items()}
    weight=load_file(str(ACT/'norm.safetensors'))['weight'].float().cuda();eps=json.loads((ACT/'norm.json').read_text())['eps']
    def norm(x):return x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*(1+weight)
    cache={}
    def layer(i):
        if i not in cache:
            if len(cache)>=2:
                for k in list(cache):
                    if k!=64:del cache[k]
            mm=np.memmap(ACT/f'layer_{i:02d}.bf16',mode='r',dtype=np.uint16,shape=shape)
            cache[i]=torch.from_numpy(np.array(mm)).view(torch.bfloat16).reshape(-1,WIDTH).cuda()
        return cache[i]
    y=layer(64);configs=configurations()
    plan=RUN/'training_plan.json'
    if not plan.exists():
        # Reserve 30% of training time for eight longer refinement runs.
        available=max(0,stop-time.time()-240)*.78
        atomic_json(plan,{'configs':configs,'sources':[60,61,62,63],'split_hashes':splits,
            'screen_seconds_per_job':available*.70/64,'refine_seconds_per_job':available*.30/8,
            'batch_size':512,'eval_every_steps':1000,'eval_every_seconds':45,'lr':3e-4,'refine_lr':8e-5,
            'weight_decay':0,'loss':'MSE after frozen final RMSNorm','early_stopping':False,
            'schedule':'100-step warmup, cosine decay by active training time to 1% peak',
            'selection':'top two architectures per layer by vector validation MSE; no generation or test selection',
            'checkpoint_storage':'best float32 for four depth winners and requested SiLU10240; others BF16; rounding affects refinement initialization',
            'no_convergence_claim':'training bounded by user 8-hour budget; curves and terminal trend reported'})
    settings=json.loads(plan.read_text())
    result_path=RUN/'probe_results.jsonl'
    results=read_jsonl(result_path) if result_path.exists() else []
    done={r['name'] for r in results}
    def fit(source,conf,phase,seconds,initial=None):
        name=f'{phase}_s{source}_{conf["id"]}'
        if name in done:return
        if time.time()>stop-60:return
        x=layer(source);torch.manual_seed(SEED+source)
        if initial:
            ck=torch.load(initial,map_location='cpu',weights_only=True);scale=ck['scale']
        else:
            inds=qs['train'][:512]*POSITIONS;scale=x[inds].float().square().mean().sqrt().item()
        p=WideProbe(conf,scale).cuda()
        if initial:p.load_state_dict(ck['state_dict']);del ck
        opt=torch.optim.AdamW(p.parameters(),lr=settings['lr'],weight_decay=0,fused=True)
        step=0;active=0.;best=float('inf');best_step=0;curve=[]
        checkpoint=RUN/'active.pt';bestfile=RUN/'active_best.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['name']==name:
                p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);step=saved['step'];active=saved['active'];best=saved['best'];best_step=saved['best_step'];curve=saved['curve']
                torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        def state(cpu_dtype=None):return {k:v.detach().to(device='cpu',dtype=cpu_dtype or v.dtype).clone() for k,v in p.state_dict().items()}
        def evaluate(split):
            details=[]
            with torch.no_grad():
                for start in range(0,len(qs[split]),8):
                    group=qs[split][start:start+8].tolist()
                    ix=torch.cat([torch.arange(q*POSITIONS,q*POSITIONS+int(counts[q]),device='cuda') for q in group])
                    a=x[ix];b=y[ix]
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=p(a)
                    err=(norm(pred)-norm(b)).square().mean(-1);at=0
                    for q in group:
                        c=int(counts[q]);details.append({'id':rows[q]['id'],'domain':rows[q]['domain'],'mse':err[at:at+c].mean().item()});at+=c
            return {d:float(np.mean([r['mse'] for r in details if d=='all' or r['domain']==d])) for d in ['all','math','code','control']},details
        start=time.time();last_eval=active;last_resume=active;train_losses=[]
        # A best checkpoint is stored separately from resumable optimizer state.
        while True:
            if not curve or step%1000==0 or active-last_eval>=45 or active>=seconds or time.time()>stop-45:
                scores,_=evaluate('validation');v=scores['all'];last_eval=active
                curve.append({'step':step,'active_seconds':active,'validation_mse':v,'train_mse':float(np.mean(train_losses)) if train_losses else None,'lr':opt.param_groups[0]['lr']});train_losses=[]
                if v<best:
                    best=v;best_step=step;save_torch(bestfile,{'name':name,'state_dict':state()})
                if active-last_resume>=60 or active>=seconds or time.time()>stop-45:
                    save_torch(checkpoint,{'name':name,'model':state(),'optimizer':opt.state_dict(),'step':step,'active':active,'best':best,'best_step':best_step,'curve':curve,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()});last_resume=active
                event('wide_training',name=name,step=step,active_seconds=active,target_seconds=seconds,best_validation_mse=best,jobs_completed=len(done))
                if active>=seconds or time.time()>stop-45:break
            t=time.time();q=qs['train'][torch.randint(len(qs['train']),(settings['batch_size'],),device='cuda')]
            ix=q*POSITIONS+(torch.rand(len(q),device='cuda')*counts[q]).long()
            progress=min(1,active/max(1,seconds));lr=(settings['refine_lr'] if phase=='refine' else settings['lr'])*(.01+.99*.5*(1+math.cos(math.pi*progress)))*min(1,(step+1)/100)
            for group in opt.param_groups:group['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=p(x[ix])
            loss=(norm(pred)-norm(y[ix])).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss {name} step {step}')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1.0);opt.step()
            train_losses.append(loss.item());step+=1;active+=time.time()-t
        # Save compact candidates; preserve float32 for deployment candidates to avoid weight-rounding mismatch.
        p.load_state_dict(torch.load(bestfile,map_location='cpu',weights_only=True)['state_dict'])
        validation,_=evaluate('validation')
        save_torch(RUN/'probes'/f'{name}.pt',{'config':conf,'source':source,'scale':scale,'name':name,'state_dict':state(torch.bfloat16),'best_step':best_step})
        current_best=RUN/f'best_s{source}.json'
        better=not current_best.exists() or best<json.loads(current_best.read_text())['validation_mse']
        if better or conf['id']=='silu_w10240':
            targets=[]
            if better:targets.append(f'best_s{source}')
            if conf['id']=='silu_w10240':
                old=RUN/f'requested_s{source}.json'
                if not old.exists() or best<json.loads(old.read_text())['validation_mse']:targets.append(f'requested_s{source}')
            for target in targets:
                save_torch(RUN/'probes'/f'{target}.pt',{'config':conf,'source':source,'scale':scale,'name':name,'state_dict':state(),'best_step':best_step})
                atomic_json(RUN/f'{target}.json',{'name':name,'config':conf,'validation_mse':best,'path':f'probes/{target}.pt'})
        row={'name':name,'phase':phase,'source':source,'config':conf,'parameters':sum(v.numel() for v in p.parameters()),'steps':step,'best_step':best_step,'best_at_last':best_step==step,'active_seconds':active,'elapsed_seconds':time.time()-start,'validation':validation,'curve':curve,'time_limited':True}
        with result_path.open('a') as f:f.write(json.dumps(row)+'\n')
        results.append(row);done.add(name);checkpoint.unlink(missing_ok=True);bestfile.unlink(missing_ok=True)
        del p,opt;gc.collect();torch.cuda.empty_cache()
    # Fixed predeclared ordering; no generation or test based prioritization.
    for source in [60,61,62,63]:
        for conf in configs:fit(source,conf,'screen',settings['screen_seconds_per_job'])
    if not (RUN/'refinement_plan.json').exists():
        selected=[]
        for s in [60,61,62,63]:
            eligible=sorted([r for r in results if r['source']==s and r['phase']=='screen'],key=lambda r:r['validation']['all'])[:2]
            selected.extend(eligible)
        atomic_json(RUN/'refinement_plan.json',{'selected':selected,'selection_uses_test':False})
    for r in json.loads((RUN/'refinement_plan.json').read_text())['selected']:
        fit(r['source'],r['config'],'refine',settings['refine_seconds_per_job'],RUN/'probes'/f'{r["name"]}.pt')
    # Evaluate only chosen deployment candidates on vector test, after selection is locked.
    vector_test={}
    for s in [60,61,62,63]:
        x=layer(s)
        for label in ['best','requested']:
            path=RUN/'probes'/f'{label}_s{s}.pt'
            if not path.exists():continue
            p,ck=load_probe(path);errs=[]
            with torch.no_grad():
                for q in qs['test'].tolist():
                    ix=torch.arange(q*POSITIONS,q*POSITIONS+int(counts[q]),device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):a=p(x[ix])
                    errs.append({'id':rows[q]['id'],'domain':rows[q]['domain'],'mse':(norm(a)-norm(y[ix])).square().mean().item()})
            vector_test[f'{label}_s{s}']={'name':ck['name'],'mse':float(np.mean([v['mse'] for v in errs])),'questions':errs}
            del p;gc.collect();torch.cuda.empty_cache()
    atomic_json(RUN/'vector_test.json',vector_test)
    screens=sum(r['phase']=='screen' for r in results);refines=sum(r['phase']=='refine' for r in results)
    atomic_json(RUN/'training_summary.json',{'completed':screens==64 and refines==8,'screen_completed':screens,'screen_planned':64,'refine_completed':refines,'refine_planned':8,'vector_tests':len(vector_test),'best_at_last':sum(r['best_at_last'] for r in results)})
    event('wide_training_finished',screen_completed=screens,refine_completed=refines)
if __name__=='__main__':main()
