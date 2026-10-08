"""Same-prefix teacher replay; scalar diagnostics only, no vector archive."""
import gc,collections
from core import *
from common import load_teacher,render,trim_generated

def main():
    stop=json.loads((RUN/'budget.json').read_text())['analysis_deadline_epoch'];plan=json.loads((ROOT/'plan.json').read_text())
    evaluation=read_jsonl(MAIN/'evaluation.jsonl');selected=[]
    for domain in ['math','code']:selected.extend([r for r in evaluation if r['split']=='eval_validation' and r['domain']==domain][:10])
    # The existing corpus uses eval_val; accept its explicit established spelling.
    if not selected:
        for domain in ['math','code']:selected.extend([r for r in evaluation if r['split']=='eval_val' and r['domain']==domain][:10])
    assert len(selected)==20,collections.Counter(r['split'] for r in evaluation)
    jobs=[j for j in plan['jobs'][:6] if j['config']['kind'] in ['silu','silu2']]
    atomic_json(RUN/'trajectory_plan.json',{'jobs':[j['name'] for j in jobs],'ids':[r['id'] for r in selected],'prefixes':['teacher','adapter'],'positions':'predicting each continuation token; prefix differs only when distance_from_first_divergence > 0','selection':'predeclared first seed, first 10 existing validation questions per domain','not_causal_proof':True})
    output=RUN/'trajectory_results.jsonl';rows=read_jsonl(output) if output.exists() else [];done={(r['configuration'],r['id']) for r in rows}
    if len(done)==80:
        event('trajectory_already_complete',completed=80,planned=80);return
    model,tok=load_teacher();text=model.model.language_model;original_norm=text.norm;types=list(text.config.layer_types)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);teacher_cache={}
    def depth(d,probe=None):
        text.norm=original_norm if probe is None else ExitNorm(probe,original_norm)
        for c in [text.config,model.config.text_config]:c.num_hidden_layers=d;c.layer_types=types[:d]
    def generate(prompt):
        encoded=tok(render(tok,prompt),return_tensors='pt').to('cuda')
        with torch.no_grad():seq=model.generate(**encoded,do_sample=False,max_new_tokens=256,use_cache=True,pad_token_id=tok.pad_token_id)
        return encoded.input_ids[0].tolist(),trim_generated(seq[0,encoded.input_ids.shape[1]:].tolist(),eos)[0]
    def normalized(x):return x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+original_norm.eps)*(1+original_norm.weight.float())
    def replay(prefix,continuation,d,probe):
        if not continuation:return []
        depth(64);vectors={};handles=[];plen=len(prefix)
        for index in [d,64]:
            def hook(_m,_a,out,index=index):
                h=out[0] if isinstance(out,tuple) else out;vectors[index]=h[0,plen-1:,:].detach()
            handles.append(text.layers[index-1].register_forward_hook(hook))
        try:
            with torch.no_grad():result=text(input_ids=torch.tensor([prefix+continuation[:-1]],device='cuda'),use_cache=False)
            del result
        finally:
            for h in handles:h.remove()
        metrics=[]
        with torch.no_grad():
            for start in range(0,len(continuation),16):
                x=vectors[d][start:start+16];y=vectors[64][start:start+16]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=probe(x)
                mse=(normalized(pred)-normalized(y)).square().mean(-1).tolist()
                teacher_top=model.lm_head(original_norm(y)).argmax(-1).tolist()
                adapter_top=model.lm_head(original_norm(pred.to(y.dtype))).argmax(-1).tolist()
                for k,(m,t,a) in enumerate(zip(mse,teacher_top,adapter_top)):
                    metrics.append({'position':start+k,'normalized_mse':m,'teacher_top1':t,'adapter_top1':a,'same_prefix_top1_agreement':t==a})
        return metrics
    for job in jobs:
        path=RUN/'probes'/f"{job['name']}.pt"
        if not path.exists():continue
        adapter,ck=load_probe(path)
        for row in selected:
            if (job['name'],row['id']) in done:continue
            if time.time()>stop-120:break
            if row['id'] not in teacher_cache:
                depth(64);teacher_cache[row['id']]=generate(row['prompt'])
            prefix,teacher=teacher_cache[row['id']];depth(job['source'],adapter);adapter_prefix,generated=generate(row['prompt']);assert prefix==adapter_prefix
            divergence=next((i for i,(a,b) in enumerate(zip(teacher,generated)) if a!=b),None)
            if divergence is None and len(teacher)!=len(generated):divergence=min(len(teacher),len(generated))
            trajectories={kind:replay(prefix,ids,job['source'],adapter) for kind,ids in [('teacher',teacher),('adapter',generated)]}
            for metrics in trajectories.values():
                for m in metrics:m['distance_from_first_divergence']=None if divergence is None else m['position']-divergence
            record={'configuration':job['name'],'id':row['id'],'domain':row['domain'],'first_divergence':divergence,'sequence_exact_agreement':teacher==generated,'teacher_tokens':teacher,'adapter_tokens':generated,'teacher_answer':tok.decode(teacher,skip_special_tokens=True),'adapter_answer':tok.decode(generated,skip_special_tokens=True),'trajectories':trajectories}
            with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            rows.append(record);done.add((job['name'],row['id']));event('trajectory',completed=len(done),planned=80)
        text.norm=original_norm;del adapter,ck;gc.collect();torch.cuda.empty_cache()
    atomic_json(RUN/'trajectory_summary.json',{'completed':len(done)==80,'completed_pairs':len(done),'planned_pairs':80,'configurations':4,'questions_per_configuration':20,'both_prefix_types':True})
if __name__=='__main__':main()
