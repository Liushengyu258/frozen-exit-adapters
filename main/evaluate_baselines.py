"""Held-out math exact match and sandboxed MBPP tests; validation-only selection."""
from collections import defaultdict
import gc
import json
import time
import numpy as np
import torch
from common import *
from judge import judge,self_test

def main():
    stop=deadline(6)
    self_test()
    evaluation=read_jsonl(ROOT/'evaluation.jsonl')
    supported_path=RUN/'supported_evaluation.json'
    if supported_path.exists():support=json.loads(supported_path.read_text())
    else:
        support={'included':[],'excluded':[]}
        for r in evaluation:
            if r['domain']=='code':
                check=judge(r,r['reference'])
                if not check['passed']:
                    support['excluded'].append({'id':r['id'],'reason':check});continue
            support['included'].append(r['id'])
        atomic_json(supported_path,support)
    evaluation=[r for r in evaluation if r['id'] in set(support['included'])]
    for split,minimum in [('eval_validation',20),('eval_test',50)]:
        for domain in ['math','code']:
            assert sum(r['split']==split and r['domain']==domain for r in evaluation)>=minimum,(split,domain,'insufficient supported evaluation items')
    configurations=[{'name':'teacher','depth':64,'probe':None}]+[{'name':f'identity_{d}','depth':d,'probe':None} for d in [60,63]]
    atomic_json(RUN/'generation_plan.json',{'configurations':configurations,'max_new_tokens':256,
        'decoding':'greedy, non-thinking, batch 4','validation_prompts':sum(r['split']=='eval_validation' for r in evaluation),
        'test_prompts':sum(r['split']=='eval_test' for r in evaluation),
        'selection':'fixed teacher and direct-exit baselines; no candidate selection',
        'memory_note':'suffix computation skipped and cache config shortened, but suffix weights kept resident to switch configurations; no deployment-memory claim'})
    output_path=RUN/'generation_results.jsonl'
    outputs=read_jsonl(output_path) if output_path.exists() else []
    done={(r['configuration'],r['id']) for r in outputs}
    model,tok=load_teacher();text=model.model.language_model
    original_norm=text.norm;original_types=list(text.config.layer_types)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    def set_config(config):
        depth=config['depth']
        for cc in [text.config,model.config.text_config]:
            cc.num_hidden_layers=depth;cc.layer_types=original_types[:depth]
        text.norm=original_norm
        gc.collect();torch.cuda.empty_cache()
    def run(config,split):
        todo=[r for r in evaluation if r['split']==split and (config['name'],r['id']) not in done]
        if not todo:return True
        if time.time()>stop-180:return False
        set_config(config)
        # Verify actual executed prefix, independent of cache configuration bookkeeping.
        observed=set();hooks=[]
        for i,block in enumerate(text.layers,1):
            def hook(_m,_a,_o,index=i):observed.add(index)
            hooks.append(block.register_forward_hook(hook))
        warm=tok(render(tok,'Return only the integer 2.'),return_tensors='pt').to('cuda')
        with torch.no_grad():model.generate(**warm,do_sample=False,max_new_tokens=2,use_cache=True,pad_token_id=tok.pad_token_id)
        for h in hooks:h.remove()
        assert observed==set(range(1,config['depth']+1)),(config,observed)
        for offset in range(0,len(todo),4):
            if time.time()>stop-120:return False
            batch=todo[offset:offset+4]
            encoded=tok([render(tok,r['prompt']) for r in batch],padding=True,return_tensors='pt').to('cuda')
            t0=time.time()
            with torch.no_grad():seq=model.generate(**encoded,do_sample=False,max_new_tokens=256,
                use_cache=True,pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize();elapsed=time.time()-t0
            with output_path.open('a') as f:
                for b,r in enumerate(batch):
                    generated,hit=trim_generated(seq[b,encoded.input_ids.shape[1]:].tolist(),eos)
                    answer=tok.decode(generated,skip_special_tokens=True)
                    result={'configuration':config['name'],'depth':config['depth'],'id':r['id'],
                        'split':split,'domain':r['domain'],'answer':answer,'grade':judge(r,answer),
                        'generated_tokens':len(generated),'hit_token_limit':hit,
                        'batch_generation_seconds':elapsed,'batch_size':len(batch)}
                    f.write(json.dumps(result,ensure_ascii=False)+'\n');outputs.append(result)
                    done.add((config['name'],r['id']))
                f.flush()
            event('generation_evaluation',configuration=config['name'],split=split,
                  completed=min(offset+4,len(todo)),target=len(todo),total_answers=len(outputs))
        return True
    for config in configurations:
        if not run(config,'eval_test'):break

if __name__=='__main__':main()
