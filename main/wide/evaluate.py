"""Every target depth gets continuous-generation tests; no test-based selection."""
import gc,ast,re,collections
from core import *
from common import load_teacher,render,trim_generated
from judge import judge,self_test

def diagnostics(answer,domain):
    tokens=answer.split();grams=[tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
    repetition=1-len(set(grams))/len(grams) if grams else 0
    result={'replacement_characters':answer.count('\ufffd'),'control_characters':sum(ord(c)<32 and c not in '\n\r\t' for c in answer),'repeated_4gram_fraction':repetition,'suspected_repetitive':len(tokens)>32 and repetition>.5,'empty':not answer.strip()}
    if domain=='code':
        blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S);code='\n'.join(blocks) if blocks else answer.strip()
        try:ast.parse(code);result['syntax_valid']=True
        except SyntaxError as exc:result['syntax_valid']=False;result['syntax_error']=str(exc)
    result['suspected_encoding_problem']=result['replacement_characters']>0 or result['control_characters']>0
    return result

def main():
    stop=json.loads((RUN/'budget.json').read_text())['generation_deadline_epoch'];self_test()
    support=json.loads((MAIN/'run/supported_evaluation.json').read_text())
    evaluation=[r for r in read_jsonl(MAIN/'evaluation.jsonl') if r['split']=='eval_test' and r['id'] in support['included']]
    assert len(evaluation)==199
    configs=[]
    # Chosen best across all four depths first, then explicit requested architecture and baseline.
    names=set()
    for label in ['best','requested']:
        for depth in [60,61,62,63]:
            meta=RUN/f'{label}_s{depth}.json'
            if not meta.exists():continue
            ck=json.loads(meta.read_text());key=(depth,ck['name'])
            if key in names:continue
            names.add(key);configs.append({'name':f'{label}_s{depth}','depth':depth,'probe':str(RUN/ck['path']),'training_name':ck['name'],'config':ck['config']})
    for d in [61,62]:configs.append({'name':f'identity_{d}','depth':d,'probe':None})
    atomic_json(RUN/'generation_plan.json',{'configurations':configs,'questions':len(evaluation),'max_new_tokens':256,'selection':'vector validation MSE only','test_previously_used':True,'reference_exclusions':support['excluded']})
    output=RUN/'generation_results.jsonl';outputs=read_jsonl(output) if output.exists() else [];done={(r['configuration'],r['id']) for r in outputs}
    model,tok=load_teacher();text=model.model.language_model;original_norm=text.norm;types=list(text.config.layer_types)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    for config in configs:
        todo=[r for r in evaluation if (config['name'],r['id']) not in done]
        if not todo:continue
        if time.time()>stop-90:break
        text.norm=original_norm;gc.collect();torch.cuda.empty_cache()
        for cc in [text.config,model.config.text_config]:cc.num_hidden_layers=config['depth'];cc.layer_types=types[:config['depth']]
        if config['probe']:
            adapter,ck=load_probe(config['probe']);text.norm=ExitNorm(adapter,original_norm);del adapter,ck
        seen=set();hooks=[]
        for i,b in enumerate(text.layers,1):
            def hook(_m,_a,_o,index=i):seen.add(index)
            hooks.append(b.register_forward_hook(hook))
        encoded=tok(render(tok,'Return only the integer 2.'),return_tensors='pt').to('cuda')
        with torch.no_grad():model.generate(**encoded,do_sample=False,max_new_tokens=2,use_cache=True,pad_token_id=tok.pad_token_id)
        for h in hooks:h.remove()
        assert seen==set(range(1,config['depth']+1)),seen
        for start in range(0,len(todo),4):
            if time.time()>stop-45:break
            batch=todo[start:start+4];encoded=tok([render(tok,r['prompt']) for r in batch],padding=True,return_tensors='pt').to('cuda');t=time.time()
            with torch.no_grad():seq=model.generate(**encoded,do_sample=False,max_new_tokens=256,use_cache=True,pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize();elapsed=time.time()-t
            for i,r in enumerate(batch):
                ids,hit=trim_generated(seq[i,encoded.input_ids.shape[1]:].tolist(),eos);answer=tok.decode(ids,skip_special_tokens=True)
                row={**r,'configuration':config['name'],'depth':config['depth'],'training_name':config.get('training_name'),'answer':answer,'grade':judge(r,answer),'generated_tokens':len(ids),'hit_token_limit':hit,'diagnostics':diagnostics(answer,r['domain']),'batch_seconds':elapsed}
                with output.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                outputs.append(row);done.add((config['name'],r['id']))
            event('wide_generation',configuration=config['name'],completed=sum(c==config['name'] for c,i in done),target=len(evaluation),total_answers=len(outputs))
    counts={c['name']:sum(r['configuration']==c['name'] for r in outputs) for c in configs}
    atomic_json(RUN/'generation_summary.json',{'completed':all(v==len(evaluation) for v in counts.values()) and len(configs)>=6,'counts':counts,'target_per_configuration':len(evaluation),'planned_configurations':len(configs),'total_answers':len(outputs)})
    event('wide_generation_finished',counts=counts)
if __name__=='__main__':main()
