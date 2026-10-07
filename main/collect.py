"""Batched teacher generation, causal replay, all-block/embedding offline vectors."""
import json
import os
import shutil
import time
import numpy as np
import torch
from safetensors.torch import save_file
from common import *

def main():
    stop=deadline(9)
    rows=read_jsonl(ROOT/'prompts.jsonl');n=len(rows)
    records_path=RUN/'collection.jsonl'
    records=read_jsonl(records_path) if records_path.exists() else []
    done={r['index'] for r in records};shape=(n,POSITIONS,WIDTH)
    maps={i:np.memmap(ACT/f'layer_{i:02d}.bf16',dtype=np.uint16,
                      mode='r+' if (ACT/f'layer_{i:02d}.bf16').exists() else 'w+',shape=shape)
          for i in range(65)}
    atomic_json(ACT/'layout.json',{'shape':shape,'storage_dtype':'uint16 encoding bfloat16','layers':list(range(65)),
        'position_policy':'up to 8 prompt positions, rest answer positions, uniform inclusive endpoints',
        'boundary':'0=input embeddings;1..64=post-block residual before final RMSNorm'})
    model,tok=load_teacher();text=model.model.language_model
    save_file({'weight':text.norm.weight.detach().cpu().contiguous()},str(ACT/'norm.safetensors'))
    atomic_json(ACT/'norm.json',{'eps':text.norm.eps})
    assert not any(p.requires_grad for p in model.parameters())
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    def capture(ids,pos,use_cache=False,past=None):
        vectors={};handles=[]
        def hook(index):
            def fn(_m,_args,out):
                h=out[0] if isinstance(out,tuple) else out
                v=h[0,pos,:].detach().contiguous().cpu()
                if not torch.isfinite(v).all():raise RuntimeError(f'nonfinite layer {index}')
                vectors[index]=v
            return fn
        handles.append(text.embed_tokens.register_forward_hook(hook(0)))
        handles.extend(layer.register_forward_hook(hook(i)) for i,layer in enumerate(text.layers,1))
        try:
            with torch.no_grad():result=text(input_ids=ids,use_cache=use_cache,past_key_values=past)
        finally:
            for h in handles:h.remove()
        return vectors,result
    # Cache-vs-replay and independent shared-output-head checks before any main data.
    if not (RUN/'integrity.json').exists():
        ids=tok(render(tok,'Calculate 19 * 7. Explain briefly.'),return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            pref=text(input_ids=ids[:,:-1],use_cache=True)
            cached,_=capture(ids[:,-1:],torch.tensor([0],device='cuda'),True,pref.past_key_values)
            replay,out=capture(ids,torch.tensor([ids.shape[1]-1],device='cuda'))
            errors={str(i):((cached[i].float()-replay[i].float()).square().mean().sqrt()/
                            replay[i].float().square().mean().sqrt().clamp_min(1e-8)).item() for i in range(65)}
            actual=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            rebuilt=model.lm_head(text.norm(replay[64].cuda()))[0].float()
            logit_error=(actual-rebuilt).abs().max().item()
            single=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            bat=tok([render(tok,'Calculate 19 * 7. Explain briefly.'),render(tok,'Write a Python function that returns the sum of three numbers.')],padding=True,return_tensors='pt').to('cuda')
            padded=model(**bat,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            padding_rel=((single-padded).square().mean().sqrt()/single.square().mean().sqrt()).item()
        report={'frozen':True,'cache_vs_replay_relative_rms':errors,
                'shared_head_max_absolute_error':logit_error,'padding_logits_relative_rms':padding_rel,
                'padding_top1_match':bool(single.argmax()==padded.argmax())}
        assert max(errors.values())<0.08,report
        assert logit_error<0.1 and padding_rel<0.08,report
        report['passed']=True
        atomic_json(RUN/'integrity.json',report)
        del pref,out,cached,replay,actual,rebuilt,single,padded
        torch.cuda.empty_cache()
    todo=[i for i in range(n) if i not in done]
    start=time.time();pending=[];new_tokens=0
    for offset in range(0,len(todo),4):
        if time.time()>stop-120:break
        if shutil.disk_usage(ACT).free<8*2**30:raise RuntimeError('less than 8GiB disk headroom')
        indices=todo[offset:offset+4]
        prompts=[render(tok,rows[i]['prompt']) for i in indices]
        encoded=tok(prompts,padding=True,return_tensors='pt').to('cuda')
        assert encoded.input_ids.shape[1]<=1024,'unexpected oversized prompt'
        t0=time.time()
        with torch.no_grad():seq=model.generate(**encoded,do_sample=False,max_new_tokens=256,
            use_cache=True,pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize();generation_time=time.time()-t0
        for b,i in enumerate(indices):
            prefix=encoded.input_ids[b][encoded.attention_mask[b].bool()].tolist()
            generated,hit_limit=trim_generated(seq[b,encoded.input_ids.shape[1]:].tolist(),eos)
            all_ids=prefix+generated;plen=len(prefix);total=len(all_ids)
            nprompt=min(plen,8);nanswer=min(len(generated),POSITIONS-nprompt)
            pos=torch.unique(torch.cat([torch.linspace(0,plen-1,nprompt).long(),
                torch.linspace(plen,total-1,nanswer).long() if nanswer else torch.empty(0,dtype=torch.long)])).cuda()
            ids=torch.tensor([all_ids],device='cuda')
            vectors,result=capture(ids,pos)
            with torch.no_grad():
                reconstructed=text.norm(vectors[64].cuda()).float()
                original=result.last_hidden_state[0,pos].float()
                norm_error=(reconstructed-original).abs().max().item()
                norm_relative=((reconstructed-original).square().mean().sqrt()/original.square().mean().sqrt().clamp_min(1e-8)).item()
            assert norm_relative<0.005,(norm_error,norm_relative)
            for layer,v in vectors.items():maps[layer][i,:len(pos),:]=v.view(torch.uint16).numpy()
            pending.append({'index':i,'id':rows[i]['id'],'split':rows[i]['split'],'domain':rows[i]['domain'],
                'count':len(pos),'positions':pos.tolist(),'input_ids':all_ids,'prompt_tokens':plen,
                'generated_tokens':len(generated),'hit_token_limit':hit_limit,
                'answer':tok.decode(generated,skip_special_tokens=True),'batch_generation_seconds':generation_time,
                'norm_max_absolute_error':norm_error,'norm_relative_rms':norm_relative})
            new_tokens+=len(generated)
            del result,vectors
        if len(pending)>=24 or offset+4>=len(todo) or time.time()>stop-180:
            for m in maps.values():m.flush()
            with records_path.open('a') as f:
                for r in pending:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
            records.extend(pending);pending=[]
            event('collection',completed=len(records),target=n,elapsed_seconds=time.time()-start,
                  generated_tokens=new_tokens,aggregate_tokens_per_second=new_tokens/max(1,time.time()-start),
                  peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
                  free_disk_GiB=shutil.disk_usage(ACT).free/2**30)
    if pending:
        for m in maps.values():m.flush()
        with records_path.open('a') as f:
            for r in pending:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        records.extend(pending)
    atomic_json(RUN/'collection_summary.json',{'completed':len(records)==n,'count':len(records),'target':n,
        'generated_tokens':sum(r['generated_tokens'] for r in records),
        'positions':sum(r['count'] for r in records),'hit_token_limit':sum(r['hit_token_limit'] for r in records),
        'elapsed_seconds':time.time()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,
        'deadline_limited':len(records)<n})
    if len(records)<500:raise RuntimeError('insufficient corpus; collection did not reach 500 prompts')
    event('collection_finished',completed=len(records),target=n)

if __name__=='__main__':main()
