import json
import os
from pathlib import Path
import time
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
RUN=ROOT/'run'
ACT=ROOT/'activations'
for p in [RUN,ACT,RUN/'probes']:p.mkdir(parents=True,exist_ok=True)
SEED=20261006
WIDTH=5120
POSITIONS=32

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def atomic_json(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(temp,path)

def event(stage,**kw):
    row={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
    atomic_json(RUN/'status.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)

def deadline(default_hours):
    return float(os.environ.get('STAGE_DEADLINE',time.time()+default_hours*3600))

def load_teacher():
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    path=os.environ.get('MODEL_PATH',str(BASE/'models/Qwen3.8-27B'))
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    torch.set_num_threads(8)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    return model,tokenizer

def render(tokenizer,prompt):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)

def trim_generated(ids,eos):
    for i,x in enumerate(ids):
        if x in eos:return ids[:i+1],False
    return ids,True
