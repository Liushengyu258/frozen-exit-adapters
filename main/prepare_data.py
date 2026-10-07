"""Prepare a reproducible prompt-only corpus and disjoint task evaluation sets."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
SEED = 20261006

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def normalized(s):
    return ' '.join(re.sub(r'\d+(?:\.\d+)?',' NUM ',s.lower()).split())

def shingles(s):
    words = normalized(s).split()
    n = min(4,len(words))
    return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}

def extract_code(s):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```',s,re.S)
    return blocks[0].strip() if blocks else s.strip()

def write_jsonl(name, rows):
    (ROOT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))

def main():
    rng = random.Random(SEED)
    src = ROOT/'data'
    gsm = [json.loads(x) for x in (src/'gsm8k_train.jsonl').read_text().splitlines()]
    gsm_test = [json.loads(x) for x in (src/'gsm8k_test.jsonl').read_text().splitlines()]
    mbpp = [json.loads(x) for x in (src/'mbpp.jsonl').read_text().splitlines()]
    alpaca = json.loads((src/'codealpaca.json').read_text())
    math_suffix = '\nGive a concise solution in at most 4 short lines. End with: #### <numeric answer>.'
    code_suffix = '\nUse Python 3. Return only executable Python code, without explanation.'
    # MBPP remains exclusively an evaluation source.
    def mbpp_row(r,split):
        return {'id':f"mbpp_{r['task_id']}",'split':split,'domain':'code','source':'google-research/mbpp',
                'prompt':r['text']+'\nYour code must satisfy this example:\n'+r['test_list'][0]+code_suffix,
                'tests':r['test_list'],'test_setup_code':r.get('test_setup_code',''),
                'reference':r['code'],'task_id':r['task_id']}
    safe_imports={'math','re','itertools','collections','functools','heapq','bisect','statistics',
                  'operator','string','fractions','decimal','random','typing','sys','cmath','copy'}
    def supported(r):
        try: tree=ast.parse(r['code']+'\n'+r.get('test_setup_code',''))
        except SyntaxError: return False
        for n in ast.walk(tree):
            if isinstance(n,ast.Import) and any(a.name.split('.')[0] not in safe_imports for a in n.names): return False
            if isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] not in safe_imports: return False
            if isinstance(n,ast.Name) and n.id in {'open','exec','eval','input','__import__'}: return False
        return True
    mbval=[r for r in mbpp if 511<=r['task_id']<=600 and supported(r)]
    mbtest=[r for r in mbpp if 11<=r['task_id']<=510 and supported(r)]
    rng.shuffle(mbval); rng.shuffle(mbtest)
    rng.shuffle(gsm); rng.shuffle(gsm_test)
    evaluation=[]
    for split,mathrows,coderows,n in [('eval_validation',gsm[-40:],mbval,40),
                                     ('eval_test',gsm_test,mbtest,100)]:
        for r in mathrows[:n]:
            evaluation.append({'id':'eval_gsm_'+digest(r['question'])[:16], 'split':split,'domain':'math',
                'source':'openai/gsm8k_test' if split=='eval_test' else 'openai/gsm8k_train_reserved',
                'prompt':r['question']+math_suffix,'reference':r['answer'].split('####')[-1].strip()})
        evaluation.extend(mbpp_row(r,split) for r in coderows[:n])
    # Exclude near-duplicate prompts to any evaluation question, including unused MBPP rows.
    reserved=[r['question'] for r in gsm[-40:]]+[r['question'] for r in gsm_test]+[r['text'] for r in mbpp]
    reserved_norm={normalized(x) for x in reserved}
    reserved_shingles=[shingles(x) for x in reserved]
    inverted=defaultdict(set)
    for i,ss in enumerate(reserved_shingles):
        for s in ss: inverted[s].add(i)
    def contaminated(text):
        if normalized(text) in reserved_norm: return True
        ss=shingles(text); candidates=Counter(i for sh in ss for i in inverted.get(sh,()))
        return any(n/len(ss|reserved_shingles[i])>=0.5 for i,n in candidates.items())
    rows=[]
    for r in gsm[:-40]:
        if contaminated(r['question']): continue
        rows.append({'id':'gsm_'+digest(r['question'])[:16],'domain':'math','source':'openai/gsm8k_train',
                     'base_prompt':r['question'],'prompt':r['question']+math_suffix})
        if len(rows)==2250: break
    candidates=[]
    for idx,r in enumerate(alpaca):
        prompt=r['instruction']+ ('\n'+r['input'] if r.get('input') else '')
        if len(prompt)>2000 or contaminated(prompt): continue
        try: tree=ast.parse(extract_code(r['output']))
        except SyntaxError: continue
        if not any(isinstance(n,(ast.FunctionDef,ast.For,ast.While,ast.ListComp,ast.Assign,ast.If)) for n in ast.walk(tree)):continue
        # Do not ask for SQL/Java/C/etc and silently reinterpret as Python.
        if re.search(r'\b(java(?:script)?|typescript|c\+\+|c#|sql|html|css|ruby|php|bash|shell|golang|rust)\b',prompt,re.I):continue
        candidates.append({'id':f'alpaca_{idx}','domain':'code','source':'sahil280114/codealpaca',
                           'base_prompt':prompt,'prompt':prompt+code_suffix})
    rng.shuffle(candidates)
    seen=set()
    selected=[]
    for r in candidates:
        key=normalized(r['base_prompt'])
        if key in seen: continue
        seen.add(key);selected.append(r)
        if len(selected)==2250:break
    assert len(selected)==2250, len(selected)
    rows.extend(selected)
    # Control prompts target simple instruction-following; not general knowledge.
    for i in range(500):
        a,b=rng.sample(['amber','birch','coral','dawn','elm','fern','glow','hazel','iris','jade'],2)
        mode=i%10
        prompts=[f'Return this word in uppercase, with no explanation: {a}',
                 f'Return this word in lowercase, with no explanation: {a.upper()}',
                 f'Extract the value of color as plain text: {{"color":"{a}","name":"{b}"}}',
                 f'Return a JSON object with key "name" and value "{a}". No explanation.',
                 f'Copy exactly the text between brackets: [{a} {b}]',
                 f'请只返回以下列表的第一个词：{a}, {b}',
                 f'请把单词 {a} 转为大写，只输出结果。',
                 f'请只提取name字段的值：{{"name":"{a}","tag":"{b}"}}',
                 f'用英文逗号连接这两个词，不添加解释：{a} 和 {b}',
                 f'Return the second item, without explanation: {a}; {b}']
        text=prompts[mode]
        rows.append({'id':f'control_{i}','domain':'control','source':'procedural_instruction_controls',
                     'base_prompt':text,'prompt':text,'family':f'control_template_{mode}'})
    # Group normalized exact duplicates and high-overlap near duplicates before splitting.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    buckets=defaultdict(list); ss_all=[]; inv=defaultdict(set)
    for i,r in enumerate(rows):
        s=shingles(r['base_prompt']);ss_all.append(s)
        key=r.get('family',normalized(r['base_prompt']))
        if buckets[key]: parent[find(i)]=find(buckets[key][0])
        buckets[key].append(i)
        if r['domain']!='control':
            counts=Counter(j for sh in s for j in inv.get(sh,()))
            for j,n in counts.items():
                if rows[j]['domain']==r['domain'] and n/len(s|ss_all[j])>=0.8: parent[find(i)]=find(j)
            for sh in s: inv[sh].add(i)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    counts=defaultdict(Counter)
    shuffled=list(groups.values()); rng.shuffle(shuffled)
    for group in shuffled:
        d=group[0]['domain']; target={'math':2250,'code':2250,'control':500}[d]
        split=max(['train','validation','test'],key=lambda s:target*{'train':.8,'validation':.1,'test':.1}[s]-counts[d][s])
        family=digest('|'.join(sorted(r['id'] for r in group)))[:16]
        for r in group:
            r['split']=split;r['family']=family;r.pop('base_prompt',None)
        counts[d][split]+=len(group)
    # Interleave domains and splits deterministically, preserving breadth if interrupted.
    rng.shuffle(rows)
    assert len(rows)==5000,len(rows)
    write_jsonl('prompts.jsonl',rows)
    write_jsonl('evaluation.jsonl',evaluation)
    manifest={'seed':SEED,'prompt_count':len(rows),'counts':{k:dict(v) for k,v in counts.items()},
              'evaluation_counts':dict(Counter(r['split']+'_'+r['domain'] for r in evaluation)),
              'sources':{'gsm8k':'https://github.com/openai/grade-school-math',
                         'mbpp':'https://github.com/google-research/google-research/tree/master/mbpp',
                         'codealpaca':'https://github.com/sahil280114/codealpaca'},
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()},
              'notes':['45% math,45% code,10% procedural instruction controls',
                       'Predominantly English public corpus; not a balanced bilingual experiment',
                       'Math scope is GSM8K word problems, not broad competition mathematics',
                       'Code references used only for Python eligibility filtering, not probe labels',
                       'Training/validation/test grouped by normalized prompts and 4-gram Jaccard >= 0.8',
                       'Prompts overlapping evaluation at Jaccard >= 0.5 removed',
                       'No claim that pretrained Qwen has not seen public evaluation benchmarks',
                       'All 5000 teacher responses are generated by frozen Qwen; module labels are vectors only']}
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
