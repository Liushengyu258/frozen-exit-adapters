import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

def numeric(text):
    marked=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)',text)
    boxed=re.findall(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}',text)
    values=marked or boxed or re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?',text)
    if not values:return None
    try:return Fraction(values[-1].replace(',',''))
    except (ValueError,ZeroDivisionError):return None

def judge(row,answer):
    if row['domain']=='math':
        predicted=numeric(answer);target=numeric(row['reference'])
        return {'passed':predicted is not None and predicted==target,'reason':'numeric_exact',
                'prediction':str(predicted),'reference':str(target)}
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S)
    code='\n'.join(blocks) if blocks else answer.strip()
    payload={'code':code,'setup':row.get('test_setup_code',''),'tests':row['tests']}
    with tempfile.TemporaryDirectory(prefix='layer-probe-eval-') as tmp:
        os.chmod(tmp,0o777)
        try:
            p=subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('code_worker.py'))],
                input=json.dumps(payload),text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                timeout=6,cwd=tmp,env={'PATH':'/usr/bin:/bin','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1'})
            if p.returncode!=0:return {'passed':False,'reason':'worker_exit','returncode':p.returncode}
            try:result=json.loads(p.stdout)
            except (ValueError,TypeError):return {'passed':False,'reason':'invalid_worker_output'}
            if result.get('reason')=='sandbox_unavailable':raise RuntimeError('code sandbox unavailable; refusing unsandboxed evaluation')
            return result
        except subprocess.TimeoutExpired:return {'passed':False,'reason':'timeout'}

def self_test():
    row={'domain':'code','tests':['assert plus(2,3)==5'],'test_setup_code':''}
    assert judge(row,'def plus(a,b): return a+b')['passed']
    assert not judge(row,'def plus(a,b): return a-b')['passed']
    assert not judge(row,'import os\ndef plus(a,b): return a+b')['passed']
    assert not judge(row,'def plus(a,b):\n while True: pass')['passed']
    assert numeric('answer #### 1,234')==1234
    print('JUDGE_SELF_TEST_PASSED',flush=True)

if __name__=='__main__':self_test()
