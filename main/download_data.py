"""Fetch public upstream files and reject any change from the archived bytes."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URLS={
 'gsm8k_train.jsonl':'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl',
 'gsm8k_test.jsonl':'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl',
 'mbpp.jsonl':'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl',
 'codealpaca.json':'https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json'}
def main():
 expected=json.loads((ROOT/'data_manifest.json').read_text())['source_sha256'];out=ROOT/'data';out.mkdir(exist_ok=True)
 for name,url in URLS.items():
  content=urllib.request.urlopen(url,timeout=60).read()
  if hashlib.sha256(content).hexdigest()!=expected[name]:raise RuntimeError(f'Upstream bytes changed: {name}; recover the original source revision before continuing')
  (out/name).write_bytes(content);print(name,'verified')
if __name__=='__main__':main()
