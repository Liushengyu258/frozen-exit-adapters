"""Create local runtime budgets explicitly; never provision or stop compute."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training-hours',type=float,required=True);p.add_argument('--generation-hours',type=float,required=True);p.add_argument('--analysis-hours',type=float,default=.5);a=p.parse_args()
assert min(a.training_hours,a.generation_hours,a.analysis_hours)>0
r=Path(__file__).resolve().parent/'run';r.mkdir(exist_ok=True);f=r/'budget.json'
if f.exists():raise SystemExit('budget.json exists: refusing to overwrite an existing run')
t=time.time();b={'started_epoch':t,'training_deadline_epoch':t+a.training_hours*3600,'generation_deadline_epoch':t+(a.training_hours+a.generation_hours)*3600,'analysis_deadline_epoch':t+(a.training_hours+a.generation_hours+a.analysis_hours)*3600}
f.write_text(json.dumps(b,indent=2));print(f)
