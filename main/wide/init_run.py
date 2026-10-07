"""Initialize a NEW run with archived active-time quotas and explicit deadlines.
Does not start a task, rent hardware, or control server power.
"""
import argparse,json,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training-hours',type=float,required=True);p.add_argument('--generation-hours',type=float,required=True);a=p.parse_args()
if min(a.training_hours,a.generation_hours)<=0:p.error('Hours must be positive')
r=Path(__file__).resolve().parent;run=r/'run';run.mkdir(exist_ok=True)
if any(run.iterdir()):raise SystemExit('Run directory must be empty; preserve previous results first.')
shutil.copyfile(r/'results/training_plan.json',run/'training_plan.json')
t=time.time();(run/'budget.json').write_text(json.dumps({'training_deadline_epoch':t+3600*a.training_hours,'generation_deadline_epoch':t+3600*(a.training_hours+a.generation_hours)},indent=2))
print('Initialized; no GPU job started.')
