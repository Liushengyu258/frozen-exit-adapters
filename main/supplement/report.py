import collections
from core import *
def main():
    planned=json.loads((ROOT/'plan.json').read_text());training=read_jsonl(RUN/'probe_results.jsonl') if (RUN/'probe_results.jsonl').exists() else []
    rows=read_jsonl(RUN/'generation_results.jsonl') if (RUN/'generation_results.jsonl').exists() else []
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['configuration']].append(r)
    summary={}
    for name,items in grouped.items():
        summary[name]={'count':len(items),'math_correct':sum(bool(r['grade'].get('passed')) for r in items if r['domain']=='math'),'code_correct':sum(bool(r['grade'].get('passed')) for r in items if r['domain']=='code'),'math_n':sum(r['domain']=='math' for r in items),'code_n':sum(r['domain']=='code' for r in items),'hit_limit':sum(r['hit_token_limit'] for r in items),'suspected_encoding':sum(r['diagnostics']['suspected_encoding_problem'] for r in items)}
    traj=json.loads((RUN/'trajectory_summary.json').read_text()) if (RUN/'trajectory_summary.json').exists() else {'completed':False}
    complete=len(training)==24 and len(summary)==43 and all(v['count']==199 for v in summary.values()) and traj['completed']
    atomic_json(RUN/'main_summary.json',{'completed':complete,'training_completed':len(training),'training_planned':24,'generation_complete_configurations':sum(v['count']==199 for v in summary.values()),'generation_planned':43,'generation':summary,'trajectory':traj})
    lines=['# 第三轮补充实验','',f'训练完成 {len(training)}/24；生成完成 {sum(v["count"]==199 for v in summary.values())}/43；轨迹完成：{traj["completed"]}。','', '每组固定12000步，无patience早停。生成测试复用既有199题，不是新的独立盲测。固定步数不等于等算力或充分收敛。旧16候选使用归档BF16 screen检查点，区别于上轮FP32最终候选。','', '| 配置 | 数学 | 代码 | 到达长度上限 | 疑似编码异常 |','|---|---:|---:|---:|---:|']
    for name,v in summary.items():lines.append(f'| {name} | {v["math_correct"]}/{v["math_n"]} | {v["code_correct"]}/{v["code_n"]} | {v["hit_limit"]} | {v["suspected_encoding"]} |')
    lines+=['','## 训练末段趋势','', '| 配置 | 最佳步数 | 最佳验证MSE | 最后验证MSE |','|---|---:|---:|---:|']
    for r in training:lines.append(f'| {r["name"]} | {r["best_step"]} | {r["validation"]["all"]:.6f} | {r["curve"][-1]["validation_mse"]:.6f} |')
    lines+=['','完整逐题回答、判分与token IDs见generation_results.jsonl；轨迹逐token标量见trajectory_results.jsonl。报告为自动汇总，最终结论需要核对原始结果与同题对照。']
    (RUN/'REPORT.md').write_text('\n'.join(lines));event('report_written',completed=complete)
if __name__=='__main__':main()
