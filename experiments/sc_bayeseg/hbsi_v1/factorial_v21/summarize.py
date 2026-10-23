"""Complete four-cell offline report, retaining the frozen Q1 report separately."""
import argparse
import json
from pathlib import Path
from ..trainability_v21.summarize import analyze, report


def main():
    p=argparse.ArgumentParser();p.add_argument('q1_snapshot');p.add_argument('f01_snapshot');args=p.parse_args()
    q1=json.loads(Path(args.q1_snapshot).read_text());f01=json.loads(Path(args.f01_snapshot).read_text())
    assert f01['RESULTS.json']['formal_updates']==800 and f01['RESULTS.json']['discarded_copy_updates']==24
    assert f01['ledger']['updates_reserved']==824 and all(v['status']=='PASS' for v in f01['ledger']['attempts'] if v['name']!='matching')
    c=f01['RESULTS.json']['counter'];assert c['global_updates']==0 and c['phi_updates']==800 and c['domain_updates']==800
    a=analyze(q1,f01);out=Path(__file__).parent
    (out/'ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
    (out/'MATCHING.json').write_text(json.dumps(f01['MATCHING.json'],indent=2)+'\n')
    execution=dict(status='COMPLETE',exit=f01['EXIT.json'],original_launch=f01['LAUNCH.json'],wall_elapsed_seconds=f01['EXIT.json']['time']-f01['LAUNCH.json']['started'],ledger={k:f01['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')},formal_updates=800,discarded_copy_Adam_updates=24,diagnostic_reverse_calls=24,preflight_reverse_calls=1,preflight_Adam_updates=0,source_updates=0,formal_counter=c,failed_attempts=[{k:v for k,v in entry.items() if k in ('name','status','seconds_actual','updates_reserved','error_type')} for entry in f01['ledger']['attempts'] if entry['status']=='FAILED'],failure_before_any_Adam_or_backward=True,repair='noncomputational runtime cleared before common payload comparison; original reservation and clock retained')
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    text=report(a,q1,f01).replace('[原Q1冻结协议](PREREG_ZH.md)','[原Q1冻结协议](../trainability_v21/PREREG_ZH.md)')
    text+='\n## F01独立协议的收口\n\n'
    for n in ('phi_given_q_fixed','phi_given_q_learning','q_given_phi_fixed','q_given_phi_learning','interaction'):
        e=a['effects'][n];text+=f"{n}：旧域{e['RUNMC']['metrics']['Dice']*100:+.4f}pp，新域{e['BMC']['metrics']['Dice']*100:+.4f}pp。\n\n"
    passing=[n for n,g in a['gates'].items() if g['status']=='PASS']
    text+=('晋级候选：'+','.join(passing)+'；仅获得另立确认协议资格。' if passing else '四臂均未通过原晋级门槛；本次补格没有产生可晋级的目标L适应配置，不追加seed、权重或增强搜索。')
    text+='\n\n交互按原始差分定义报告，源/目标两域须分别解释；不把某一域正交互当成双域净收益。完整四臂只是补齐条件对照，不是四个独立训练seed。\n\n'
    text+='匹配审计通过：原v19实现与新快照各10步副本逐步loss、完整payload、Adam和RNG相同；四策略各1步冻结检查，共24次丢弃副本更新。正式只新增F01的800步，F11/F10/F00和源锚点没有重训或重评分。\n\n'
    text+=f"首次匹配在比较非计算runtime元数据时退出，发生于反向/Adam之前；失败记录保留，24次原审计预留未清零，修复阶段沿用该预留（新增预留0）。原时钟包含修复与等待，总历时{execution['wall_elapsed_seconds']:.2f}秒；不把副本审计或预检成本记成0。\n\n"
    text+='[F01独立冻结协议](PREREG_ZH.md)；[匹配证据](MATCHING.json)；[执行摘要](EXECUTION_SUMMARY.json)；[Q1原始收口报告](../trainability_v21/REPORT_ZH.md)。后续研究优先分析目标更新与后验/识别器/域记忆兼容性，任何新训练必须另立可证伪假设和预算，不从当前开发分数降低门槛。\n'
    (out/'REPORT_ZH.md').write_text(text)
    print(json.dumps(dict(gates=a['gates'],effects={n:{d:v['metrics']['Dice']*100 for d,v in ds.items()} for n,ds in a['effects'].items() if n in ('phi_given_q_fixed','phi_given_q_learning','q_given_phi_fixed','q_given_phi_learning','interaction')})))

if __name__=='__main__':main()
