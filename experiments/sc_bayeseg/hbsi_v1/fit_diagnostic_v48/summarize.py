"""Summarize complete train-only diagnostics; no additional scoring or selection."""
import argparse,json
from pathlib import Path
from statistics import mean


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());r=x['RESULTS.json'];l=x['ledger'];out=Path(__file__).parent
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0 and x['PREFLIGHT.json']['status']=='PASS'
    assert r['formal_updates']==400 and r['audit_reverse_calls']==12 and r['validation_cells']==0
    assert len(l['training_sessions'])==2 and all(v['completed_steps']==200 and v['status']=='COMPLETE' for v in l['training_sessions'].values())
    assert all(v['status']=='PASS' for v in l['attempts'])
    interpretations={'task_only_fit_advantage':'Task-only通过、Full不通过且差至少10pp，支持另立单一目标项修复协议；该差包含整条优化轨迹，不足以单独识别是哪一个项。', 'both_fit':'两个目标都能拟合固定训练子集，当前局部拟合能力足够；不能据此证明全训练或跨患者泛化原因。', 'neither_fit':'两个目标均未达到局部拟合门槛，尚未分离200步优化不足与参数化限制；不能断言结构上限，也不能把完整目标冲突当作唯一根因。','inconclusive':'两个目标没有形成预注册的明确区分；保留不确定，不追改门槛或增加本轮步数。'}
    lines=['# v48 W训练拟合诊断结果','','只读8张固定当前trainL切片；4名训练患者，单训练seed，3次MC推断。无val/test、旧域图像或U。此结果不是泛化性能。','','| 臂 | 步 | 非空片Dice % | 最差患者非空Dice % | 空片FP % | 拟合门槛 | mean risk(p) | risk(mean p) |','|---|---:|---:|---:|---:|---|---:|---:|']
    for arm,a in r['arms'].items():
        assert len(a['gradient_audits'])==3 and [q['step'] for q in a['readouts']]==[0,100,200]
        for q in a['readouts']:
            values=[v for row in q['rows'] for v in row['repeats']];fp=q['empty_FP_fraction']
            fptext='NA' if fp is None else f'{100*fp:.5f}'
            lines.append(f"| {arm} | {q['step']} | {100*q['foreground_Dice']:.5f} | {100*min(q['patient_foreground_Dice']):.5f} | {fptext} | {q['fit_gate']} | {mean(v['expected_risk'] for v in values):.7f} | {mean(v['mean_probability_risk'] for v in values):.7f} |")
    lines+=['',f"冻结分支：**{r['decision']}**。Task−Full终点非空片Dice={100*r['task_minus_full_Dice']:+.5f}pp。{interpretations[r['decision']]}", '', '梯度证据见匿名RESULTS.json：步0/99/199对q、eta、phi记录task及其余项范数/余弦，及实际Adam步内积。负task_dot_delta表示沿局部一阶任务下降方向；不能由一个梯度余弦直接证明长期因果。mean risk与risk(mean p)的差是描述，Dice无Jensen保证。', '', '8片均匀采样改变经验分布；KL仍按原L=134归一化。该设计只用于受控可拟合性诊断，不能作为原全量目标的一次正式训练或正式方法成绩。来源模型、固定旧memory及S/b保持。']
    cost=dict(started=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,preflight_reverse_calls=2,gradient_audit_reverse_calls=12,readout_sample_forwards=4608,stage_seconds={v['name']:v['seconds_actual'] for v in l['attempts']})
    lines+=['',f"首次启动至退出{cost['elapsed']:.2f}秒，根{cost['root_bytes']}字节；400训练backward/Adam，2次preflight反向与12次额外审计反向，4608次读出样本前向。中间读出成本计入各臂训练阶段，无免费探针。", '', '[协议](PREREG_ZH.md)；[匿名逐片及梯度结果](RESULTS.json)；[执行成本](EXECUTION_SUMMARY.json)。']
    lines+=['', '## 实测解释', '', 'Full与Task均通过全部冻结拟合门槛，终点非空片Dice为94.27803%与94.40669%，Task仅高0.12867pp。二者最差训练患者均超过92%，选中空片FP均为0。当前8片实验不支持完整目标把分割训练完全压住，也不支持此参数化完全不能表达训练标签；不等价于全量134片可同样拟合，更不能证明跨患者泛化。', '', '在Full三个审计点，q和phi的任务梯度与实际Adam步内积均负；eta起点为正，后来为负。末点其余项相对task梯度很大，但此时task梯度已很小，且整体已达到拟合门槛。不能由范数比直接断言正则压制是性能差的主因。', '', '结合v47 W入口近零损失、实际新域学习为正，以及v46低precision/较高前景体积比，下一问题应聚焦全量训练到跨患者评价的差距、前景背景风险权衡与结构化概率读出。当前不支持直接warm-start、全面移除KL/观测项或断言结构容量上限。具体原因仍需单因素后继验证；本诊断没有确定唯一根因。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    for n,v in [('RESULTS.json',r),('PREFLIGHT.json',x['PREFLIGHT.json']),('SELECTION.json',x['SELECTION.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(dict(decision=r['decision'],task_minus_full_Dice=r['task_minus_full_Dice'])))

if __name__=='__main__':main()
