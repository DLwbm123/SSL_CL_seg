"""Collect all three fixed seeds without selecting checkpoint or changing gates."""
import json,sys
from pathlib import Path

def run(root,dest):
    results={str(seed):json.loads((root/f'P0_s{seed}'/'METRICS.json').read_text()) for seed in (163,164,165)}
    for seed,r in results.items():
        assert r['status']=='COMPLETE' and r['patients']==4 and r['optimizer_updates']==0
        assert json.loads((root/f'P0_s{seed}'/'EXIT.json').read_text())['exit_code']==0
    report={'gate':'PASS' if all(r['gate']=='PASS' for r in results.values()) else 'FAIL','seeds':results,'optimizer_updates':0,'forward_calls':sum(r['forward_calls'] for r in results.values()),'patient_count':4,'independent_patients_not_seed_multiples':True}
    (dest/'PROSTATE_FROZEN_TEACHER_P0_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# P0 冻结旧教师正确性诊断结果','',f"总 gate：**{report['gate']}**。3 seed，各相同4名当前域训练标注患者，全部切片，零优化器更新。此集已被 EMA 训练见过，仅为有偏的可行性诊断。",'', '## 早期 S1000 放行指标','', '| seed | view | old FG recall | old FG coverage | FG complement | mix−EMA FG NLL | BG NLL | FG confwrong | BG confwrong |','|---|---|---|---|---|---|---|---|---|']
    for seed,r in results.items():
        for name in ('current','history'):
            fg=r['means'][f'{name}/1000/fg'];bg=r['means'][f'{name}/1000/bg'];cov=r['means'][f'{name}/1000/coverage']['old']
            vals=[1-fg['old_error'],cov,fg['old_correct_ema_wrong'],fg['mix_nll']-fg['ema_nll'],bg['mix_nll']-bg['ema_nll'],fg['mix_confident_wrong']-fg['ema_confident_wrong'],bg['mix_confident_wrong']-bg['ema_confident_wrong']]
            lines.append('| '+seed+' | '+name+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 固定门槛判定','']
    for seed,r in results.items():lines.append(f"- {seed}: {r['gate']}; "+(', '.join(r['failures']) or '全部通过')+f"; {r['slices']} slices, {r['forward_calls']} forward calls.")
    lines+=['','## 解释与边界','', '本轮三个 seed 均具备基本前景覆盖，但固定混合在前景 NLL 上超过风险门槛，history 早期前景互补也不足。不能用背景指标的小幅改善抵消前景风险。P0 否决本轮固定混合的训练资格，四臂未启动。', '', '这是当前训练标注患者上的诊断；EMA 已见过这些标签，故不能推断真实 U 上绝无旧教师互补，也不能证明旧域保持无效。下一问题应首先解决资格诊断的训练暴露偏差，不能以调小 rho、换 checkpoint 或追加 seed 绕过本 gate。任何独立验证设计需先冻结数据隔离和匹配对照，保持真实 U 标签、val/test/HK 禁止用于诊断选择的边界。', '', '完整逐患者匿名数值（NLL/Brier/错误/高置信错误/覆盖率、互补及反向互补），含 original/current/history × S1000/S3000/S5000，见同目录 JSON。患者是四人，seed不能作为新增患者。', '', 'PASS才放行固定四臂；FAIL不启动、不改变阈值或换checkpoint。混合标签质量不是训练收益或旧域保持的证明。']
    (dest/'PROSTATE_FROZEN_TEACHER_P0_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
