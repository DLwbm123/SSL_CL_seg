"""Audit local inference ledgers and write anonymous results/report."""
import json,sys
from pathlib import Path


def main(raw,out):
    p,out=Path(raw),Path(out);r=json.loads((p/'RESULTS.json').read_text());c=r['costs']
    assert r['status']=='COMPLETE' and json.loads((p/'EXIT.json').read_text())['exit_code']==0 and not (p/'FAILURE.json').exists()
    assert c==dict(model_optimizer=0,model_parameter_updates=0,local_reverse=50,local_updates=50,candidates=119,elbo_evaluations=179,predictive_batches=4)
    for n,num in [('reverse_calls',50),('local_updates',50),('candidate_evaluations',119),('candidate_outcomes',119)]:
        v=[json.loads(x) for x in (p/(n+'.jsonl')).read_text().splitlines()];assert [x['index'] for x in v]==list(range(1,num+1))
    ref=json.loads((out/'results/INFERENCE.json').read_text());rows=['# 无标签ELBO验收局部推断报告','','局部ELBO真实改善，但未挽救图像条件语义预测。终态后验25次尝试全部接受，验收流与另一读出流每图ELBO均改善，保留图Dice仍约11.8%；原始先验对照也有类似改善。不能把失败全部归结为细化位移太小或步长未验收。','','|生成分布|SAA平均ELBO初→末|另一流ELBO初→末|均值训练Dice初→末%|均值保留Dice初→末%|SNIS保留Dice初→末%|','|---|---|---|---|---|---|']
    mean=lambda z:sum(z)/len(z)
    d=lambda g,st,mode,part:100*sum(map(sum,g[st+'_metrics'][mode][part]['dice']))/6
    for post,g in r['groups'].items():
        assert g['model_unchanged'] and len(g['trace'])==25
        for i,t in enumerate(g['trace']):
            assert t['step']==i+1 and t['alpha']>0 and all(a>=b-1e-6 for a,b in zip(t['elbo_after'],t['elbo_before'])) and mean(t['elbo_after'])>mean(t['elbo_before'])+1e-5
            if i:assert t['elbo_before']==g['trace'][i-1]['elbo_after']
        assert g['initial_elbo']==g['trace'][0]['elbo_before'] and g['final_elbo']==g['trace'][-1]['elbo_after']
        for channel in ['mean','SNIS']:
            for part,a in g['initial_metrics'][channel].items():
                b=ref['groups']['terminal__'+post][channel]['metrics'][part]
                assert a['dice']==b['dice'] and a['class2_confusion']==b['class2_confusion']
                assert all(abs(a[k]-b[k])<1e-6 for k in ['ce','class2_ce'])
        assert all(a>b for a,b in zip(g['cross_final_elbo'],g['cross_initial_elbo']))
        rows.append(f"|{post}|{mean(g['initial_elbo']):.6f}→{mean(g['final_elbo']):.6f}|{mean(g['cross_initial_elbo']):.6f}→{mean(g['cross_final_elbo']):.6f}|{d(g,'initial','mean','train'):.5f}→{d(g,'final','mean','train'):.5f}|{d(g,'initial','mean','heldout'):.5f}→{d(g,'final','mean','heldout'):.5f}|{d(g,'initial','SNIS','heldout'):.5f}→{d(g,'final','SNIS','heldout'):.5f}|")
    rows+=['','终态后验末SNIS ESS逐图约[1,1,1,1.00002]，仍退化；原先验最后一图约1.50，其余1。不能把SNIS看成稳定后验平均。ELBO改进不等于语义提高：后验训练图均值Dice97.81751%→97.77611%，SNIS97.77864%→97.61889%。保留图改善很小，原先验并不更差，未证明学到的生成参数有独立纠错收益。','','实际0模型optimizer/0模型参数更新；50局部反向、50接受状态步、119候选、179批量ELBO评价（每批8样本）、4预测批次。reverse/local_updates/candidate及对应结果四账本连续，每图门限、均值严格提升、25步链连续通过；初始Dice/混淆完全复现、CE误差<1e-6（CPU/GPU计分归约差），模型完整张量不变，退出0。历史失败和源状态保留，窗口空间<1GB/20GB。','','边界：优化的是有限样本变分ELBO；另一既有流每图也改善，降低了仅拟合验收噪声的疑虑，但不等于全局后验已求解。25步不证明收敛；没有证据支持盲目延长。下一项用物理(x,m)与语义(ell,c)局部shift零更新四格交叉，分解ELBO改进的来源及交互，再判断是否只优化语义坐标。标签仅计分。','','完整轨迹和各图结果见[LOCAL_GATE.json](results/LOCAL_GATE.json)。']
    r['audit']=dict(status='PASS',ledgers_contiguous=True,all_50_steps_accepted=True,all_step_gates_pass=True,baseline_dice_confusion_exact=True,baseline_CE_tolerance=1e-6,all_models_unchanged=True,exit_code=0)
    (out/'results/LOCAL_GATE.json').write_text(json.dumps(r,indent=2)+'\n');(out/'LOCAL_GATE_REPORT_ZH.md').write_text('\n'.join(rows)+'\n');print(json.dumps(r['audit']))


if __name__=='__main__':main(*sys.argv[1:])
