"""One terminal, saved-rows-only audit. Never imports or evaluates a network."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path): return json.loads(Path(path).read_text())


def scores(path, site):
    groups = defaultdict(list)
    for r in read(Path(path)/'PRIVATE_ROWS.json'):
        if r['site'] == site: groups[r['key']].append(r['dice'])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def comparison(a, b):
    assert a.keys() == b.keys()
    d = np.array([b[k]-a[k] for k in sorted(a)])*100
    rng = np.random.default_rng(163)
    means = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
    return dict(mean_pp=float(d.mean()), CI95_pp=np.quantile(means, [.025, .975]).tolist(),
                over1=int((d < -1).sum()), over2=int((d < -2).sum()), p10_pp=float(np.quantile(d, .1)), n=len(d))


def main(cfg):
    root = Path(cfg['root']); assert not (root/'AUDIT.json').exists(), 'Audit is once only'
    launch = read(root/'LAUNCH.private.json')
    members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            # Entry path and argv are neutral; inspect process group, never kill it.
            text = (p/'stat').read_text(); rest = text[text.rindex(')')+2:].split()
            if int(rest[2]) == launch['process_group'] and rest[0] != 'Z': members.append(int(p.name))
        except (FileNotFoundError, PermissionError, ValueError): pass
    assert not members, members
    terminal = read(root/'TERMINAL.json')
    physical = sum(1 for _ in (root/'physical_optimizer.jsonl').open())
    assert physical == terminal['counts']['optimizer'] <= 80096
    summary = dict(terminal=terminal, audit_model_forward=0, optimizer_ledger=physical,
                   single_seed=20272401, independent_test=False, current_U_used=True,
                   historical_raw_images_in_adaptation=False, datasets={})
    complete = {}
    for dataset, sites in cfg['sites'].items():
        d = dict(source={}, stages={}, paper_reference_mean=87.81 if dataset == 'mnms' else 83.10,
                 direct_paper_comparison=False, comparability_reason='different frozen development-validation split' if dataset == 'mnms' else 'three sites; second REFUGE camera absent; development validation')
        source = root/f'{dataset}_0_SOURCE'
        if (source/'SOURCE_GATE.json').exists(): d['source'] = dict(gate=read(source/'SOURCE_GATE.json'), metrics=read(source/'METRICS.json'))
        for depth in range(1, len(sites)):
            for arm in ('B', 'C'):
                p = root/f'{dataset}_{depth}_{arm}'
                if (p/'METRICS.json').exists() and (p/'EXIT.json').exists() and read(p/'EXIT.json')['exit_code'] == 0:
                    assert (p/'QUALIFICATION.json').exists() and read(p/'QUALIFICATION.json')['status'] == 'PASS'
                    d['stages'][p.name] = read(p/'METRICS.json')
        finals = {a: root/f'{dataset}_{len(sites)-1}_{a}' for a in ('B', 'C')}
        done = all((p/'EXIT.json').exists() and read(p/'EXIT.json')['exit_code'] == 0 for p in finals.values())
        complete[dataset] = done
        if done:
            d['final'] = {a: read(p/'METRICS.json') for a, p in finals.items()}
            d['final_mean'] = {a: float(np.mean([m['Dice_percent'] for m in v.values()])) for a, v in d['final'].items()}
            d['C_minus_B'] = {s: comparison(scores(finals['B'], s), scores(finals['C'], s)) for s in sites}
            d['tails_C_relative_arrival'] = {}
            for i, s in enumerate(sites[:-1]):
                arrival = source if i == 0 else root/f'{dataset}_{i}_C'
                d['tails_C_relative_arrival'][s] = comparison(scores(arrival, s), scores(finals['C'], s))
            d['source_warm_vs_final_C'] = comparison(scores(source/'warm_reference', sites[0]), scores(finals['C'], sites[0]))
            d['numerically_within_2pp'] = d['final_mean']['C'] >= d['paper_reference_mean']-2
            tails = d['tails_C_relative_arrival'].values()
            d['old_tail_pass'] = all(x['over1'] <= 2 and x['over2'] == 0 and x['p10_pp'] >= -.5 and x['CI95_pp'][0] >= -.3 for x in tails)
            ni = d['C_minus_B'][sites[-1]]
            d['final_new_noninferiority_pass'] = ni['mean_pp'] >= -.2 and ni['CI95_pp'][0] >= -.2
            d['joint_safety_pass'] = d['old_tail_pass'] and d['final_new_noninferiority_pass']
            # Stage-arrival new NI is retained separately from the final last site.
            d['arrival_new_noninferiority'] = {}
            for i, s in enumerate(sites[1:], 1):
                x = comparison(scores(root/f'{dataset}_{i}_B', s), scores(root/f'{dataset}_{i}_C', s))
                d['arrival_new_noninferiority'][s] = x
        d['complete'] = done; summary['datasets'][dataset] = d
    summary['decision'] = ('ABANDON_CURRENT_PROJECT_PERFORMANCE_ROUTE' if all(complete.values()) and any(not d['numerically_within_2pp'] for d in summary['datasets'].values()) else
                           'NUMERICALLY_CLOSE_BUT_NOT_DIRECT_PAPER_VALIDATION' if all(complete.values()) else 'INCOMPLETE_OR_SOURCE_STOP_NO_SUCCESS_CLAIM')
    report = ['# 最后12小时双数据集验证', '', '固定单种子；当前L+U训练；适配不访问旧图；全部是重复使用的开发验证。眼底缺第二REFUGE相机，不称DC²T复现或直接超越。', '',
              '|数据集|完整|B站点均值%|C站点均值%|论文均值%|C数值距论文≤2pp|旧尾部|最后新域非劣|', '|---|---|---:|---:|---:|---|---|---|']
    for name, d in summary['datasets'].items():
        if d['complete']:
            report.append(f"|{name}|是|{d['final_mean']['B']:.4f}|{d['final_mean']['C']:.4f}|{d['paper_reference_mean']:.2f}|{d['numerically_within_2pp']}|{d['old_tail_pass']}|{d['final_new_noninferiority_pass']}|")
        else: report.append(f"|{name}|否|未完成|未完成|{d['paper_reference_mean']:.2f}|不可判定|不可判定|不可判定|")
    report += ['', '决策：'+summary['decision'], '', '完整阶段、患者配对CI/下降人数/p10、源门禁和物理成本：', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (root/'PUBLIC_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (root/'AUDIT.json').write_text(json.dumps(dict(status='PASS', retired=True, physical_optimizer=physical, model_forward=0, complete=complete, audited_epoch=time.time()), indent=2))
