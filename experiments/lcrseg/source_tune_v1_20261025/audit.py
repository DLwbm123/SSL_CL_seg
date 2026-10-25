"""One terminal saved-result audit; no model construction or forward."""
import json
import time
from pathlib import Path


def main(cfg):
    root = Path(cfg['campaign_root'])
    assert not (root/'AUDIT.json').exists()
    launch = json.loads((root/'LAUNCH.private.json').read_text())
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            s = (p/'stat').read_text(); fields = s[s.rindex(')')+2:].split()
            assert not (fields[0] != 'Z' and int(fields[2]) == launch['process_group']), p.name
        except (FileNotFoundError, PermissionError): pass
    terminal = json.loads((root/'TERMINAL.json').read_text())
    physical = sum(sum(1 for _ in p.open()) for p in root.glob('C*/physical_optimizer.jsonl'))
    assert physical == terminal['counts']['optimizer'] <= cfg['optimizer_cap']
    for arm in cfg['arms']:
        p = root/arm['name']/'mnms_0_SOURCE'
        if arm['name'] not in terminal['results']: continue
        assert json.loads((p/'QUALIFICATION.json').read_text())['status'] == 'PASS'
        assert json.loads((p/'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((p/'EXIT.json').read_text())['step'] == 6000
    report = ['# 心脏源模型四配置开发调优结果', '',
              '共享原FixMatch EMA；单种子；验证集已多次使用且本轮用于选择，不是独立测试。ED+ES为唯一主指标；不据ED单独结果声称追平论文。没有持续适配。', '',
              '|配置|ED+ES Dice%|ED Dice%|ES Dice%|合并变化pp|配对CI pp|>1/>2|变化p10|能力门禁|',
              '|---|---:|---:|---:|---:|---|---|---:|---|']
    for name, r in terminal['results'].items():
        d = r['phase']; z = d['ED_ES']['contrast']
        report.append(f"|{name}|{d['ED_ES']['Dice_percent']:.4f}|{d['ED']['Dice_percent']:.4f}|{d['ES']['Dice_percent']:.4f}|{z['mean_pp']:+.4f}|{z['CI95_pp']}|{z['over1']}/{z['over2']}|{z['p10_pp']:.4f}|{r['source_capability_pass']}|")
    report += ['', '开发候选：'+str(terminal['selected_development_candidate']), '',
               '门禁不通过的配置完整保留；通过也只构成开发验证证据，未证明独立泛化或训练收敛。', '',
               '```json', json.dumps(terminal, ensure_ascii=False, indent=2), '```']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (root/'PUBLIC_SUMMARY.json').write_text(json.dumps(terminal, ensure_ascii=False, indent=2))
    (root/'AUDIT.json').write_text(json.dumps(dict(status='PASS', retired=True, optimizer=physical, model_forward=0, audited_epoch=time.time()), indent=2))
