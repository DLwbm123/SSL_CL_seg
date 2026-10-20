"""Format aggregate inference diagnostics without new model evaluations."""
import json,csv
from pathlib import Path

def main(folder):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text())
    t='''# GYRO几何—分割功能机制诊断

用户明确授权的单次诊断，优化器更新0，不恢复参数搜索或分组裁剪。选择source_coordinate_v1 SGD G/M的同一入口和固定2000终态；全模型eval且无梯度，所有原参数/frame/projector诊断后精确不变。与BayeSeg/CARE独立，源训练图像回放路线，不是history-free。

## encoder—readout交叉

E0/D0为入口，E1/D1为终态；encoder包含全部skip和bottleneck，readout包含完整decoder/原头。入口/原生终态复用已有Siemens验证记录，只新增两个混合格每组各544切片。混合模型协同失配可能影响结果，不能作严格可加因果分解。

| 组 | E0D0% | E1D0% | E0D1% | E1D1% | ΔE pp | ΔD pp | 交互 pp |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    records=[]
    for arm,c in s['cross_grid'].items():
        v={k:100*x['primary_dice'] for k,x in c.items()};de=v['E1_D0']-v['E0_D0'];dd=v['E0_D1']-v['E0_D0'];interaction=v['E1_D1']-v['E1_D0']-v['E0_D1']+v['E0_D0']
        t+='| '+arm+' | '+' | '.join(f'{x:.4f}' for x in [v['E0_D0'],v['E1_D0'],v['E0_D1'],v['E1_D1'],de,dd,interaction])+' |\n'
        records.append([arm,v['E0_D0'],v['E1_D0'],v['E0_D1'],v['E1_D1'],de,dd,interaction])
    with (folder/'cross_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['arm','E0D0_percent','E1D0_percent','E0D1_percent','E1D1_percent','encoder_compatibility_pp','readout_compatibility_pp','interaction_pp']);w.writerows(records)
    t+='\n## 各类和ED/ES\n\n| 组/格 | LV% | MYO% | RV% | ED% | ES% |\n|---|---:|---:|---:|---:|---:|\n'
    for arm,cells in s['cross_grid'].items():
        for name,r in cells.items():t+=f'| {arm}/{name} | '+' | '.join(f'{100*x:.4f}' for x in [*[r['classes'][k] for k in ['LV','MYO','RV']],*[r['phases'][k]['primary_dice'] for k in ['ED','ES']]])+' |\n'
    t+='\n## 患者配对\n\n| 组/格相对入口 | 改善 | 下降 | 不变 | 中位数 pp |\n|---|---:|---:|---:|---:|\n'
    for arm,rs in s['cross_paired'].items():
        for k,r in rs.items():t+=f"| {arm}/{k} | {r['increased']} | {r['decreased']} | {r['unchanged']} | {r['median_patient_delta_pp']:+.4f} |\n"
    t+='\n## 同图几何与输出变化\n\n固定源L每体中心切片20图（10患者EDES）及bank固定前16源U条目。V2质心、同源阶段0锚点；源U绝不读取GT，输出漂移不是预测错误的同义词。投影质心cosine漂移<=.01且锚不变是预设局部稳定标记，前景并集argmax不一致>=.1是输出漂移标记。阈值只描述，不证明全部几何稳定。\n\n| 组/样本 | n | 投影漂移中位 | bottleneck局部漂移中位 | decoder局部漂移中位 | 前景输出不一致中位 | 锚改变 | 稳定几何 | 稳定且输出漂移 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for arm,roles in s['geometry'].items():
        for role,r in roles.items():
            m=r['metrics'];t+=f"| {arm}/{role} | {r['samples']} | {m['projection_centroid_cosine_drift']['median']:.6f} | {m['bottleneck_token_cosine_drift']['median']:.6f} | {m['decoder_token_cosine_drift']['median']:.6f} | {m['foreground_union_disagreement']['median']:.6f} | {r['anchor_changes']} | {r['geometry_stable_count']} | {r['stable_geometry_but_function_drift_count']} |\n"
    t+='\n## 源L中心切片标签诊断\n\n以下是20张中心切片按10患者/EDES聚合，绝不是全量源10L训练Dice，也不是独立泛化。\n\n| 状态 | LV% | MYO% | RV% | 均值% |\n|---|---:|---:|---:|---:|\n'
    for a,r in s['center_slice_Dice'].items():t+=f'| {a} | '+' | '.join(f'{100*x:.4f}' for x in [*[r['classes'][k] for k in ['LV','MYO','RV']],r['primary_dice']])+' |\n'
    t+='\n源L前景准确率变化与标准化真类margin、其余局部层漂移和描述性相关见SUMMARY。相关不作独立患者/seed显著性证据。argmax和Dice不因单纯置信度提高而改善。\n\n## 成本、合同与边界\n\n真实完整分割model-slices 2284（混合2176+同图108），另encoder-only108，合成32×32分割9；优化器0、无backward、无权重/图像/特征保存。前向预留/执行账本一致，三模型自交换精确、手动完整skip读取路径与原native一致，原参数/frame/projector全state不变。仅合法Siemens验证及源L标注、源U image-only bank；不接触test或U隐藏GT。未重新评估旧原生终点，未自动执行新训练。公开仅源码/协议/匿名聚合报告；全部患者身份、图像、权重、特征、原始记录与配置留私有。20GB总新增硬cap持续有效。\n'
    (folder/'REPORT_ZH.md').write_text(t)
