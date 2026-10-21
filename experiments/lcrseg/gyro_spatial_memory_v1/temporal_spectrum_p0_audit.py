"""Once-only CPU audit and aggregate P0 report; never evaluates a model."""
from pathlib import Path
import json,collections
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main(cfg):
    p=Path(cfg['root']);assert not (p/'AUDIT.json').exists();assert read(p/'STATUS.json')['status']=='COMPLETE' and read(p/'EXIT.json')['exit_code']==0
    for name in ['LAUNCH.private.json','WORKER.private.json']:
        r=read(p/name);s=Path('/proc')/str(r['pid'])/'stat';assert not s.exists() or s.read_text().split()[21]!=str(r['birth'])
    c=read(p/'COUNTER.json');assert c['pending'] is None and c['optimizer']==c['functional']==c['reverse']==0 and c['executed']=={'native_validation':9632,'encoder_training':5376}
    events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];counts={a:collections.Counter() for a in ['reserved','executed']}
    for e in events:counts[e['action']][e['kind']]+=e['n']
    assert dict(counts['reserved'])==dict(counts['executed'])==c['executed']
    result=read(p/'RESULTS.json');assert result['read_only_exact'] and len(result['exchange'])==8 and len(result['features'])==7 and all(len(v)==16 for v in result['features'].values())
    assert result['optimizer']==result['functional']==result['reverse']==0
    audit={'status':'PASS','counts':c,'events_reserved_executed_exact':True,'workers_retired':True,'read_only_exact':True,'exchange_models':8,'feature_states':7,'feature_batches_per_state':16,'old_endpoints_unchanged':True,'audit_model_forwards':0,'import_failure_before_forward_retained':(p/'failures/attempt0/FAILURE.json').exists()}
    write(p/'AUDIT.json',audit);write(p/'PUBLIC_SUMMARY.json',{'audit':audit,'results':result})
    r=['# P0：患者代价、encoder–decoder兼容性及谱分解完整结果','','0更新，9632 native验证切片+5376 encoder训练图像暴露；完整旧学生/EMA结果复用。一次CPU审计PASS，所有受管进程退休。未改变旧终点。','','## 患者损益与幅度','','阈值只作开发描述，不是临床重要性标准。负中位数不自动等于多数患者明显受损。','','|流/域|比较|均值/中位/p10 pp|下降>.5/1/2pp人数|留一均值范围pp|','|---|---|---:|---:|---:|']
    for seed,domains in result['patient_analysis']['comparisons'].items():
        for domain,cc in domains.items():
            for name,d in cc.items():r.append(f"|{seed}/{domain}|{name}|{d['mean_pp']:+.6f}/{d['median_pp']:+.6f}/{d['p10_pp']:+.6f}|{d['decline_over_pp']['0.5']}/{d['decline_over_pp']['1']}/{d['decline_over_pp']['2']}|{d['leave_one_mean_range_pp']}|")
    r+=['','跨流同患者差值一致性（不输出身份，不据此选择训练患者）：', '',json.dumps(result['patient_analysis']['cross_stream_difference_consistency'],ensure_ascii=False,indent=2),'','## 只读模块交换','','非可加因果分解，不选择新主模型。','','|流/方法/组合|旧域Dice%|新域Dice%|旧/新相对学生pp|','|---|---:|---:|---:|']
    for name,domains in result['exchange'].items():
        a,b=domains['Siemens'],domains['Philips'];r.append(f"|{name}|{a['metrics']['primary_dice']*100:.6f}|{b['metrics']['primary_dice']*100:.6f}|{a['versus_student']['delta_B_minus_A_pp']:+.6f}/{b['versus_student']['delta_B_minus_A_pp']:+.6f}|")
    r+=['','## 固定同批训练图像的特征分解','','七状态均使用seed20261028、16相同批次与位置；未中心化矩阵与中心化协方差严格区分，零分母NA。跨批均值不是把所有批合并后的谱。','','|状态|centroid平均平方范数|r*未中心化|r*中心化|协方差trace|','|---|---:|---:|---:|---:|']
    for name,stats in result['features'].items():
        def mean(k):
            v=[x[k] for x in stats if x[k] is not None];return None if not v else float(np.mean(v))
        r.append(f"|{name}|{mean('centroid_mean_squared_norm')}|{mean('r_star_uncentered')}|{mean('r_star_centered')}|{mean('centered_covariance_trace')}|")
    r+=['','完整奇异值、跨视图距离、协方差谱、患者类/EDES及bootstrap在PUBLIC_SUMMARY。谱多样性不等于器官语义保持，交换改善不证明encoder震荡或decoder单独失效。','','## 执行与继续规则','','原始2h/<500MB硬预算、所有新产物独立root，旧checkpoint/数据只读。一次依赖遗漏在import阶段失败、0前向/更新，现场保留；补齐既有依赖、同原时钟恢复，没有重置成本或增加cap。实际时长、占用和峰值见RESULTS resources。','','工程检查通过后可进入另行冻结的S/G/T/GT机制实验；P0不要求某交换组合改善，不据验证分数选择权重/入口/seed。下一轮最多16320optimizer，校准/教师/评价另列硬预算，仍须先完整解释本报告。U/frame/生成记忆/强VC等条件分支不展开。']
    (p/'REPORT_ZH.md').write_text('\n'.join(r)+'\n')
