"""Canonical320 admission without modifying the frozen training manifests."""
from lcrseg.data import H5LabeledDataset,H5UnlabeledDataset,LabeledTransform,WeakStrongTransform


def datasets(root,site):
    options=dict(data_root=root,seed=0,dataset='mnms',sites=[site])
    l=H5LabeledDataset(**options,transform=LabeledTransform())
    u=H5UnlabeledDataset(**options,transform=WeakStrongTransform(cutout_probability=0))
    for dataset in (l,u):
        dataset.records=[r for r in dataset.records if r['cohort']=='canonical320' and r['evaluation_eligible']=='true']
        allowed={r['case_id'] for r in dataset.records}
        dataset.samples=[s for s in dataset.samples if s.row['case_id'] in allowed]
        if not dataset.samples:raise ValueError('empty admitted site population')
    return l,u
