"""Zero-update source patient appearance swap with a fixed U-Net reader."""
import json
import os
import time
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F

from ..budgets import ResourceLedger
from ..data_adapter import confined, restore_probability
from ..fullfield_bayeseg_a1.run import records
from ..real_diagnostic import write
from ...sslcl_v1.prostate_metrics import binary_phase_metrics
from ..semantic_shape_l1.model import SemanticShape


def main():
    root=Path(os.environ['RUN_ROOT'])
    p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='APPEARANCE_SWAP_M0'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=1800,max_updates=0,
                          max_bytes=1_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    def load_arm(name):
        model=SemanticShape(name=='Cut').double().cuda()
        saved=torch.load(Path(p['l1_root'])/f'results/{name}/PRIVATE_CHECKPOINT_1600.pt',
                         map_location='cpu',weights_only=True)
        assert saved['step']==1600
        model.load_state_dict(saved['model'])
        model.eval()
        return model

    try:
        l1=Path(p['l1_root'])
        assert json.loads((l1/'EXIT.json').read_text())['exit_code']==0
        rows=records(p,'RUNMC','train_labeled')
        assert len(rows)==4
        images=[];labels=[];spacings=[]
        for row in rows:
            with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['image_h5_relpath']),'r') as f:
                images.append(torch.from_numpy(f['image'][...].copy())[:,None].double())
                spacings.append(json.loads(f.attrs['processed_spacing']))
            with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['label_h5_relpath']),'r') as f:
                labels.append(f['label'][...].copy())
        assert sum(len(v) for v in images)==80
        reference=torch.load(Path(p['unet_root'])/'results/U1600/PRIVATE_CHECKPOINT.pt',
                             map_location='cpu',weights_only=True)['model']
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=180,updates=0):
            models={name:load_arm(name) for name in ('Cut','Joint')}
            for name,model in models.items():
                state=model.base.unet.state_dict()
                assert set(state)==set(reference)
                assert max(float((state[k].cpu()-reference[k]).abs().max())
                           for k in state)==0
                with torch.no_grad():
                    image=F.interpolate(images[0][:1],(128,128),mode='bilinear',
                                        align_corners=False).cuda()
                    shape=model.shape_to_image(torch.tanh(model.means(image)[0]))
                    appearance=model.base.appearance_posterior(image)[:,:1]
                    assert torch.isfinite(shape).all() and torch.isfinite(appearance).all()
            assert all(rows[i]['patient_id']!=rows[(i+1)%4]['patient_id'] for i in range(4))
            write(root/'PREFLIGHT.json',dict(status='PASS',formal_updates=0,
                l1_complete=True,checkpoint_step1600=True,U_exact=True,
                source_patients=4,source_slices=80,partner_different=True,
                generator_finite=True))
        status('AUDIT')
        results={}
        with ledger.attempt('audit',seconds=1500,updates=0) as out:
            for arm,model in models.items():
                status('AUDIT',arm=arm)
                arm_rows=[]
                with torch.no_grad():
                    for i,(row,volume,truth,spacing) in enumerate(zip(rows,images,labels,spacings)):
                        partner=images[(i+1)%4]
                        predictions={key:[] for key in ('original','self','swap')}
                        reconstruction_sse=0.;reconstruction_pixels=0
                        swap_outside=0;swap_pixels=0
                        for k,slice_ in enumerate(volume):
                            ledger.check()
                            x=F.interpolate(slice_[None],(128,128),mode='bilinear',
                                            align_corners=False).cuda()
                            j=k*len(partner)//len(volume)
                            xpartner=F.interpolate(partner[j:j+1],(128,128),
                                mode='bilinear',align_corners=False).cuda()
                            shape=model.shape_to_image(torch.tanh(model.means(x)[0]))
                            own_a=model.base.appearance_posterior(x)[:,:1]
                            other_a=model.base.appearance_posterior(xpartner)[:,:1]
                            self_image=shape+own_a
                            swap_image=shape+other_a
                            assert torch.isfinite(self_image).all() and torch.isfinite(swap_image).all()
                            reconstruction_sse+=float((self_image-x).square().sum())
                            reconstruction_pixels+=x.numel()
                            swap_outside+=int(((swap_image < -1)|(swap_image > 1)).sum())
                            swap_pixels+=swap_image.numel()
                            for key,value in (('original',x),('self',self_image),('swap',swap_image)):
                                prob=model.base.unet(value).logits.softmax(1).cpu()
                                predictions[key].append(prob)
                        metrics={}
                        for key,parts in predictions.items():
                            prob=restore_probability(torch.cat(parts),volume.shape[-2:])
                            metrics[key]=binary_phase_metrics(prob,truth,spacing)
                            torch.save(prob,out/f'PRIVATE_{arm}_{i}_{key}.pt')
                        arm_rows.append(dict(patient=row['patient_id'],metrics=metrics,
                            reconstruction_mse=reconstruction_sse/reconstruction_pixels,
                            swap_outside_fraction=swap_outside/swap_pixels,
                            slices=len(volume)))
                write(out/f'PRIVATE_{arm}_PATIENTS.json',arm_rows)
                total_slices=sum(v['slices'] for v in arm_rows)
                results[arm]=dict(patients=4,slices=sum(v['slices'] for v in arm_rows),
                    reconstruction_mse=sum(v['reconstruction_mse']*v['slices']
                                           for v in arm_rows)/total_slices,
                    swap_outside_fraction=sum(v['swap_outside_fraction']*v['slices']
                                              for v in arm_rows)/total_slices,
                    **{key+'_Dice':sum(v['metrics'][key]['classes'][0]['Dice']
                                        for v in arm_rows)/4
                       for key in ('original','self','swap')})
        write(root/'RESULTS.json',dict(arms=results,formal_updates=0,
            source_patients=4,source_slices=80,target_data_read=False))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()
