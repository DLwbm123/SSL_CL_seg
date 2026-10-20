import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageOps
from ..di_dmpa_jascl.data import stable_seed

def paired_view(images, step, erase=False, return_masks=False):
    """Same grayscale photometric draws; optional legacy Cutout for exact checks.

    Spatial changes are shared in the weak view so dense pseudo labels stay aligned.
    This excludes classification RandAugment's independent geometric operations.
    """
    rng = np.random.default_rng(stable_seed(163, step, 'fixmatch_strong'))
    result = []; masks=[]
    for image in images.detach().cpu():
        pixels = ((image[0].numpy()+1)*127.5).clip(0,255).round().astype(np.uint8)
        im = Image.fromarray(pixels)
        for _ in range(2):
            op = int(rng.integers(8)); level = float(rng.uniform(0,1)); sign = int(rng.choice([-1,1]))
            if op == 1: im = ImageOps.autocontrast(im)
            elif op == 2: im = ImageOps.equalize(im)
            elif op == 3: im = ImageEnhance.Brightness(im).enhance(1+sign*.9*level)
            elif op == 4: im = ImageEnhance.Contrast(im).enhance(1+sign*.9*level)
            elif op == 5: im = ImageEnhance.Sharpness(im).enhance(1+sign*.9*level)
            elif op == 6: im = ImageOps.posterize(im, max(1,8-round(4*level)))
            elif op == 7: im = ImageOps.solarize(im, round(256*(1-level)))
        a = np.asarray(im).copy(); h,w = a.shape; side = min(h,w)//2
        cy,cx = int(rng.integers(h)),int(rng.integers(w))
        mask=np.zeros((h,w),dtype=bool)
        mask[max(0,cy-side//2):min(h,cy+side//2),max(0,cx-side//2):min(w,cx+side//2)]=True
        if erase:a[mask]=128
        masks.append(torch.from_numpy(mask))
        result.append(torch.from_numpy(a.astype(np.float32)/127.5-1)[None])
    output=torch.stack(result).to(images.device)
    return (output,torch.stack(masks)) if return_masks else output


def strong_view(images,step):
    return paired_view(images,step,erase=False)
