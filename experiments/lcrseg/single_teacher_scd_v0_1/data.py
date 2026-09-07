"""Role-scoped canonical HDF5 access and stateless matched random streams."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from experiments.lcrseg.di_dmpa_gate1.binding import safe_asset, check_hash
from experiments.lcrseg.di_dmpa_jascl.data import stable_seed, batch_indices

DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
COUNTS = ((40, 160), (16, 63), (10, 41))
MANIFEST_SHA = "0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3"
SPLIT_SHA = "f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88"


def metadata(data, *, expected=(MANIFEST_SHA, SPLIT_SHA)):
    data = Path(data)
    manifest, split = data / "manifests/training/lcrseg_v1_seed0.csv", data / "splits/fundus_seed0.json"
    for path, sha in zip((manifest, split), expected):
        check_hash(path, sha)
    with manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == "fundus"]
    split_payload = json.loads(split.read_text())
    if split_payload["seed"] != 0:
        raise ValueError("only seed0 is authorized")
    split_rows = {r["case_id"]: r for r in split_payload["records"]}
    seen = {}
    cases = set()
    for r in rows:
        if r["split_seed"] != "0" or r["case_id"] in cases:
            raise ValueError("wrong seed or duplicate case")
        cases.add(r["case_id"])
        if r["site_or_vendor"] not in DOMAINS or r["primary_20pct_split"] not in ("train_labeled", "train_unlabeled", "val", "test"):
            raise ValueError("unknown role/domain")
        sr = split_rows[r["case_id"]]
        for field in ("patient_id", "site_or_vendor", "primary_20pct_split"):
            if r[field] != sr[field]:
                raise ValueError("manifest/split identity mismatch")
        identity = (r["site_or_vendor"], r["primary_20pct_split"])
        if seen.setdefault(r["patient_id"], identity) != identity:
            raise ValueError("patient crosses stage or role")
        if r["primary_20pct_split"] == "train_unlabeled" and (r["label_h5_relpath"] or r["label_sha256"]):
            raise ValueError("unlabeled GT metadata is forbidden")
    return rows


class CurrentData:
    def __init__(self, data, stage, role, *, purpose="train", domain=None, expected=(MANIFEST_SHA, SPLIT_SHA), shape=(384, 384)):
        if stage not in (0, 1, 2):
            raise ValueError("invalid stage")
        domain = stage if domain is None else domain
        if purpose == "train":
            if domain != stage or role not in ("train_labeled", "train_unlabeled"):
                raise PermissionError("trainer cannot access other domains or evaluation")
        elif purpose == "evaluate":
            if role != "val" or not 0 <= domain <= stage:
                raise PermissionError("only seen-domain val is authorized")
        else:
            raise PermissionError("invalid accessor purpose")
        self.data, self.stage, self.role, self.shape = Path(data), stage, role, tuple(shape)
        raw = metadata(data, expected=expected)
        selected = sorted((r for r in raw if r["site_or_vendor"] == DOMAINS[domain] and r["primary_20pct_split"] == role), key=lambda r:r["case_id"])
        self.rows = []
        for r in selected:
            # Strip all forbidden fields before constructing the unlabeled accessor.
            item = {k:r[k] for k in ("case_id", "image_h5_relpath", "image_sha256")}
            if role != "train_unlabeled":
                item.update({k:r[k] for k in ("label_h5_relpath", "label_sha256")})
            self.rows.append(item)
        if not self.rows:
            raise ValueError("empty authorized dataset")
        self.checked = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        result = {}
        for key in (("image",) if self.role == "train_unlabeled" else ("image", "label")):
            relative = r[key + "_h5_relpath"]
            if not relative:
                raise ValueError("missing canonical relative path")
            path = safe_asset(self.data, relative)
            if (i,key) not in self.checked:
                check_hash(path, r[key + "_sha256"])
            with h5py.File(path, "r") as f:
                array = f[key][...]
            if key == "image":
                if array.shape != (3, *self.shape) or array.dtype != np.uint8:
                    raise ValueError("invalid canonical RGB geometry/dtype")
                result[key] = torch.from_numpy(array.astype(np.float32) / 255.)
            else:
                if array.shape != self.shape or not np.isin(array, (0,1,2,255)).all():
                    raise ValueError("invalid canonical label geometry/classes")
                result[key] = torch.from_numpy(array.astype(np.int64))
            self.checked.add((i,key))
        result["geometry"] = torch.ones(self.shape, dtype=torch.bool)
        return result


def batches(size, stage, epoch, stream, steps):
    result, cycle = [], 0
    while len(result) < steps:
        result.extend(v for _,v in batch_indices(size, 2, shuffle=True,
            seed_parts=(0,stage,epoch,stream,cycle)))
        cycle += 1
    return result[:steps]


def geometry(image, label, valid, generator):
    dimensions = []
    if torch.rand((), generator=generator) < .5:
        dimensions.append(-1)
    if torch.rand((), generator=generator) < .5:
        dimensions.append(-2)
    k = int(torch.randint(4, (), generator=generator))
    def change(x):
        if dimensions:
            x = x.flip(dimensions)
        return x.rot90(k, (-2,-1)).contiguous()
    return change(image), None if label is None else change(label), change(valid)


def load_batch(dataset, indices, stage, epoch, step, stream, device, *, augment=True):
    items = []
    g = torch.Generator().manual_seed(stable_seed(0,stage,epoch,step,"geometry",stream))
    for i in indices:
        item = dataset[i]
        if augment:
            x,y,v = geometry(item["image"], item.get("label"), item["geometry"], g)
            item = dict(image=x, geometry=v)
            if y is not None:
                item["label"] = y
        items.append(item)
    return {k:torch.stack([x[k] for x in items]).to(device) for k in items[0]}


def strong(weak, stage, epoch, step):
    g = torch.Generator(device=weak.device).manual_seed(stable_seed(0,stage,epoch,step,"strong"))
    outputs = []
    weights = weak.new_tensor([.2989,.587,.114])[:,None,None]
    for image in weak:
        factors = .8 + .4 * torch.rand(3, generator=g, device=weak.device)
        x = image * factors[0]
        gray = (x * weights).sum(0,keepdim=True)
        x = (x - gray.mean()) * factors[1] + gray.mean()
        gray = (x * weights).sum(0,keepdim=True)
        x = (x - gray) * factors[2] + gray
        if torch.rand((),generator=g,device=weak.device) < .5:
            sigma = .1 + .9 * torch.rand((),generator=g,device=weak.device)
            axis = torch.arange(-2,3,device=weak.device,dtype=x.dtype)
            kernel = (-axis.square()/(2*sigma.square())).exp()
            kernel = kernel / kernel.sum()
            kernel = (kernel[:,None]*kernel[None,:]).expand(3,1,5,5)
            x = F.conv2d(F.pad(x[None],(2,2,2,2),mode="reflect"),kernel,groups=3)[0]
        if torch.rand((),generator=g,device=weak.device) < .5:
            x = x + .02 * torch.randn(x.shape,generator=g,device=x.device)
        outputs.append(x.clamp(0,1))
    return torch.stack(outputs)
