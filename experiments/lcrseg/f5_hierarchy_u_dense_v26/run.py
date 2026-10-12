"""Training-only hierarchy addition; reuse the single V13 training/readout engine."""
import os
from pathlib import Path
from ..f5_module_pilots_v1 import run as engine
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..five_frameworks_v1.recipes import generator
from .model import HierarchyProtoModel,HierarchyProtoTrainer
from . import protocol

def make_model(parent,options,previous,rng,*,enabled=True):
    return HierarchyProtoModel(parent,'F5',ratio=options['rank_ratio'],previous=previous,joint_kl=True,hierarchy=enabled,dense=enabled,dense_weight=protocol.DENSE_WEIGHT,parent_map=protocol.HIERARCHY['parent_map'],hierarchy_weight=protocol.HIERARCHY['weight'],temperature=protocol.HIERARCHY['temperature'],margin=protocol.HIERARCHY['margin'])

def construct(config, j, stage, device, permit, previous=None, native=None, source_id=None, allow_u=True):
    if native is None:
        native, sr = engine.source(config, j['seed'], device)
        source_id = dict(node_id=sr['node_id'], student_hash=sr['student_hash'], seed=j['seed'], domain='REFUGE')
    provider = NativeCurrentDomain(config['data'], j['seed'], j['order'], stage, source_id, device, permit, allow_u=allow_u)
    options = {**protocol.OPTIONS, 'total_steps': protocol.STEPS[provider.domain]}
    model = make_model(NativeLRParent(native, j['seed'], source_id), options, previous,
                       generator(j['seed'], j['order'], stage, 0, 'pilot_adapter_initialization')).to(device)
    return HierarchyProtoTrainer(model, provider, options, execution=permit)


def bind():
    for name in ('plan', 'jobs', 'CAPS', 'ARMS', 'compare'):
        setattr(engine, name, getattr(protocol, name))
    engine.construct = construct
    original_readout=engine.final_readout
    def readout(config):
        original_readout(config)
        root=Path(config["run_root"]);r=protocol.read(root/"RESULTS.json")
        for version,arm in ((25,"HIERARCHY_DENSE"),(24,"HIERARCHY_ORIENTATION"),(23,"HIERARCHY_RELATION"),(21,"HIERARCHY_PROTO"),(22,"TANGENT_PROTO"),(13,"CLASS_REFERENCE"),(15,"CLASS_CONE"),(17,"SARD"),(19,"CARE"),(20,"BLOCK_GUARD"),(10,"ANCHOR_GUARD")):
            rows=protocol.read(config[f"v{version}_comparison_results"])["rows"]
            r[f"vs_V{version}"]=protocol.paired_compare(r["rows"]+rows,arm,arms=("HIERARCHY_U_DENSE",))
        protocol.write(root/"RESULTS.json",r)
    engine.final_readout=readout


def main():
    bind()
    config = protocol.read(os.environ['EXEC_CONFIG'])
    if os.environ.get('EXEC_MODE') == 'qualify':
        from .tests import qualify
        qualify(config)
    else:
        engine.main()
