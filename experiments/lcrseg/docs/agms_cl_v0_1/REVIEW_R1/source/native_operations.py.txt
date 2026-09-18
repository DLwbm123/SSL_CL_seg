"""Extend the existing operation recorder for the designated native bridge."""
import functools,json
from unittest.mock import patch
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from .native_parent import NativeLRParent


class NativeOperations(Operations):
    def __enter__(self):
        previous=self.root/'operation_counts.json'
        if previous.exists():self.counts.update(json.loads(previous.read_text())['counts'])
        super().__enter__()
        for method in ('features','native_readout','update_dense_ema'):
            original=getattr(NativeLRParent,method)
            def wrap(fn,name):
                @functools.wraps(fn)
                def call(parent,*a,**kw):
                    label=name+('_validation' if kw.get('validate_only') else '')
                    self.counts[label+'_attempts']+=1
                    if name=='features' and kw.get('overrides'):
                        self.counts['probe_override_extra_convolution_forwards']+=len(kw['overrides'])
                    result=fn(parent,*a,**kw);self.counts[label]+=1
                    return result
                return call
            self.stack.enter_context(patch.object(NativeLRParent,method,wrap(original,method)))
        return self
