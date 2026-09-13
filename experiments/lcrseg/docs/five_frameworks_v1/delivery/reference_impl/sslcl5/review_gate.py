"""Fail-closed workflow gate. NOT a substitute for genuine external approval."""
from __future__ import annotations
from typing import Mapping


class ReviewRequired(RuntimeError):
    pass


def require_execution_approval(receipt: Mapping, actual: Mapping,
                               requested_phases: list[str], requested_updates: int) -> None:
    if receipt.get('is_template',True) or receipt.get('decision')!='APPROVED_FOR_EXPERIMENTS':
        raise ReviewRequired('No genuine reviewed approval; remain CODE_ONLY')
    if receipt.get('user_launch_confirmation') is not True or not receipt.get('review_evidence'):
        raise ReviewRequired('Explicit user launch confirmation and external review evidence required')
    for key in ('reviewed_code_commit','reviewed_code_tree_sha256','parent_binding_sha256',
                'expanded_plan_sha256','external_dependency_lock_sha256'):
        value=receipt.get(key)
        if not isinstance(value,str) or not value or actual.get(key)!=value:
            raise ReviewRequired(f'Review binding mismatch: {key}')
    if not set(requested_phases)<=set(receipt.get('approved_phases',[])):
        raise ReviewRequired('Phase is outside reviewed scope')
    cap=receipt.get('max_formal_optimizer_updates')
    if type(cap) is not int or type(requested_updates) is not int or not 0<=requested_updates<=cap:
        raise ReviewRequired('Unresolved or excessive optimizer budget')
