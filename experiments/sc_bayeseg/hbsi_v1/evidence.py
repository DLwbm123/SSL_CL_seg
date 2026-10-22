"""Canonical identities of observation events, independent of stage names and output paths."""
from .qualification import fingerprint


def evidence_set(data_version, domain, observations):
    if not isinstance(data_version, str) or not data_version or not domain or not observations:
        raise ValueError('nonempty data version, domain and observation events required')
    events = []
    for item in observations:
        if item.get('kind') not in ('image', 'label') or not item.get('record'):
            raise ValueError('invalid observation event')
        # Image identity excludes its L/U role and task name: a newly labelled image is still the same image.
        identity = dict(data_version=data_version, record=item['record'], kind=item['kind'])
        if item['kind'] == 'label':
            if not item.get('annotation_version'):
                raise ValueError('label observation version required')
            identity['annotation_version'] = item['annotation_version']
        events.append(fingerprint(identity))
    if len(set(events)) != len(events):
        raise ValueError('duplicate events within current evidence')
    payload = dict(schema=1, data_version=data_version, domain=domain, events=sorted(events))
    return dict(payload, fingerprint=fingerprint(payload))


def validate_evidence(evidence):
    payload = {k: evidence[k] for k in ('schema', 'data_version', 'domain', 'events')}
    if (payload['schema'] != 1 or not payload['events'] or
            payload['events'] != sorted(set(payload['events'])) or
            any(not isinstance(v, str) or len(v) != 64 for v in payload['events']) or
            fingerprint(payload) != evidence['fingerprint']):
        raise ValueError('invalid evidence identity')
    return set(payload['events'])
