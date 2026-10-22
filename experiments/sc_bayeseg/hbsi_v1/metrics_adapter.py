"""Frozen metric implementations are imported only by isolated evaluators."""


def phase_metrics(probability, labels, spacing, classes):
    if classes == 2:
        from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics
        return binary_phase_metrics(probability, labels, spacing)
    if classes == 4:
        from sc_bayeseg.window_v1.patient_validation import phase_metrics as frozen
        return frozen(probability, labels, spacing)
    raise ValueError('MRI adapter supports audited binary or four-class tasks')


def patient_aggregate(rows, classes, phases):
    from sc_bayeseg.sslcl_v1.train import aggregate
    return aggregate(rows, classes, phases)


def fundus_metrics(probability, labels):
    from sc_bayeseg.sslcl_v1.fundus import image_metrics
    return image_metrics(probability, labels)
