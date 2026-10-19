from ..gyro_native_v1.data import current,ImagePool


def provider(cfg):
    data=current(cfg,'Siemens',69)
    if cfg['arm'] in ['calibration','GYRO_low']:data.pool=ImagePool(cfg)
    return data
