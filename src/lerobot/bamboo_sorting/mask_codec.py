"""PNG/base64 mask encoding shared by perception and runtime modules."""

import base64

import cv2
import numpy as np


def encode_mask(mask, *, allow_empty=False):
    if mask.dtype != np.bool_ or mask.shape != (480, 640) or (not allow_empty and not mask.any()):
        raise ValueError('nonempty bool mask required')
    ok, encoded = cv2.imencode('.png', mask.astype(np.uint8))
    if not ok:
        raise ValueError('mask encoding failed')
    return base64.b64encode(encoded).decode('ascii')


def decode_mask(encoded, *, allow_empty=False):
    if not isinstance(encoded, str) or len(encoded) > 100000:
        raise ValueError('invalid seed mask payload')
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != (480, 640) or not np.isin(mask, [0, 1]).all() or (not allow_empty and not mask.any()):
        raise ValueError('nonempty binary 480x640 mask required')
    return mask.astype(bool)
