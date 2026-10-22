"""Shared synthetic masks for target-separation and review tests."""

import cv2
import numpy as np


def crossing_seeds():
    a = np.zeros((480, 640), np.uint8)
    b = a.copy()
    c = a.copy()
    cv2.line(a, (230, 100), (310, 400), 1, 15)
    cv2.line(b, (310, 100), (230, 400), 1, 15)
    cv2.line(c, (420, 100), (440, 400), 1, 15)
    # A is visibly on top at the crossing; B has two disconnected pieces.
    return {'s-001': a.astype(bool), 's-002': b.astype(bool) & ~a.astype(bool), 's-003': c.astype(bool)}


def instance_map(masks):
    array = np.zeros((480, 640), np.uint16)
    segments = []
    for i, mask in enumerate(masks.values(), 1):
        array[mask] = i * 7
        segments.append({'mask_id': i * 7, 'class_id': i % 2})
    return array, segments
