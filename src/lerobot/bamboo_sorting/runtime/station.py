"""Read-only AUBO station, camera observation and wire image encoding.

No motion API is called here. SDK and camera imports are deferred to connect().
"""

import base64
import json
import math
import time
import zlib

from ..aubo_joint_contract import finite_vector
from ..joint_trial import TrialLimits, check_step
from ..paths import CONFIG_ROOT

CAMERA_SET = CONFIG_ROOT / 'CameraSetV2.json'
RPC_TIMEOUT_MS = 300


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def encode_images(images, codec):
    """Keep RGB order explicit; legacy lossless encoding remains for comparison."""
    import cv2
    encoded = {}
    for name, image in images.items():
        if image.shape != (480, 640, 3) or str(image.dtype) != 'uint8':
            raise ValueError('expected RGB uint8 [480,640,3]')
        if codec == 'rgb_zlib':
            payload = zlib.compress(image.tobytes(), 1)
        elif codec == 'jpeg95':
            ok, payload = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('JPEG encode failed')
        else:
            raise ValueError('unsupported image codec')
        encoded[name] = base64.b64encode(payload).decode()
    return encoded


class ReadOnlyStation:
    def __init__(self):
        self.rpc = None
        self.cameras = {}
        self.previous = {}

    def connect(self, *, include_cameras=True):
        import pyaubo_sdk
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from ..c0_smoke_capture import frozen_c0_smoke_camera_mapping
        mapping = frozen_c0_smoke_camera_mapping()
        for name, p in (mapping.items() if include_cameras else []):
            cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=p['device'], width=640, height=480,
                fps=30 if name == 'global_rgb' else 25, fourcc='MJPG', warmup_s=3))
            self.cameras[name] = cam
            cam.connect(warmup=True)
            cam.async_read_with_timestamp()
        self.rpc = pyaubo_sdk.RpcClient()
        self.rpc.setRequestTimeout(RPC_TIMEOUT_MS)
        self.rpc.connect('192.168.31.200', 30004)
        if not self.rpc.hasConnected():
            raise ConnectionError('AUBO not connected')
        self.rpc.login('aubo', '123456')
        if not self.rpc.hasLogined():
            raise ConnectionError('AUBO login failed')
        names = self.rpc.getRobotNames()
        if len(names) != 1:
            raise ValueError('ambiguous robot identity')
        self.iface = self.rpc.getRobotInterface(names[0])
        self.state = self.iface.getRobotState()
        self.config = self.iface.getRobotConfig()
        self.algorithm = self.iface.getRobotAlgorithm()
        # SDK exposes reads and writes on the same object; only getters below.
        self.io_readback = self.iface.getIoControl()
        self.lower = [math.degrees(v) for v in finite_vector(self.config.getJointMinPositions(), 6, 'lower')]
        self.upper = [math.degrees(v) for v in finite_vector(self.config.getJointMaxPositions(), 6, 'upper')]

    def current(self, *, require_stationary=True):
        if not self.state.isPowerOn():
            raise RuntimeError('robot is not in a safe powered state: isPowerOn=False (robot power is off)')
        if not self.state.isWithinSafetyLimits():
            raise RuntimeError('robot is not in a safe powered state: isWithinSafetyLimits=False')
        if self.state.isCollisionOccurred():
            raise RuntimeError('robot is not in a safe powered state: isCollisionOccurred=True')
        safety = self.state.getSafetyModeType()
        if getattr(safety, 'name', str(safety).split('.')[-1]) != 'Normal':
            raise RuntimeError(f'safety mode {safety}')
        ts = time.perf_counter()  # Timestamp the joint read, not earlier safety RPCs.
        q = [math.degrees(v) for v in finite_vector(self.state.getJointPositions(), 6, 'SDK joints')]
        if require_stationary:
            speeds = finite_vector(self.state.getJointSpeeds(), 6, 'SDK speeds')
            if max(abs(v) for v in speeds) > math.radians(.1):
                raise RuntimeError('robot moving: stationary trial required')
        tcp = finite_vector(self.state.getTcpPose(), 6, 'SDK TCP')
        pins = (self.io_readback.getStandardDigitalOutput(2), self.io_readback.getStandardDigitalOutput(3))
        if pins == (True, False):
            suction = 100.
        elif pins == (False, True):
            suction = 0.
        else:
            raise ValueError(f'unknown suction output pair: {pins}')
        return q + [suction], tcp[:3], ts

    def observe(self, *, require_stationary=True, state_snapshot=None, max_camera_age_ms=100):
        if state_snapshot is not None and require_stationary:
            raise ValueError('state reuse only applies to moving observations')
        images = {}
        times = []
        for name in ('global_rgb', 'grasp_rgb'):
            image, stamp = self.cameras[name].read_latest_with_timestamp(max_age_ms=max_camera_age_ms)
            if stamp <= self.previous.get(name, -1):
                raise ValueError(f'repeated camera frame: {name}')
            if image.shape != (480, 640, 3) or str(image.dtype) != 'uint8':
                raise ValueError('camera schema mismatch')
            images[name] = image.copy()
            times.append(stamp)
            self.previous[name] = stamp
        state, tcp, stamp = state_snapshot if state_snapshot is not None else self.current(
            require_stationary=require_stationary)
        times.append(stamp)
        return state, images, times

    def gate(self, action, sensor_times, *, captured_at=None, observed_state=None,
             limits=TrialLimits(), full_episode=False):
        state, tcp, stamp = self.current()
        current_fk, ret = self.algorithm.forwardKinematics([math.radians(v) for v in state[:6]])
        if ret != 0 or math.dist(finite_vector(current_fk, 6, 'current FK')[:3], tcp) > .001:
            raise ValueError('FK/current TCP mismatch; tool frame not verified')
        target_fk, ret = self.algorithm.forwardKinematics([math.radians(v) for v in action[:6]])
        if ret != 0:
            raise ValueError('target FK failed')
        target_fk = finite_vector(target_fk, 6, 'target FK')
        return check_step(action, current=state, lower=self.lower, upper=self.upper, current_tcp=tcp,
            target_tcp=target_fk[:3], sensor_times=sensor_times, state_time=stamp, now=time.perf_counter(),
            captured_at=captured_at, observed_state=observed_state, limits=limits, full_episode=full_episode)

    def close(self):
        errors = []
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception as exc:
                errors.append(str(exc))
        if self.rpc:
            try:
                if self.rpc.hasLogined():
                    self.rpc.logout()
                self.rpc.disconnect()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('cleanup: ' + '; '.join(errors))
