"""Reject-only boundary for an initial single-step joint trial, no SDK imports.

This is not a continuous policy controller or a collision certificate. A step
is allowed only near the established start, with no suction change. No target
is clipped or wrapped. A session is consumed even if arming/sending fails.
"""
import math
import time
from dataclasses import dataclass, replace

from .aubo_joint_contract import finite_vector, require_joint_target

START_DEG = (-65.29, -5.88, 113.77, 31.07, 90.88, -185.32)


@dataclass(frozen=True)
class TrialLimits:
    max_observation_age_s: float = .100
    max_state_age_s: float = .100
    max_camera_skew_s: float = .050
    max_joint_step_deg: float = .2
    max_joint_speed_deg_s: float = 5.
    max_tcp_step_m: float = .002
    max_tcp_speed_m_s: float = .05
    start_tolerance_deg: float = .5
    command_duration_s: float = .040
    max_stationary_prediction_wait_s: float = .250
    max_stationary_state_drift_deg: float = .020

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('limits must be finite and positive')


def stationary_step_limits():
    """A 200-ms human-triggered step, with the original velocity ceilings.

    This is an explicit separate experiment, not a 25-Hz policy loop.
    Position limits and the envelope around the start remain unchanged.
    """
    base = TrialLimits()
    duration = .2
    return replace(base, command_duration_s=duration,
                   max_joint_step_deg=base.max_joint_speed_deg_s*duration,
                   max_tcp_step_m=base.max_tcp_speed_m_s*duration)


def short_loop_limits():
    """Three-step experiment: explicit 3 x 0.5-degree start envelope."""
    return replace(stationary_step_limits(), start_tolerance_deg=1.5)


def check_step(action, *, current, lower, upper, current_tcp, target_tcp,
               sensor_times, state_time, now, limits=TrialLimits(),
               captured_at=None, observed_state=None, full_episode=False):
    """Check the actual first action; timestamps all use workstation monotonic."""
    action = finite_vector(action, 7, 'action')
    current = finite_vector(current, 7, 'current state')
    tcp = finite_vector(current_tcp, 3, 'current TCP')
    target = finite_vector(target_tcp, 3, 'target TCP')
    ts = finite_vector(sensor_times, 3, 'two cameras and model state timestamps')
    now, state_time = finite_vector([now, state_time], 2, 'clock')
    reasons = []
    if min(ts) < 0 or now < max(*ts, state_time) or state_time < 0:
        reasons.append('invalid_clock')
    if captured_at is None and observed_state is None:
        if now-min(ts) > limits.max_observation_age_s:
            reasons.append('expired_observation')
        timing_mode = 'continuous'
    elif captured_at is None or observed_state is None:
        raise ValueError('stationary mode needs both capture time and observed state')
    else:
        captured_at = finite_vector([captured_at], 1, 'capture time')[0]
        observed_state = finite_vector(observed_state, 7, 'observed state')
        timing_mode = 'stationary_single_step'
        if not max(ts) <= captured_at <= now:
            reasons.append('invalid_capture_clock')
        if captured_at-min(ts) > limits.max_observation_age_s:
            reasons.append('expired_at_capture')
        if now-captured_at > limits.max_stationary_prediction_wait_s:
            reasons.append('stationary_prediction_timeout')
        if (max(abs(a-b) for a,b in zip(observed_state[:6],current[:6])) > limits.max_stationary_state_drift_deg
                or observed_state[-1] != current[-1]):
            reasons.append('state_changed_during_prediction')
    if now-state_time > limits.max_state_age_s:
        reasons.append('expired_current_state')
    if abs(ts[0]-ts[1]) > limits.max_camera_skew_s:
        reasons.append('camera_skew')
    if current[-1] not in (0.,100.) or action[-1] not in (0.,100.):
        reasons.append('invalid_suction')
    if not full_episode and (current[-1] != 0 or action[-1] != current[-1]):
        reasons.append('single_step_requires_suction_off_and_unchanged')
    if not full_episode and max(abs(q-s) for q,s in zip(current[:6],START_DEG)) > limits.start_tolerance_deg:
        reasons.append('not_at_start')
    try:
        require_joint_target(action[:6],current[:6],lower,upper,limits.max_joint_step_deg)
    except ValueError as exc:
        reasons.append(str(exc))
    if not full_episode and max(abs(q-s) for q,s in zip(action[:6],START_DEG)) > limits.start_tolerance_deg:
        reasons.append('target_outside_start_envelope')
    if full_episode:
        # Same base-frame TCP workspace as record_joint.py, reject-only.
        for position in (tcp,target):
            if any(not lo <= value <= hi for value,lo,hi in zip(position,(-.8,-1.2,0.),(1.,0.,.8))):
                reasons.append('outside_capture_workspace')
                break
    dq = max(abs(a-b) for a,b in zip(action[:6],current[:6]))
    distance = math.dist(tcp,target)
    if dq/limits.command_duration_s > limits.max_joint_speed_deg_s:
        reasons.append('joint_command_speed')
    if distance > limits.max_tcp_step_m or distance/limits.command_duration_s > limits.max_tcp_speed_m_s:
        reasons.append('tcp_command_step_or_speed')
    return {'passed': not reasons, 'reasons': reasons, 'action': list(action),
            'timing_mode': timing_mode,
            'observation_age_s': now-min(ts), 'max_joint_delta_deg': dq, 'tcp_delta_m': distance}


class SingleStepSession:
    """One-shot SDK-like writer. Always disable servo; never retry or write IO."""
    def __init__(self, motion, *, clock=time.perf_counter, sleep=time.sleep, limits=TrialLimits()):
        self.motion, self.clock, self.sleep, self.limits = motion, clock, sleep, limits
        self.consumed = False
        self.command_attempted = False
        self.command_accepted = False
        self.servo_disabled = False

    def _require_zero(self, result, operation):
        if isinstance(result, bool) or result != 0:
            raise RuntimeError(f'{operation} failed: {result!r}')

    def _wait_servo(self, enabled, *, timeout_s=.025):
        # Same 5 x 5-ms state propagation allowance as the existing driver.
        # Poll status only; never resend the mode or motion command.
        attempts=math.ceil(timeout_s/.005)
        for attempt in range(attempts+1):
            if self.motion.isServoModeEnabled() == enabled:
                return
            if attempt < attempts:
                self.sleep(.005)
        raise RuntimeError('servo enable not confirmed' if enabled else
                           'servo remains enabled: use physical emergency stop')

    def execute(self, gate, *, revalidate):
        if self.consumed:
            raise RuntimeError('single-step session already consumed')
        self.consumed = True
        if gate.get('passed') is not True:
            raise ValueError('rejected action cannot be sent')
        if self.motion.isServoModeEnabled():
            raise RuntimeError('existing servo owner; refusing to take over')
        # No persistent speed-fraction setting, no generic send_action, no IO.
        try:
            self._require_zero(self.motion.setServoMode(True), 'enable servo')
            self._wait_servo(True)
            fresh = revalidate()
            if not fresh['passed'] or fresh['action'] != gate['action']:
                raise ValueError('action failed revalidation after servo enable')
            q = [math.radians(v) for v in fresh['action'][:6]]
            self.command_attempted = True
            self._require_zero(self.motion.servoJoint(q, math.radians(10),
                math.radians(self.limits.max_joint_speed_deg_s), self.limits.command_duration_s, 0., 200.), 'servoJoint')
            self.command_accepted = True
            self.sleep(self.limits.command_duration_s)
        finally:
            self._require_zero(self.motion.setServoMode(False), 'disable servo')
            self._wait_servo(False)
            self.servo_disabled = True


def wait_until_stationary(read_state, is_steady, *, clock=time.perf_counter, sleep=time.sleep):
    """Read feedback until settled; no motion commands or retries of actions."""
    deadline = clock()+1.
    while clock() < deadline:
        result = read_state()
        if is_steady():
            return result
        sleep(.02)
    raise TimeoutError('robot did not settle within one second')


def home_to_start(motion, read_state, *, lower, upper, is_steady,
                  clock=time.perf_counter, sleep=time.sleep, progress=None):
    """Explicit homing to the known start, suction off; no policy/IO.

    Uses move_to_start.py's 80/60-degree parameters and 50% speed fraction.
    Restores the previous fraction after stopping. No automatic retries.
    """
    state = finite_vector(read_state(), 7, 'homing state')
    distance = max(abs(a-b) for a,b in zip(state[:6],START_DEG))
    # Homing is a complete moveJoint trajectory, not a single policy step.
    # Validate actual/target joint limits without the old 10-degree proximity cap.
    require_joint_target(START_DEG, state[:6], lower, upper, max(distance, .1))
    if state[-1] != 0 or not is_steady():
        raise ValueError('homing requires a stationary robot and suction off')
    if motion.isServoModeEnabled():
        raise RuntimeError('existing servo owner; refusing homing')
    if distance <= .1:
        return {'moved':False, 'final_state':list(state)}
    def require_zero(ret, operation):
        if isinstance(ret, bool) or ret != 0:
            raise RuntimeError(f'{operation} failed: {ret!r}')
    previous_fraction=finite_vector([motion.getSpeedFraction()],1,'speed fraction')[0]
    if not 0<=previous_fraction<=1:raise ValueError('invalid controller speed fraction')
    timeout_s=max(30.,distance/(60.*.5)+15.)
    started=clock()
    if progress is not None:
        progress({'event':'start','state':list(state),'distance_deg':distance,'timeout_s':timeout_s,
                  'previous_speed_fraction':previous_fraction,'speed_fraction':.5,
                  'speed_deg_s':60.,'acceleration_deg_s2':80.})
    try:
        require_zero(motion.setSpeedFraction(.5),'homing speed fraction')
        require_zero(motion.moveJoint([math.radians(q) for q in START_DEG],
                                     math.radians(80), math.radians(60), 0., 0.), 'homing moveJoint')
        deadline = started+timeout_s
        while clock() < deadline:
            state = finite_vector(read_state(), 7, 'homing feedback')
            if state[-1] != 0:
                raise RuntimeError('suction changed while homing')
            error=max(abs(a-b) for a,b in zip(state[:6],START_DEG))
            steady=is_steady()
            if progress is not None:
                progress({'event':'feedback','elapsed_s':clock()-started,'state':list(state),
                          'max_error_deg':error,'is_steady':steady})
            if error <= .1 and steady:
                return {'moved':True, 'final_state':list(state), 'initial_max_distance_deg':distance,
                        'timeout_s':timeout_s,'speed_deg_s':60.,'acceleration_deg_s2':80.,
                        'speed_fraction':.5,'previous_speed_fraction':previous_fraction}
            sleep(.05)
        raise TimeoutError(f'homing did not arrive within {timeout_s:g} seconds')
    except BaseException:
        require_zero(motion.stopJoint(math.radians(80)), 'homing stopJoint')
        raise
    finally:
        # Avoid changing a global speed setting while an interrupted move is
        # still stopping. If stopping fails, retain the homing fraction.
        for attempt in range(101):
            if is_steady():
                require_zero(motion.setSpeedFraction(previous_fraction),'restore speed fraction')
                if progress is not None:progress({'event':'speed_fraction_restored','value':previous_fraction})
                break
            if attempt==100:raise RuntimeError('homing not stopped; speed fraction not restored')
            sleep(.02)
