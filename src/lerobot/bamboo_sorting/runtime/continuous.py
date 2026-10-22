"""Continuous 25-Hz servo run: connection, target loop, switching and teardown.

This module contains the execution implementation; the command entry only
parses arguments, prints the plan and calls :func:`execute_continuous`.
Connections and recorder threads are only created after the operator confirms
the run. Exceptions still propagate to the single outer cleanup boundary.
"""

import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from ..aubo_joint_contract import JOINT_FIELDS, finite_vector, require_joint_target
from ..policy.joint_target_policy import TargetInputSession, is_transport
from .inference import InferencePipe
from .joint_smooth import (
    DT, MAX_ACCEL, MAX_SPEED, MAX_TRACKING_ERROR_DEG, SmoothTrajectory,
    select_approach_action, select_transport_action, validate_prediction, workspace,
)
from .joint_target_runtime import TargetForegroundMissing
from .joint_trial import START_DEG, SingleStepSession, home_to_start, wait_until_stationary
from .joint_video import TrialVideoRecorder
from .station import ReadOnlyStation, write
from .task_prompt import read_task

MAX_TARGET_REOBSERVATIONS = 3  # Per target, including separate intermittent misses.


@dataclass
class ContinuousState:
    """Inputs and mutable state for one continuous run."""

    args: object
    sequence: object
    target_mode: bool
    target_seed: object
    target_options: object
    target_setups: object
    target_session: object
    plan: dict
    config: dict
    task: str
    output: Path
    station: object = None
    pipe: object = None
    pool: object = None
    servo: object = None
    owned: bool = False
    failure: str | None = None
    initial_state_known: bool = False
    reason: str = "not_started"
    sent: int = 0
    predictions: int = 0
    completed_cycles: int = 0
    cycle: bool = False
    current_target_id: object = None
    recorder: object = None
    video_result: object = None
    preview: object = None
    motion_tracker: object = None
    trace: object = None
    commands: object = None
    suction: object = None
    motion: object = None
    future: object = None
    goal: object = None
    pending_suction: bool = False
    last_observation: object = None
    trajectory: object = None
    active_chunk: object = None
    active_times: object = None
    selection: object = None
    prediction_id: object = None
    active_transport: bool = False
    discard_pending_result: bool = False
    request_index: int = 0
    inflight_index: object = None
    current_state: object = None
    current_tcp: object = None
    current_stamp: object = None
    observed_state: object = None
    images: object = None
    sensor_times: object = None
    metadata: object = None
    last_servo_sent_at: object = None
    target_misses: int = 0
    recovery_after: object = None
    deadline: float = 0.0
    next_tick: float = 0.0
    last_tick: float = 0.0
    prediction_deadline: object = None
    held_suction: float = 0.0


def log(handle, value):
    handle.write(json.dumps(value, allow_nan=False) + "\n")


def save_target_failure(output, index, state, images, sensor_times, metadata, exc):
    """Lossless request inputs and same-frame segmentation; never replace a run."""
    import base64
    import cv2
    folder = output / "target_failures" / f"{index:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "failure.json", dict(index=index, observed_state=state, sensor_times=sensor_times,
        target_input=metadata, error=str(exc), error_code=exc.code, target_evidence=exc.evidence))
    for name, img in images.items():
        if not cv2.imwrite(str(folder / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("failed input image save failed")
    (folder / "segmentation_instances.png").write_bytes(base64.b64decode(
        exc.evidence["segmentation_instances_png"], validate=True))
    return str(folder.relative_to(output))


def forward_kinematics(state, q):
    pose, ret = state.station.algorithm.forwardKinematics([math.radians(x) for x in q])
    if ret != 0:
        raise ValueError(f"FK failed: {ret}")
    return list(finite_vector(pose, 6, "FK")[:3])


def execute_continuous(*, args, sequence, target_mode, target_seed, target_options,
                       target_setups, target_session, plan, config):
    """Run one explicit live session; no device is touched before confirmation."""
    try:
        task = read_task()
    except (EOFError, KeyboardInterrupt):
        return 0
    if task is None:
        return 0
    plan["task"] = task
    output = args.output or Path("artifacts") / ("joint_smooth_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    state = ContinuousState(args=args, sequence=sequence, target_mode=target_mode,
        target_seed=target_seed, target_options=target_options, target_setups=target_setups,
        target_session=target_session, plan=plan, config=config, task=task, output=output,
        current_target_id=args.target_id, station=ReadOnlyStation())
    state.trace = (output / "trace.jsonl").open("w", buffering=1)
    state.commands = (output / "commands.jsonl").open("w", buffering=1)
    try:
        if setup_continuous(state):
            run_targets(state)
    except KeyboardInterrupt:
        state.reason = "operator_interrupt"
    except BaseException as exc:
        state.failure = f"{type(exc).__name__}: {exc}"
        state.reason = "error"
        print(state.failure, file=sys.stderr, flush=True)
        (output / "error_traceback.txt").write_text(traceback.format_exc())
    finally:
        finalize_continuous(state)
    return 1 if state.failure else 0


def setup_continuous(state):
    """Connect devices, verify the start state and confirm the operator."""
    args, output = state.args, state.output
    if args.target_preview:
        from .joint_target_preview import LiveTargetPreview
        state.preview = LiveTargetPreview()
        print(f"目标预测预览（浏览器打开）：{state.preview.url}", flush=True)
        write(output / "preview.json", dict(url=state.preview.url, read_only=True))
    if args.track_target_motion:
        import cv2
        from ..perception.joint_target_motion import TargetMotionReferences
        from ..mask_codec import decode_mask
        initial_rgb = cv2.cvtColor(cv2.imread(str(args.target_snapshot / "global_rgb.png")), cv2.COLOR_BGR2RGB)
        state.motion_tracker = TargetMotionReferences(initial_rgb,
            {t["target_id"]: decode_mask(t["mask_png"]) for t in state.target_seed["visible_targets"]})
    if state.target_mode:
        if state.sequence:
            print(f'目标顺序：{" → ".join(state.sequence)}；每次释放后自动归位再抓下一根，最后一根结束后停止。', flush=True)
        else:
            print(f"本轮目标：{args.target_id}；一次夹紧→释放后结束。请保持圈选时的竹条摆放和全局相机位置。", flush=True)
        if args.paired_target_separation:
            print("本轮启用双目标前景分离试验；仅使用当前分割像素，无所选目标像素时停止。", flush=True)
        if args.multi_target_separation:
            print("本轮启用多实例可见种子分离；目标为空时停伺服，target-stack每根最多3次新观测重试，其他错误停止。", flush=True)
        if args.track_target_motion:
            print("实验性目标运动跟随：RGB运动更新参考，再核对当前分割；身份或前景支持不足时停止。", flush=True)
        if args.transport_age_aligned:
            print("本轮搬运按观察年龄取步；保留首次释放边界，到达目标后才松爪。", flush=True)
    print("加载模型、连接相机；尚未发送运动。", flush=True)
    state.pipe = InferencePipe(output, task=state.task, **state.config["pipe_options"])
    state.station.connect()
    state.motion = state.station.iface.getMotionControl()
    if state.motion.isServoModeEnabled():
        raise RuntimeError("existing servo owner")
    from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
    from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
    state.suction = AuboI10Robot(AuboI10Config())
    state.suction.io_control = state.station.io_readback
    try:
        initial_state, tcp, _ = state.station.current()
    except ValueError as exc:
        if str(exc) != "unknown suction output pair: (False, False)":
            raise
        print("DO2/DO3 均为低电平，当前吸盘指令状态未知。", flush=True)
        if input("确认吸盘未持物。按回车仅发送释放指令并核对输出，输入 q 退出：").strip():
            state.reason = "cancelled"
            return False
        try:
            if not state.suction.suction_release():
                raise RuntimeError("initial suction release failed")
        finally:
            log(state.trace, {"initial_suction_release": state.suction.last_gripper_command_trace})
        initial_state, tcp, _ = state.station.current()
    state.initial_state_known = True
    if initial_state[-1] != 0 or max(abs(a - b) for a, b in zip(initial_state[:6], START_DEG)) > 1.5:
        raise ValueError("请先归位并关闭吸盘，再启动")
    state.servo = SingleStepSession(state.motion)
    if math.dist(forward_kinematics(state, initial_state[:6]), tcp) > .001:
        raise ValueError("FK/current TCP mismatch")
    print(f"记录：{output.resolve()}\n持续伺服；Ctrl+C停止，停止不自动释放吸盘。", flush=True)
    if input("确认工作区无人、操作者在场且急停在手边。按回车开始，输入q退出：").strip():
        state.reason = "cancelled"
        return False
    state.recorder = TrialVideoRecorder(state.station.cameras, output)
    state.recorder.start()
    print(f"双相机录像已启动：{output.resolve()}/videos/", flush=True)
    state.pool = ThreadPoolExecutor(max_workers=1)
    return True


def infer(state, index, observed_state, images, metadata=None):
    """One worker request; the pipe is read at call time so a target switch is safe."""
    extra = {}
    if state.motion_tracker is not None:
        if state.sequence:
            offset = state.sequence.index(metadata["target_id"])
            tracked = state.sequence[offset + (metadata["phase"] == "transport"):]
        else:
            tracked = [t["target_id"] for t in state.target_seed["visible_targets"]
                       if metadata["phase"] == "approach" or t["target_id"] != metadata["target_id"]]
        # Preserve remaining-target motion across transport and new inference
        # sessions; never track a carried strip as a scene seed.
        motion = state.motion_tracker.update(images["global_rgb"], tracked,
            frame_id=metadata["frame_id"], required=False)
        log(state.trace, {"target_motion": {"index": index, "target_input": metadata, "reference": motion}})
        if metadata["phase"] == "approach":
            extra["target_motion"] = motion
    result = (state.pipe.predict(index, observed_state, images, codec="rgb_zlib", target_input=metadata, **extra)
              if state.target_mode else state.pipe.predict(index, observed_state, images))
    if index % 5 == 0:
        import cv2
        for name, img in images.items():
            if not cv2.imwrite(str(state.output / f"{index:04d}_{name}.jpg"),
                               cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("snapshot save failed")
    return result


def run_targets(state):
    """Single-target loop; sequence switching is delegated to switch_target."""
    config = state.config
    state.deadline = time.perf_counter() + config["max_seconds"]
    state.next_tick = time.perf_counter()
    state.last_tick = state.next_tick
    state.prediction_deadline = None
    state.reason = "time_limit"
    while time.perf_counter() < state.deadline:
        now = time.perf_counter()
        if now < state.next_tick:
            time.sleep(state.next_tick - now)
        now = time.perf_counter()
        if state.owned and now - state.last_tick > config["max_control_gap"]:
            raise TimeoutError(f'servo loop delayed over {config["max_control_gap"] * 1000:.0f} ms')
        state.last_tick = now
        state.next_tick = now + DT
        state.current_state, state.current_tcp, state.current_stamp = state.station.current(require_stationary=False)
        if state.current_state[-1] != state.held_suction:
            raise ValueError("external suction state change")
        workspace(state.current_tcp)
        future_ready = state.future is not None and state.future.done()
        if future_ready:
            try:
                result = state.future.result()
            except TargetForegroundMissing as exc:
                state.future = None
                if not handle_target_missing(state, exc):
                    raise
                continue
        if future_ready and state.discard_pending_result:
            state.future = None
            state.discard_pending_result = False
            log(state.trace, {"index": state.inflight_index, "discarded_prediction": result,
                "reason": "observation_precedes_suction_change",
                "observed_state": state.observed_state, "sensor_times": state.sensor_times})
        if state.future is not None and future_ready:
            state.future = None
            accept_prediction(state, result)
        if state.goal is not None:
            outcome = command_goal(state)
            if outcome == "done":
                break
            if outcome == "switch":
                continue
        if state.future is None and (not state.pending_suction
                                     or (config["approach"] and state.held_suction == 0)):
            request_observation(state)
    if state.sent == 0:
        state.reason = "no_motion"


def handle_target_missing(state, exc):
    """Stop, save evidence and decide whether a fresh observation may recover."""
    state.goal = None
    state.pending_suction = False
    state.trajectory = None
    state.last_observation = None
    state.active_chunk = None
    state.active_times = None
    state.selection = None
    state.prediction_id = None
    state.active_transport = False
    state.prediction_deadline = None
    state.last_servo_sent_at = None
    state.discard_pending_result = False
    try:
        if state.owned:
            state.servo._require_zero(state.motion.setServoMode(False), "disable servo on empty target")
            state.servo._wait_servo(False, timeout_s=1.)
            state.owned = False
        wait_until_stationary(lambda: state.station.current(require_stationary=False), state.station.state.isSteady)
    finally:
        evidence_path = save_target_failure(state.output, state.inflight_index, state.observed_state,
            state.images, state.sensor_times, state.metadata, exc)
        log(state.trace, {"target_foreground_missing": {"index": state.inflight_index,
            "target_id": state.current_target_id, "evidence": evidence_path,
            "held_suction": state.held_suction}})
        if state.preview is not None:
            state.preview.publish(images=state.images, evidence=exc.evidence, index=state.inflight_index,
                sensor_times=state.sensor_times, status="target missing; robot stopped")
    state.target_misses += 1
    if (state.args.model != "target-stack" or state.held_suction != 0
            or state.target_session.accepted_close_time is not None
            or state.target_misses > MAX_TARGET_REOBSERVATIONS):
        return False
    state.recovery_after = time.perf_counter()
    state.last_tick = state.recovery_after
    state.next_tick = state.last_tick + DT
    log(state.trace, {"target_reobserve": {"attempt": state.target_misses,
        "target_id": state.current_target_id, "stopped_at": state.recovery_after,
        "held_suction": state.held_suction}})
    return True


def accept_prediction(state, result):
    """Validate one received prediction and apply its action-selection rules."""
    config = state.config
    previous_pending_target = state.goal.copy() if config["approach"] and state.pending_suction else None
    log(state.trace, {"index": state.inflight_index, "task": state.task,
        "observed_state": state.observed_state, "sensor_times": state.sensor_times,
        "prediction": result, "received_at": time.perf_counter()})
    candidate = result["action"]
    if config["approach"]:
        # InferencePipe already checked and normalized the full chunk.
        state.active_chunk = result["action_chunk"]
        state.active_times = list(state.sensor_times)
        state.active_transport = (state.args.transport_age_aligned
                                  and result.get("target_input", {}).get("phase") == "transport")
        state.selection = None
        if state.held_suction == 0:
            candidate, state.selection = select_approach_action(
                state.active_chunk, state.active_times, time.perf_counter())
        elif state.active_transport:
            candidate, state.selection = select_transport_action(
                state.active_chunk, state.active_times, time.perf_counter())
        state.prediction_id = state.inflight_index
    accepted = validate_prediction(candidate, state.current_state, state.current_tcp,
        forward_kinematics(state, candidate[:6]), state.station.lower, state.station.upper,
        state.sensor_times, time.perf_counter(),
        max_prediction_age=config["max_prediction_age"], max_camera_skew=config["max_camera_skew"])
    state.predictions += 1
    state.goal = accepted
    state.last_observation = min(state.sensor_times)
    if state.preview is not None:
        state.preview.publish(images=state.images, evidence=result["target_evidence"],
            index=state.inflight_index, sensor_times=state.sensor_times)
    if previous_pending_target is not None:
        log(state.trace, {"pending_suction_replanned": {"previous_target": previous_pending_target,
            "updated_target": state.goal, "prediction_index": state.prediction_id}})
    state.prediction_deadline = state.last_observation + config["max_without_prediction"]
    state.pending_suction = state.goal[-1] != state.held_suction
    if not state.owned:
        state.trajectory = SmoothTrajectory(state.current_state[:6])
        state.owned = True
        state.servo._require_zero(state.motion.setServoMode(True), "enable servo")
        state.servo._wait_servo(True)
        if state.recovery_after is not None:
            log(state.trace, {"target_recovered": {"index": state.inflight_index,
                "target_id": state.current_target_id, "state": state.current_state,
                "sensor_times": state.sensor_times,
                "trajectory_velocity": state.trajectory.v.copy()}})
            state.recovery_after = None


def command_goal(state):
    """Track the accepted goal; return done/switch or None to keep observing."""
    config = state.config
    if not state.pending_suction and time.perf_counter() > state.prediction_deadline:
        raise TimeoutError(f'no fresh prediction within {config["max_without_prediction"] * 1000:.0f} ms')
    if state.pending_suction and time.perf_counter() - state.last_observation > 1.5:
        raise TimeoutError("suction transition target not reached within 1.5 seconds")
    if (config["approach"] and state.active_chunk is not None
            and (state.held_suction == 0 or state.active_transport) and not state.pending_suction):
        select_action = select_approach_action if state.held_suction == 0 else select_transport_action
        candidate, new_selection = select_action(state.active_chunk, state.active_times, time.perf_counter())
        if new_selection["selected_index"] != state.selection["selected_index"]:
            state.goal = validate_prediction(candidate, state.current_state, state.current_tcp,
                forward_kinematics(state, candidate[:6]), state.station.lower, state.station.upper,
                state.active_times, time.perf_counter(),
                max_prediction_age=config["max_prediction_age"],
                max_camera_skew=config["max_camera_skew"])
        state.selection = new_selection
        state.pending_suction = state.goal[-1] != state.held_suction
    # Hold this coherent IO-transition row while awaiting arrival, but let an
    # in-flight fresh observation replan BEFORE IO changes.
    q, target_tcp = state.trajectory.advance(state.goal[:6], lambda joint: forward_kinematics(state, joint))
    require_joint_target(q, state.current_state[:6], state.station.lower, state.station.upper,
                         MAX_TRACKING_ERROR_DEG)
    state_age = time.perf_counter() - state.current_stamp
    if state_age > config["max_state_age"]:
        if not config["mixed"]:
            raise TimeoutError("current state expired")
        state.current_state, state.current_tcp, state.current_stamp = state.station.current(require_stationary=False)
        if state.current_state[-1] != state.held_suction:
            raise ValueError("external suction state change")
        workspace(state.current_tcp)
        require_joint_target(q, state.current_state[:6], state.station.lower, state.station.upper,
                             MAX_TRACKING_ERROR_DEG)
        if time.perf_counter() - state.current_stamp > config["max_state_age"]:
            raise TimeoutError("current state expired after refresh")
        log(state.trace, {"state_refresh_before_servo": {"previous_age_s": state_age}})
    command_started_at = time.perf_counter()
    if (config["mixed"] and state.last_servo_sent_at is not None
            and command_started_at - state.last_servo_sent_at > config["max_control_gap"]):
        raise TimeoutError(f'servo command gap exceeded {config["max_control_gap"] * 1000:.0f} ms')
    state.servo._require_zero(state.motion.servoJoint([math.radians(x) for x in q], math.radians(MAX_ACCEL),
        math.radians(MAX_SPEED), config["servo_command_time"], 0., 200.), "servoJoint")
    state.last_servo_sent_at = command_started_at
    state.sent += 1
    command_record = {"time": time.perf_counter(), "state": state.current_state, "command": q,
        "velocity": state.trajectory.v, "target": state.goal, "tcp": state.current_tcp,
        "command_tcp": target_tcp}
    if state.sequence:
        command_record.update(target_id=state.current_target_id, pick_index=state.completed_cycles)
    if config["approach"]:
        command_record.update(prediction_index=state.prediction_id, action_selection=state.selection,
            suction_target_frozen=state.pending_suction)
    log(state.commands, command_record)
    if (state.pending_suction
            and max(abs(a - b) for a, b in zip(state.goal[:6], state.current_state[:6])) < .15
            and max(abs(v) for v in state.trajectory.v) < .3):
        # Goal is held while switching IO; no averaging of discrete suction.
        state.suction.is_suction_on = state.held_suction == 100
        try:
            state.suction._control_suction_based_on_gripper(state.goal[-1])
        finally:
            log(state.trace, {"suction": state.suction.last_gripper_command_trace})
        if state.target_mode:
            accepted_at = time.perf_counter()
            state.target_session.accept_command(dict(zip(JOINT_FIELDS, state.goal, strict=True)),
                accepted_monotonic_s=accepted_at)
            log(state.trace, {"target_phase_command": {"action": state.goal,
                "accepted_monotonic_s": accepted_at,
                "first_accepted_close_s": state.target_session.accepted_close_time,
                "target_id": state.current_target_id}})
            if state.preview is not None:
                state.preview.publish(target_id=state.current_target_id, phase="transport",
                    status="gripper command accepted; waiting for next observation", jpeg=None)
        previous_suction = state.held_suction
        state.held_suction = state.goal[-1]
        state.pending_suction = False
        if config["approach"]:
            state.discard_pending_result = state.future is not None
            state.active_chunk = None
            state.active_times = None
            state.selection = None
            state.active_transport = False
        if previous_suction == 100 and state.held_suction == 0:
            state.completed_cycles += 1
            if state.sequence:
                log(state.trace, {"target_cycle_completed": {"target_id": state.current_target_id,
                    "completed_cycles": state.completed_cycles, "physical_grasp_success": None}})
            if state.completed_cycles == config["expected_cycles"]:
                state.cycle = True
                state.reason = ("model_suction_cycle_completed" if config["expected_cycles"] == 1
                                else "model_suction_cycles_completed")
                return "done"
            if state.sequence:
                switch_target(state)
                return "switch"
        # IO readback can take 175 ms; do not catch up missed ticks.
        state.last_tick = time.perf_counter()
        state.next_tick = state.last_tick + DT
        # A separate deadline allows a post-IO observation; original sensor
        # timestamps remain unchanged in prediction records.
        state.prediction_deadline = state.last_tick + config["max_without_prediction"]
        # IO changed after the cycle's read: the next observation must contain
        # the new suction state, not the old snapshot.
        state.current_state, state.current_tcp, state.current_stamp = state.station.current(require_stationary=False)
    if state.sent % 25 == 0:
        print(f"已发送 {state.sent} 帧，预测 {state.predictions} 次，吸盘 {state.held_suction:.0f}", flush=True)
    return None


def switch_target(state):
    """End the old servo/worker, home, and start the next target session."""
    state.servo._require_zero(state.motion.setServoMode(False), "disable servo between targets")
    state.servo._wait_servo(False, timeout_s=1.)
    state.owned = False
    wait_until_stationary(lambda: state.station.current(require_stationary=False), state.station.state.isSteady)
    if state.future is not None:
        discarded = state.future.result()
        state.future = None
        log(state.trace, {"index": state.inflight_index, "discarded_prediction": discarded,
            "reason": "previous_target_completed", "target_id": state.current_target_id})
    state.pipe.close()
    state.pipe = None
    print(f"{state.current_target_id} 指令周期完成；自动归位，随后抓 {state.sequence[state.completed_cycles]}。", flush=True)
    receipt = home_to_start(state.motion, lambda: state.station.current(require_stationary=False)[0],
        lower=state.station.lower, upper=state.station.upper, is_steady=state.station.state.isSteady,
        progress=lambda event: log(state.trace, {"between_target_home": event}))
    log(state.trace, {"between_target_home_complete": receipt})
    state.current_target_id = state.sequence[state.completed_cycles]
    if state.preview is not None:
        state.preview.publish(target_id=state.current_target_id, phase="approach",
            status="loading next target; waiting for prediction", jpeg=None)
    state.target_seed, next_options = state.target_setups[state.completed_cycles]
    state.target_options = next_options
    state.target_session = TargetInputSession(state.current_target_id)
    worker_output = state.output / f"target_{state.completed_cycles + 1:02d}"
    worker_output.mkdir()
    state.pipe = InferencePipe(worker_output, task=state.task,
        **{**next_options, "request_timeout_s": state.config["inference_timeout_s"]})
    state.goal = None
    state.pending_suction = False
    state.last_observation = None
    state.trajectory = None
    state.active_chunk = None
    state.active_times = None
    state.selection = None
    state.prediction_id = None
    state.active_transport = False
    state.discard_pending_result = False
    state.inflight_index = None
    state.prediction_deadline = None
    state.last_servo_sent_at = None
    state.target_misses = 0
    state.recovery_after = None
    state.next_tick = time.perf_counter()
    state.last_tick = state.next_tick
    state.deadline = state.next_tick + state.config["max_seconds"]
    log(state.trace, {"target_started": {"target_id": state.current_target_id,
        "remaining_targets": state.sequence[state.completed_cycles:], "accepted_close_time": None}})


def request_observation(state):
    """Capture a fresh observation and enqueue one prediction."""
    config = state.config
    state.observed_state, state.images, state.sensor_times = state.station.observe(
        require_stationary=False,
        state_snapshot=(state.current_state, state.current_tcp, state.current_stamp),
        max_camera_age_ms=config["max_camera_age_ms"])
    if time.perf_counter() - min(state.sensor_times) > config["max_camera_age_ms"] / 1000:
        raise ValueError("stale input at capture")
    if state.recovery_after is not None and min(state.sensor_times) <= state.recovery_after:
        if time.perf_counter() - state.recovery_after > config["max_without_prediction"]:
            raise TimeoutError("no post-stop observation for target recovery")
        return
    state.inflight_index = state.request_index
    state.request_index += 1
    if state.target_mode:
        state.metadata = dict(target_id=state.current_target_id,
            frame_id=f'{state.target_seed["snapshot_id"]}/live/{state.inflight_index}',
            phase="transport" if is_transport(state.sensor_times[0], state.target_session.accepted_close_time)
            else "approach")
        state.future = state.pool.submit(partial(infer, state), state.inflight_index,
            state.observed_state, state.images, state.metadata)
    else:
        state.metadata = None
        state.future = state.pool.submit(partial(infer, state), state.inflight_index,
            state.observed_state, state.images)


def finalize_continuous(state):
    """Stop servo, close recorder/pool/preview/devices and write the summary."""
    if state.owned:
        try:
            state.servo._require_zero(state.servo.motion.setServoMode(False), "disable servo")
            state.servo._wait_servo(False, timeout_s=1.)
        except BaseException as exc:
            state.failure = f'{state.failure or ""}; stop: {exc}'
    if state.initial_state_known and getattr(state.station, "iface", None) is not None:
        try:
            q, tcp, stamp = wait_until_stationary(
                lambda: state.station.current(require_stationary=False), state.station.state.isSteady)
            write(state.output / "final_state.json", {"state": q, "tcp": tcp,
                "is_steady": state.station.state.isSteady(),
                "servo_enabled": state.station.iface.getMotionControl().isServoModeEnabled()})
        except BaseException as exc:
            state.failure = f'{state.failure or ""}; final state: {exc}'
    if state.recorder is not None:
        try:
            state.video_result = state.recorder.close()
            if not state.video_result["complete"]:
                print(f'录像不完整：{state.video_result["errors"]}', file=sys.stderr, flush=True)
        except BaseException as exc:
            state.video_result = {"complete": False, "errors": [str(exc)]}
            print(f"录像收尾失败：{exc}", file=sys.stderr, flush=True)
    if state.pool is not None:
        state.pool.shutdown(wait=True, cancel_futures=True)
    if state.preview is not None:
        state.preview.close()
    for close in (state.station.close, state.pipe.close if state.pipe else lambda: None):
        try:
            close()
        except BaseException as exc:
            state.failure = f'{state.failure or ""}; cleanup: {exc}'
    state.trace.close()
    state.commands.close()
    write(state.output / "complete.json", {"reason": state.reason, "sent_frames": state.sent,
        "predictions": state.predictions, "model": state.args.model, "target_id": state.args.target_id,
        **({"target_sequence": state.sequence, "current_target_id": state.current_target_id,
            "completed_target_ids": state.sequence[:state.completed_cycles]} if state.sequence else {}),
        "approach_age_aligned": state.config["approach"],
        "transport_age_aligned": state.args.transport_age_aligned,
        "requested_suction_cycles": state.config["expected_cycles"],
        "completed_suction_cycles": state.completed_cycles,
        "failure": state.failure, "suction_cycle_commanded": state.cycle,
        "physical_grasp_success": None, "video": state.video_result})
    print(f"结束：{state.reason}，记录：{state.output.resolve()}", flush=True)
