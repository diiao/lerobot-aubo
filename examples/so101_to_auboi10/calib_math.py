"""标定与遥操共用的纯数学工具（无 lerobot 依赖，便于离线单测）。

包含:
  - 旋转矩阵 ↔ 四元数转换
  - R_align 四元数平均求解 (Markley)
  - 尺度 s、平移 t 线性最小二乘求解
  - ZYX 欧拉分解（提取 yaw/pitch，锁 roll；含 gimbal guard）
  - 旋转矩阵夹角
"""
import math

import numpy as np


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 (x, y, z, w)，Shepperd 方法，数值稳定。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 (x, y, z, w) → 旋转矩阵。"""
    qx, qy, qz, qw = q
    nrm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / nrm, qy / nrm, qz / nrm, qw / nrm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def solve_r_align(pairs) -> np.ndarray:
    """由姿态对应点对求 R_align：M_i = R_aubo_i @ R_leader_i.T，四元数平均 (Markley)。

    pairs: list of (p_leader, R_leader, p_aubo, R_aubo)
    """
    K = np.zeros((4, 4))
    first = True
    sign_ref = None
    for p_l, R_l, p_a, R_a in pairs:
        M = R_a @ R_l.T
        q = rotmat_to_quat(M)
        if first:
            sign_ref = 1.0 if q[3] >= 0 else -1.0
            first = False
        if (q[3] * sign_ref) < 0:
            q = -q
        K += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(K)  # 升序
    q_avg = eigvecs[:, -1]
    R_align = quat_to_rotmat(q_avg)
    # 强制正交且行列式 +1
    U, _, Vt = np.linalg.svd(R_align)
    R_align = U @ Vt
    if np.linalg.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = U @ Vt
    return R_align


def solve_scale_translation(pairs, R_align: np.ndarray):
    """R_align 已知下解 p_aubo = t + s * (R_align @ p_leader)，线性最小二乘。

    返回 (t (3,), s (float))。
    """
    A = np.zeros((3 * len(pairs), 4))
    b = np.zeros(3 * len(pairs))
    for i, (p_l, R_l, p_a, R_a) in enumerate(pairs):
        u = R_align @ p_l
        A[3 * i:3 * i + 3, 0:3] = np.eye(3)
        A[3 * i:3 * i + 3, 3] = u
        b[3 * i:3 * i + 3] = p_a
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    t = sol[:3]
    s = float(sol[3])
    return t, s


def zyx_extract(R: np.ndarray, last_yaw=None, last_pitch=None):
    """ZYX 内在欧拉分解 R = Rz(yaw)·Ry(pitch)·Rx(roll)。返回 (yaw, pitch)，roll 丢弃。
    奇异点 (pitch≈±90°) 冻结上一帧。

    矩阵展开:
      R[2,0] = -sin(pitch)
      R[1,0] = sin(yaw)cos(pitch),  R[0,0] = cos(yaw)cos(pitch)
    """
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    cp = math.sqrt(max(0.0, 1.0 - R[2, 0] ** 2))
    if cp > 1e-3:
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        yaw = last_yaw if last_yaw is not None else 0.0
        pitch = last_pitch if last_pitch is not None else 0.0
    return yaw, pitch


def zyx_extract_full(R: np.ndarray):
    """ZYX 分解，返回完整 (yaw, pitch, roll)。用于标定取 roll_const。"""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    cp = math.sqrt(max(0.0, 1.0 - R[2, 0] ** 2))
    if cp > 1e-3:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return yaw, pitch, roll


def Rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(pitch: float) -> np.ndarray:
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(roll: float) -> np.ndarray:
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """两旋转矩阵间夹角(弧度)：acos((tr(R1^T R2)-1)/2)。"""
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(tr)


# 理想方向对齐：两臂基座系朝向一致（X前/Y左/Z上 对齐），R_align = 单位阵。
# 列 = leader 各轴在 Aubo 系下的方向：leader +X(前)→Aubo +X，leader +Y(左)→Aubo +Y，+Z(上)→+Z。
# 若两臂物理摆放有 90° 转角（如一个朝前一个朝侧），改回原先的 90° 矩阵：
#   [[0,1,0],[-1,0,0],[0,0,1]] (绕 Z 转 -90°：leader前→Aubo -Y，leader左→Aubo +X)
IDEAL_R_ALIGN = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def solve_scale_translation_fixed_r(pairs, R_align: np.ndarray):
    """给定 R_align，从位置对应点线性最小二乘解 p_aubo = t + s*(R_align @ p_leader)。

    不依赖姿态对应(手工摆姿态误差大)，只用位置，方向由固定 R_align 保证。
    返回 (t (3,), s (float))。
    """
    n = len(pairs)
    A = np.zeros((3 * n, 4))
    b = np.zeros(3 * n)
    for i, (p_l, R_l, p_a, R_a) in enumerate(pairs):
        u = R_align @ p_l
        A[3 * i:3 * i + 3, 0:3] = np.eye(3)
        A[3 * i:3 * i + 3, 3] = u
        b[3 * i:3 * i + 3] = p_a
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3], float(sol[3])


def extract_target_rotvec(R_aligned: np.ndarray, R_neutral: np.ndarray,
                          last_yaw=None, last_pitch=None, roll_rad: float = 0.0) -> np.ndarray:
    """从 leader 末端姿态(已表达到 Aubo 基座系)提取 Aubo 目标姿态 rotvec。

    对"相对中性姿态的偏差"做 ZYX 分解，左右侧倾(Rx)固定为 roll_rad（不跟随 leader）：
      R_dev = R_aligned @ R_neutral^T
      R_dev = Rz(yaw)·Ry(pitch)·Rx(roll)   # roll 取固定值 roll_rad
      R_target = Rz(yaw)·Ry(pitch)·Rx(roll_rad)·R_neutral

    中性姿态(吸盘朝下)处偏差=I，分解良态：
      - Rz(yaw) = heading(绕竖直轴，吸盘朝下时即工具自转) — 控制
      - Ry(pitch) = 前后俯仰 — 控制
      - Rx(roll) = 左右侧倾 — 锁定

    Args:
        R_aligned: R_align @ R_leader，leader 末端姿态在 Aubo 基座系。
        R_neutral: Aubo 中性姿态(吸盘平贴)旋转矩阵。
        last_yaw/last_pitch: 奇异点冻结用。
    Returns:
        目标姿态 rotvec(弧度)。
    """
    R_dev = R_aligned @ R_neutral.T
    yaw, pitch = zyx_extract(R_dev, last_yaw, last_pitch)
    R_dev_target = Rz(yaw) @ Ry(pitch) @ Rx(roll_rad)
    R_target = R_dev_target @ R_neutral
    # 用仓库外约定：调用方负责转 rotvec，这里直接返回矩阵也行；为方便返回 rotvec
    # 用 Rodrigues 公式
    return rotmat_to_rotvec(R_target), yaw, pitch


def extract_target_rotmat(R_aligned: np.ndarray, R_neutral: np.ndarray,
                          last_yaw=None, last_pitch=None, roll_rad: float = 0.0):
    """同 extract_target_rotvec，但返回目标姿态矩阵(供姿态步进用)。

    Args:
        roll_rad: 左右侧倾(Rx)固定值(弧度)，不跟随 leader。默认 0。
    Returns:
        (R_target (3x3), yaw, pitch)
    """
    R_dev = R_aligned @ R_neutral.T
    yaw, pitch = zyx_extract(R_dev, last_yaw, last_pitch)
    R_target = (Rz(yaw) @ Ry(pitch) @ Rx(roll_rad)) @ R_neutral
    return R_target, yaw, pitch


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 旋转向量(弧度)，Rodrigues 公式。"""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # 接近 π：用对角线最大元素求轴
        # R + I = 2 * outer(k, k)
        M = (R + np.eye(3)) / 2.0
        # 每行是 k_i * k，取范数最大行
        idx = int(np.argmax(np.diag(M)))
        k = M[idx] / math.sqrt(max(M[idx, idx], 1e-12))
        return k * angle
    sin_angle = math.sin(angle)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * sin_angle)
    return k * angle


def rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    """旋转向量 → 旋转矩阵，Rodrigues 公式。"""
    ang = float(np.linalg.norm(rv))
    if ang < 1e-9:
        return np.eye(3)
    k = np.asarray(rv, dtype=float) / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def align_rotvec(rv: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 rotvec 调整到与参考 rotvec ref 数值最接近的 2π 等价分支。

    rotvec 有双重多值性：同一姿态可表示为 axis*angle 或 axis*(angle-2π)
    (反向轴+补角)。servoCartesian 按 rotvec 数值插值，若相邻指令跨越 2π 分支
    会误判为大跳变而拒绝执行。此函数保证输出数值连续贴近 ref。
    """
    rv = np.asarray(rv, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-9:
        return rv
    axis = rv / ang
    best = rv
    best_d = np.linalg.norm(rv - ref)
    # 尝试 ±2π 的等价表示(同轴加减 2π，以及反向轴补角)
    for k in (-1, 1):
        alt = axis * (ang + k * 2 * math.pi)
        d = np.linalg.norm(alt - ref)
        if d < best_d:
            best, best_d = alt, d
    return best


def step_toward_orientation(R_current: np.ndarray, R_target: np.ndarray,
                            max_step_rad: float, ref_rotvec=None) -> np.ndarray:
    """从当前姿态朝目标姿态转一小步(测地线)，返回步进后姿态的 rotvec。

    保证输出是"从当前 TCP 姿态出发的小增量"，且 rotvec 数值贴近 ref_rotvec，
    避免 servoCartesian 因 rotvec ±2π 分支跳变/大角度拒绝执行整条指令(位置也不动)。

    Args:
        R_current: Aubo 当前 TCP 姿态矩阵。
        R_target: 目标姿态矩阵。
        max_step_rad: 单帧最大旋转角(弧度)。
        ref_rotvec: 参考 rotvec(通常是 Aubo 当前 TCP 的 rotvec)，输出会对齐到它
            最近的 2π 分支，保证数值连续。默认用 R_current 的 rotvec。
    Returns:
        步进后姿态的 rotvec(数值连续贴近 ref_rotvec)。
    """
    if ref_rotvec is None:
        ref_rotvec = rotmat_to_rotvec(R_current)
    R_delta = R_target @ R_current.T          # 世界系下相对旋转
    rv_delta = rotmat_to_rotvec(R_delta)
    ang = float(np.linalg.norm(rv_delta))
    if ang > max_step_rad and ang > 1e-9:
        rv_delta = rv_delta * (max_step_rad / ang)
    R_next = rotvec_to_rotmat(rv_delta) @ R_current
    return align_rotvec(rotmat_to_rotvec(R_next), ref_rotvec)

