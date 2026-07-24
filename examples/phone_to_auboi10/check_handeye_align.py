"""校验 handeye 残余偏差 (加强版): 三种方法交叉验证。
方法1: ORB + Lowe比值测试 + RANSAC 相似变换 (旋转+平移+缩放)
方法2: ECC 欧式变换 (旋转+平移, 无缩放) - 对小偏差更稳
方法3: 相位相关 (纯平移)
三方法一致 = 可信; 不一致 = 拟合不稳/场景差异大。
"""
import cv2, numpy as np, sys, os

REF = "/tmp/handeye_ref_ep0_00000.png"
LIVE = "/tmp/handeye_live_check.png"
OUT = "/tmp/handeye_align_overlay.png"

ref = cv2.imread(REF); live = cv2.imread(LIVE)
if ref is None or live is None:
    print(f"读取失败 ref={os.path.exists(REF)} live={os.path.exists(LIVE)}"); sys.exit(1)
g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
print(f"ref {ref.shape}  live {live.shape}")

# 方法1: ORB + 比值测试 + RANSAC
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(g1, None)
kp2, des2 = orb.detectAndCompute(g2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
raw = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in raw if m.distance < 0.75 * n.distance]
print(f"\n[方法1 ORB] 特征 ref={len(kp1)} live={len(kp2)} 比值测试通过={len(good)}")
M_best = None
if len(good) >= 8:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=2000, confidence=0.999)
    nin = int(inl.sum()) if inl is not None else 0
    pct = 100 * nin / len(good)
    if M is not None:
        a, b = M[0, 0], M[1, 0]
        scale = float(np.hypot(a, b)); rot = float(np.degrees(np.arctan2(b, a)))
        tx, ty = float(M[0, 2]), float(M[1, 2]); trans = float(np.hypot(tx, ty))
        print(f"  旋转={rot:+.3f}°  平移=({tx:+.2f},{ty:+.2f}) |t|={trans:.2f}px  缩放={scale:.4f}  内点={nin}/{len(good)}({pct:.0f}%)")
        M_best = M

# 方法2: ECC 欧式 (旋转+平移, 无缩放)
print(f"\n[方法2 ECC欧式]")
try:
    M_ecc = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-5)
    M_ecc = cv2.findTransformECC(g1, g2, M_ecc, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    a, b = M_ecc[0, 0], M_ecc[1, 0]
    rot_e = float(np.degrees(np.arctan2(b, a)))
    tx_e, ty_e = float(M_ecc[0, 2]), float(M_ecc[1, 2])
    print(f"  旋转={rot_e:+.3f}°  平移=({tx_e:+.2f},{ty_e:+.2f}) |t|={np.hypot(tx_e, ty_e):.2f}px")
except Exception as e:
    print(f"  失败: {e}")

# 方法3: 相位相关 (纯平移)
(sh, resp) = cv2.phaseCorrelate(g1.astype(np.float64), g2.astype(np.float64))
print(f"\n[方法3 相位相关] 平移=({sh[0]:+.2f},{sh[1]:+.2f})px  响应={resp:.3f}")

# 叠加图
if M_best is not None:
    warped = cv2.warpAffine(live, M_best, (ref.shape[1], ref.shape[0]))
    cv2.imwrite(OUT, cv2.addWeighted(ref, 0.5, warped, 0.5, 0))
    print(f"\n叠加图: {OUT}")
