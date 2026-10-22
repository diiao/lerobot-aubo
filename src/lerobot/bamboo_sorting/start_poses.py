"""AUBO i10 joint start poses shared by capture, inference and recovery entries.

This module never connects to the robot. Keeping one definition prevents the
executable scripts from drifting to different poses.
"""

# Current normal recording/inference start.
NORMAL_START_DEG = [-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]
START_DEG = tuple(NORMAL_START_DEG)

# run05 trial02 frame 472 closed-loop failure state, used for recovery demos.
RECOVERY_START_DEG = [-23.27, -5.91, 111.13, 28.57, 89.55, -162.21]
RECOVERY_EXPECTED_TCP_XYZ_M = [0.5749, -0.4706, 0.1716]

# 12D ACT live10 90-degree single-strip failure state. Median of the final
# stable hover segment, not one jittering last frame.
ACT90_RECOVERY_START_DEG = [
    -27.5262,
    -8.2700,
    103.3084,
    23.1320,
    89.6794,
    -164.6689,
]
ACT90_RECOVERY_EXPECTED_TCP_XYZ_M = [0.58369, -0.53438, 0.22060]
