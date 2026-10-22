"""兼容入口：起点常量现统一维护在任务包内。

此文件不连接机器人；保留同名导出供既有脚本使用。
"""

from lerobot.bamboo_sorting.start_poses import (
    ACT90_RECOVERY_EXPECTED_TCP_XYZ_M,
    ACT90_RECOVERY_START_DEG,
    NORMAL_START_DEG,
    RECOVERY_EXPECTED_TCP_XYZ_M,
    RECOVERY_START_DEG,
)

__all__ = [
    'ACT90_RECOVERY_EXPECTED_TCP_XYZ_M', 'ACT90_RECOVERY_START_DEG',
    'NORMAL_START_DEG', 'RECOVERY_EXPECTED_TCP_XYZ_M', 'RECOVERY_START_DEG',
]
