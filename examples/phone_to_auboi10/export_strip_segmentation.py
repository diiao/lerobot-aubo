"""导出已复核竹条的原始RGB、可见实例PNG及上层类别，不连接设备。"""

import argparse
import json
from pathlib import Path

from lerobot.bamboo_sorting.strip_segmentation_data import export_reviewed_piles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument("--test-only", action="store_true", help="Export independent test groups only")
    args = parser.parse_args()
    print(json.dumps(export_reviewed_piles(args.selection, args.output, test_only=args.test_only),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
