"""从明确的多边形草稿生成离线复核页与掩码，不连接任何设备。"""

import argparse
import json
from pathlib import Path

from lerobot.bamboo_sorting.placement_review import save_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing placement sequence directory")
    parser.add_argument("--polygons", type=Path, required=True, help="Polygon drafts or browser-exported JSON")
    parser.add_argument("--output", type=Path,
                        help="Review directory; defaults to SOURCE_PARENT/review/SOURCE_NAME; updates the same group")
    args = parser.parse_args(argv)
    annotations = json.loads(args.polygons.read_text())
    output = args.output or args.source.resolve().parent / "review" / args.source.resolve().name
    manifest = save_review(args.source, annotations, output)
    print(f"已保存 {output / 'review.html'}，{len(manifest['frames']) - 1} 张非空帧；标签未接入训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
