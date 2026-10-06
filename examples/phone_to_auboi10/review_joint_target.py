"""Preview frame-local target annotations from saved videos; no model or hardware access."""

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.bamboo_sorting.joint_target import TARGET_IMAGE_KEY, prepare_target_images, target_contract_record


def association_section(root):
    folder = root / "association"
    name = next((name for name in ("reconfirmation", "rigid_association", "association")
                 if (folder / f"{name}.json").exists()), "association")
    path = folder / f"{name}.json"
    if not path.exists():
        return ""
    report = json.loads(path.read_text())
    counts, lost = report["counts"], report["first_lost"]
    start, end = report["plan"]["start_frame"] / 25, report["plan"]["end_frame"] / 25
    loss_text = (f'首次持续失效在{lost["video_time_s"]:.2f}秒，该帧有{lost["predicted_instances"]}个分割候选。'
                 if lost else "没有触发持续失效，不代表所有帧都有目标输出或身份已获验证。")
    recovery = report["plan"].get("reconfirmation")
    rule_text = (f'空检测后最多保留{recovery["max_gap_frames"]}帧关联记忆；需要{recovery["required_confirmations"]}张不同来源图像连续满足更严格的几何条件，才标为reconfirmed。'
                 'unobserved是暂时未观测，confirming是待确认，这两种状态不显示绿色目标；歧义、明显不匹配或超时为lost，必须显式重新选择。'
                 if recovery else 'lost时不显示绿色目标，也不沿用旧轮廓；后续候选重新出现也不会自动恢复身份。')
    recovery_text = (f'重新确认{counts.get("reconfirmed",0)}帧，未观测{counts.get("unobserved",0)}帧，待确认{counts.get("confirming",0)}帧；片段最终状态为{report["final_status"]}。'
                     if recovery else "")
    events = "、".join(f'{r["video_time_s"]:.2f}秒' for r in report.get("reconfirmation_events", []))
    if events:
        recovery_text += f'本次在{events}恢复关联；这是几何证据支持的判断，仍非人工身份真值。'
    return f'''<h2 id="association">本次：{start:g}～{end:g}秒新分割＋目标关联</h2>
<p><strong>离线关联原型，尚未通过整段身份保持验证；不是机械臂输入。</strong>
左侧原图；右侧蓝色是当前帧分割候选，绿色是几何关联选中的候选，仍待人工复核。
{rule_text}</p>
<p>共{report["frames"]}帧：初始化{counts.get("seed",0)}帧，匹配候选{counts.get("matched",0)}帧，丢失{counts.get("lost",0)}帧。
{recovery_text}{loss_text}这些计数不是身份正确率。初始对象由助手从唯一预测中选定，尚待用户确认；本片段只有一根木条，未验证多目标竞争。</p>
<p>旋转对齐只用于比较形状，输出仍采用当前帧的新预测掩码。首次仅补偿平移的尝试拒绝了快速转动；补上旋转后复用了同一批预测，数值门限未变。
本结果是开发诊断，不是独立测试；缺失轮廓与身份丢失分别记录。</p>
<video id="association-video" controls preload="metadata" style="width:100%;max-width:1280px" src="association/{name}.mp4"></video>
<p>定位到原录像：
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=3.8">27.80秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=3.84">27.84秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=5.8">29.80秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=5.84">29.84秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=5.88">29.88秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=5.96">29.96秒</button>
<button onclick="const v=document.getElementById('association-video');v.pause();v.currentTime=6">30秒</button>
<button onclick="document.getElementById('association-video').playbackRate=0.25">四分之一速度</button>
<button onclick="document.getElementById('association-video').playbackRate=1">正常速度</button></p>
<details><summary>关联统计、固定条件与局限</summary><pre>{html.escape(json.dumps(report,ensure_ascii=False,indent=2))}</pre></details>'''


def segmentation_section(root, spec, encoded):
    """Display frozen predictions separately from reviewed target annotations."""
    folder = root / "segmentation"
    if not (folder / "result/predictions.json").exists():
        return ""
    report = json.loads((folder / "result/predictions.json").read_text())
    comparisons = json.loads((folder / "comparison.json").read_text())
    by_id = {s["id"]: s for s in comparisons["samples"]}
    reviewed = {r["video_frame_index"]: r for r in spec["frames"] if r["reviewed"]}
    views, table = [], []
    palette = [(0, 190, 255), (255, 165, 0), (50, 220, 60), (230, 60, 220)]
    for row in report["samples"]:
        bgr = cv2.imread(str(folder / row["image"]))
        ids = cv2.imread(str(folder / "result" / row["instance_map"]), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.shape != (480, 640, 3) or ids is None or ids.shape != (480, 640):
            raise ValueError("missing diagnostic image or prediction mask")
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        marked, truth = raw.copy(), raw.copy()
        legend = []
        for segment in row["segments"]:
            mask = ids == segment["mask_id"]
            color = palette[(segment["mask_id"] - 1) % len(palette)]
            marked[mask] = (marked[mask] * 0.8 + np.asarray(color) * 0.2).astype(np.uint8)
            boundary = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
            marked[boundary] = color
            name = {0: "top_strip", 1: "covered_strip"}[segment["class_id"]]
            legend.append(f'实例{segment["mask_id"]}：{name}，置信度{segment["score"]:.3f}，{segment["pixels"]}像素')
        gold = reviewed.get(row["video_frame_index"])
        if gold:
            for polygon in gold["visible_polygons"]:
                cv2.polylines(truth, [np.asarray(polygon, np.int32)], True, (255, 0, 255), 2)
        comparison = by_id[row["id"]]["comparison"]
        if bool(gold) != (comparison is not None):
            raise ValueError("comparison and reviewed frame coverage disagree")
        best = comparison["best"] if comparison else None
        status = (f'已确认轮廓：最佳重合实例IoU={best["iou"]:.3f}，可见像素召回={best["recall"]:.1%}'
                  if best else "已确认轮廓，模型无可见实例" if gold else "本帧尚未人工标注，不计算IoU；中列显示原图")
        if gold:
            table.append(f'<tr><td>{row["video_time_s"]:g}秒</td><td>{best["iou"] if best else 0:.3f}</td>'
                         f'<td>{best["recall"] if best else 0:.1%}</td></tr>')
        views.append({"label": f'{row["video_time_s"]:g}秒', "raw": encoded(raw), "truth": encoded(truth),
                      "prediction": encoded(marked), "status": status, "legend": "；".join(legend) or "无预测实例"})
    data = json.dumps(views, ensure_ascii=False).replace("<", "\\u003c")
    return f'''<h2 id="segmentation">冻结分割模型：夹持与搬运画面</h2>
<p>冻结best；原始640×480 RGB、置信度阈值0.5，无训练或阈值调整。共{len(views)}帧，每帧独立预测，尚无跨帧身份关联。
紫色是已确认轮廓；模型预测用彩色描边和浅色填充，类别、置信度及实例数见每帧图下说明。</p>
<p>IoU仅对{len(table)}个已确认关键帧计算；可见像素召回指人工轮廓中被预测覆盖的比例。
选取与人工轮廓最重合的实例仅用于分析，不代表模型自主选对目标。其余3帧没有真值，不计算成绩。
置信度不代表轮廓完整度；本批也不属于独立测试或实机成功率评价。</p>
<table><tr><th>已确认关键帧</th><th>IoU</th><th>可见像素召回</th></tr>''' + "".join(table) + '''</table>
<p><label>预测帧 <select id="seg-frame"></select></label></p><p id="seg-status"></p>
<section><article>原始全局图<img id="seg-raw"></article><article>人工轮廓（紫色）<img id="seg-truth"></article>
<article>模型预测<img id="seg-prediction"></article></section><p id="seg-legend"></p>
<script>const segSamples=''' + data + ''';const segSelect=document.getElementById('seg-frame');
segSamples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.label;segSelect.append(o)});
function showSeg(){const s=segSamples[Number(segSelect.value)];for(const k of ['raw','truth','prediction'])document.getElementById('seg-'+k).src=s[k];
document.getElementById('seg-status').textContent=s.status;document.getElementById('seg-legend').textContent=s.legend}
segSelect.onchange=showSeg;showSeg();</script>'''


def build_page(annotations_path, output, *, update=False):
    annotations_path, output = Path(annotations_path).resolve(), Path(output).resolve()
    if output.exists() and not update:
        raise FileExistsError(output)
    spec = json.loads(annotations_path.read_text())
    run = (annotations_path.parent / spec["source_run"]).resolve()
    videos = {k: cv2.VideoCapture(str(run / "videos" / f"{k}.mp4")) for k in ("global_rgb", "grasp_rgb")}
    samples = []

    def encoded(rgb):
        ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("preview encoding failed")
        return "data:image/png;base64," + base64.b64encode(data).decode()

    try:
        for row in spec["frames"]:
            index = row["video_frame_index"]
            if type(index) is not int or index < 0:
                raise ValueError("invalid video frame index")
            images = {}
            for name, cap in videos.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"missing video frame {name}:{index}")
                images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_id = f"{run.name}/global_rgb/{index}"
            mask = np.zeros((480, 640), np.uint8)
            for polygon in row["visible_polygons"]:
                vertices = np.asarray(polygon, dtype=float)
                if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
                        or not np.isfinite(vertices).all()
                        or np.any(vertices < 0) or np.any(vertices > [639, 479])):
                    raise ValueError("invalid visible polygon")
                cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 1)
            prepared = prepare_target_images(images, mask.astype(bool), row,
                                             frame_id=frame_id, target_id=spec["target_id"], allow_draft=True)
            samples.append({"label": row["label"], "frame_id": frame_id,
                            "preview_only": prepared["preview_only"], "annotation": row,
                            "raw": encoded(images["global_rgb"]),
                            "marked": encoded(prepared["images"][TARGET_IMAGE_KEY]),
                            "wrist": encoded(images["grasp_rgb"])})
    finally:
        for cap in videos.values():
            cap.release()
    if not samples:
        raise ValueError("no annotated frames")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>指定目标输入复核</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f6f8;color:#17202a}p{line-height:1.6}select{padding:8px;font-size:16px}section{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}article{background:white;padding:10px}pre{white-space:pre-wrap}strong{color:#a10068}@media(max-width:900px){section{grid-template-columns:1fr}}</style>
<h1>指定目标输入复核</h1>
<p><strong>跨帧目标关联原型尚未通过整段验证。逐帧分割实例ID不等于同一根木条的长期ID。</strong>
<a href="#association">查看本次24～30秒关联回放</a> · <a href="#segmentation">查看冻结模型的逐帧分割</a> · <a href="#failed-flow">查看已失败的旧光流案例</a></p>
__ASSOCIATION__
<p><strong>人工关键帧轮廓，仅用于接口复核；每帧确认状态见下方，不代表中间帧已标注或可用于训练。</strong>
紫色为目标可见区域内侧2像素描边。原图保留，腕部图保持原样；同一根沿用同一个target_id，轮廓随当前帧变化，夹爪遮挡部分不补全。</p>
<p>目标完全遮挡或身份丢失时，接口拒绝生成目标输入，不沿用上一帧轮廓。关键帧之间尚无人工确认标注。
当前旧模型不接收这套输入，训练适配、示教与目标服从性验证仍待完成。</p>
<label>阶段 <select id="frame"></select></label><p id="status"></p>
<section><article>原始全局RGB<img id="raw"></article><article>目标标记图 target_global_rgb<img id="marked"></article><article>原始腕部RGB<img id="wrist"></article></section>
<details><summary>当前帧与目标记录</summary><pre id="record"></pre></details>
__SEGMENTATION__
__TRACKING__
<script>const samples=__SAMPLES__;const select=document.getElementById('frame');
samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.label;select.append(o)});
function show(){const s=samples[Number(select.value)];for(const k of ['raw','marked','wrist'])document.getElementById(k).src=s[k];
document.getElementById('status').textContent=s.frame_id+' | '+(s.preview_only?'待人工复核':'已确认轮廓，仍不代表可执行');
document.getElementById('record').textContent=JSON.stringify(s.annotation,null,2)}select.onchange=show;show();</script></html>'''
    page = page.replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace("__ASSOCIATION__", association_section(annotations_path.parent))
    page = page.replace("__SEGMENTATION__", segmentation_section(annotations_path.parent, spec, encoded))
    tracking_path = annotations_path.parent / "tracking.json"
    tracking = ""
    if tracking_path.exists():
        report = json.loads(tracking_path.read_text())
        checkpoints = report["checkpoints_before_human_reset"]
        assessment = ("本次所有检查点IoU均为0，光流基线未通过，不能作为连续目标输入。"
                      if checkpoints and all(r["iou"] == 0 for r in checkpoints)
                      else "这是开发诊断，检查点重合不代表整段连续跟踪正确。")
        rows = "".join(f'<tr><td>{r["time_s"]:g}秒</td><td>{r["iou"]:.3f}</td>'
                       f'<td>{"有草稿" if r["track_available"] else "已丢失"}</td></tr>'
                       for r in report["checkpoints_before_human_reset"])
        tracking = f'''<details id="failed-flow"><summary>失败案例：旧光流回放（已否定用于连续目标输入，点击展开）</summary>
<h2>历史诊断：光流轮廓漂移</h2>
<p><strong>{assessment}</strong></p>
<p><strong>橙色是未复核的光流草稿，绿色是已确认关键帧。lost表示已停止传播。</strong>
光流是从相邻图像估计像素移动，不是分割模型。40、180、222秒在比较后使用人工轮廓重新初始化，不能当作自主连续跟踪。
即使显示draft，也可能漂移或漏掉重新露出的区域；本录像不作为训练标签。</p>
<video id="tracking-video" controls preload="metadata" style="max-width:960px;width:100%" src="tracking.mp4"></video>
<p>跳转：<button onclick="document.getElementById('tracking-video').currentTime=25">接近夹持 25秒</button>
<button onclick="document.getElementById('tracking-video').currentTime=40">40秒</button>
<button onclick="document.getElementById('tracking-video').currentTime=180">180秒</button>
<button onclick="document.getElementById('tracking-video').currentTime=222">222秒</button>
<button onclick="document.getElementById('tracking-video').playbackRate=2">2倍速</button>
<button onclick="document.getElementById('tracking-video').playbackRate=1">正常速度</button></p>
<p>IoU是传播轮廓与已确认轮廓的交并比，越接近1重合越好；下表均在人工重新初始化之前计算，丢失时按空掩码记0。</p>
<table><tr><th>检查帧</th><th>IoU</th><th>传播状态</th></tr>{rows}</table>
<details><summary>诊断统计与局限</summary><pre>{html.escape(json.dumps(report,ensure_ascii=False,indent=2))}</pre></details></details>'''
    page = page.replace("__TRACKING__", tracking)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    return {"page": str(output), "frames": len(samples), "draft_frames": sum(s["preview_only"] for s in samples),
            "contract": target_contract_record(), "training_or_inference_run": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", action="store_true", help="Explicitly regenerate the existing review page")
    args = parser.parse_args()
    print(json.dumps(build_page(args.annotations, args.output, update=args.update), ensure_ascii=False, indent=2))
