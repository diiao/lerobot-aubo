"""Audit saved validation masks and build a standalone review; no model/device loading."""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from lerobot.bamboo_sorting.strip_segmentation_data import CLASSES, child, read_json, read_mask


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_map(path, segments, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(mask is not None and mask.shape == shape and mask.dtype == np.uint16, f"invalid map: {path}")
    ids = [s["mask_id"] for s in segments]
    require(len(ids) == len(set(ids)) and all(i > 0 for i in ids), f"invalid IDs: {path}")
    require(set(np.unique(mask)) - {0} == set(ids), f"map/segments mismatch: {path}")
    require(all(s["class_id"] in (0, 1) for s in segments), f"unknown class: {path}")
    return mask


def assignment(quality, threshold):
    # Maximize valid match count, then total IoU, as in the training evaluator.
    weight = (quality >= threshold) * (min(quality.shape) + 1) + quality
    rows, cols = linear_sum_assignment(-weight)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if quality[i, j] >= threshold]


def components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels, [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]


def encoded(image):
    ok, buffer = cv2.imencode(".png", image)
    require(ok, "PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buffer).decode()


def overlay(mask, segments, display_ids, outline=False):
    colors = [(90, 220, 60), (40, 165, 255), (230, 90, 210), (240, 210, 60)]  # BGR
    out = np.zeros((*mask.shape, 4), np.uint8)
    for segment in segments:
        identity = display_ids[segment["mask_id"]]
        area = (mask == segment["mask_id"]).astype(np.uint8)
        draw = area - cv2.erode(area, np.ones((3, 3), np.uint8)) if outline else area
        out[draw > 0] = [*colors[(identity - 1) % len(colors)], 255 if outline else 100]
        ys, xs = np.nonzero(area)
        # Anchor within a visible component instead of the occluded centroid.
        label = f"{identity}:{'T' if segment['class_id'] == 0 else 'C'}"
        point = (int(xs[len(xs) // 3]), int(ys[len(ys) // 3]))
        cv2.putText(out, label, point, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0, 255), 3)
        cv2.putText(out, label, point, cv2.FONT_HERSHEY_SIMPLEX, .4, (*colors[(identity - 1) % len(colors)], 255), 1)
    return encoded(out)


def build_review(dataset, result, review_root, split="validation"):
    data = read_json(dataset / "dataset.json")
    report_name = "best_validation.json" if split == "validation" else "test_evaluation.json"
    mask_dir = result / f"{split}_masks"
    report = read_json(result / report_name)
    complete = read_json(result / "complete.json")
    require(complete["best_reloaded_and_evaluated"], "best reload not confirmed")
    require(data["classes"] == CLASSES, "class mapping changed")
    samples = {s["id"]: s for s in data["samples"] if s["split"] == split}
    predictions = {s["id"]: s for s in report["samples"]}
    require(bool(samples) and len(samples) == len([s for s in data["samples"] if s["split"] == split]), "empty or duplicate GT")
    require(len(predictions) == len(report["samples"]) and samples.keys() == predictions.keys(), "sample IDs differ")
    require(report["images"] == len(samples), "image count differs")
    groups = {s["placement_id"] for s in samples.values()}
    require(not groups & {s["placement_id"] for s in data["samples"] if s["split"] != split}, "group leakage")
    require({p.name for p in mask_dir.glob("*.png")} ==
            {p["instance_map"] for p in predictions.values()}, "prediction file inventory differs")
    thresholds = (.5, .75, .9)
    counts = {str(t): [{"tp": 0, "fp": 0, "fn": 0} for _ in CLASSES] for t in thresholds}
    records, page_samples, pairs_all = [], [], []
    empty_count = empty_fp = 0
    for name, sample in sorted(samples.items()):
        pred = predictions[name]
        require(pred["placement_id"] == sample["placement_id"], f"group mismatch: {name}")
        shape = (sample["height"], sample["width"])
        gt = read_map(child(dataset, sample["instance_map"]), sample["segments"], shape)
        pm = read_map(child(mask_dir, pred["instance_map"]), pred["segments"], shape)
        require(pred["true_instances"] == len(sample["segments"]) and
                pred["predicted_instances"] == len(pred["segments"]), f"count mismatch: {name}")
        require(all(s["score"] >= report["score_threshold"] for s in pred["segments"]), f"score mismatch: {name}")
        directory = child(review_root, sample["placement_id"])
        review = read_json(directory / "review.json")
        frame = review["frames"][sample["frame_index"]]
        sequence_path = Path(review["source_sequence"])
        sequence = read_json(sequence_path)
        raw = child(sequence_path.parent, sequence["frames"][sample["frame_index"]]["image"])
        require(review["split"] == sequence["split"] == sample["split"] and
                review["placement_id"] == sequence["placement_id"] == sample["placement_id"], f"lineage: {name}")
        rgb_bytes = child(dataset, sample["image"]).read_bytes()
        require(rgb_bytes == raw.read_bytes() == child(directory, frame["image"]).read_bytes(), f"RGB differs: {name}")
        image = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.shape == (*shape, 3), f"RGB dimensions: {name}")
        objects = {}
        if sample["segments"]:
            labels = read_json(child(directory, frame["labels"]))
            require(labels["mask_review_complete"] and labels["placement_order_confirmed"], f"unreviewed: {name}")
            objects = {obj["id"]: obj for obj in labels["instances"]}
            for s in sample["segments"]:
                obj = objects[s["strip_id"]]
                require(np.array_equal(gt == s["mask_id"], read_mask(child(directory, obj["visible_mask"]), shape)),
                        f"GT differs from reviewed visible mask: {name}")
                require(s["class_id"] == (0 if s["strip_id"] in labels["top_ids_candidate"] else 1), f"GT class: {name}")
        else:
            require(sample["empty_baseline"] and frame["role"] == "empty_baseline", f"empty lineage: {name}")
            empty_count += 1
            empty_fp += int(bool(pred["segments"]))
        truth_masks = [gt == s["mask_id"] for s in sample["segments"]]
        pred_masks = [pm == s["mask_id"] for s in pred["segments"]]
        quality = np.zeros((len(truth_masks), len(pred_masks)))
        for i, a in enumerate(truth_masks):
            for j, b in enumerate(pred_masks):
                quality[i, j] = np.count_nonzero(a & b) / np.count_nonzero(a | b)
        class_quality = quality.copy()
        for i, s in enumerate(sample["segments"]):
            for j, p in enumerate(pred["segments"]):
                if s["class_id"] != p["class_id"]:
                    class_quality[i, j] = 0
        for threshold in thresholds:
            matches = assignment(class_quality, threshold)
            for c in range(2):
                tp = sum(sample["segments"][i]["class_id"] == c for i, _ in matches)
                local = {"tp": tp, "fp": sum(s["class_id"] == c for s in pred["segments"]) - tp,
                         "fn": sum(s["class_id"] == c for s in sample["segments"]) - tp}
                if threshold == .5:
                    require(local == pred["counts"][c], f"saved metrics disagree: {name}")
                for k, v in local.items():
                    counts[str(threshold)][c][k] += v
        # Class-agnostic correspondence exposes class errors rather than hiding them.
        matches = assignment(quality, .5)
        details, display = [], {}
        error = np.zeros((*shape, 4), np.uint8)
        for i, j in matches:
            s, p = sample["segments"][i], pred["segments"][j]
            a, b = truth_masks[i], pred_masks[j]
            display[p["mask_id"]] = s["mask_id"]
            component_map, gt_areas = components(a)
            _, pred_areas = components(b)
            footprint = read_mask(child(directory, objects[s["strip_id"]]["footprint_in_image"]), shape)
            hidden = footprint & ~a
            interior = cv2.distanceTransform(hidden.astype(np.uint8), cv2.DIST_L2, 5) > 2
            entry = {"sample_id": name, "strip_id": s["strip_id"], "gt_mask_id": s["mask_id"],
                     "pred_mask_id": p["mask_id"], "class_id": s["class_id"], "pred_class_id": p["class_id"],
                     "score": p["score"], "iou": float(quality[i, j]), "gt_area": int(a.sum()),
                     "extra_pixels": int((b & ~a).sum()), "missed_pixels": int((a & ~b).sum()),
                     "other_instance_pixels": int((b & (gt > 0) & ~a).sum()),
                     "gt_component_areas": gt_areas, "pred_component_areas": sorted(pred_areas, reverse=True),
                     "component_recall": [float((b & (component_map == k)).sum() / area)
                                          for k, area in enumerate(gt_areas, 1)],
                     "hidden_pixels": int(hidden.sum()), "predicted_hidden_pixels": int((b & hidden).sum()),
                     "hidden_interior_pixels": int(interior.sum()),
                     "predicted_hidden_interior_pixels": int((b & interior).sum())}
            details.append(entry)
            pairs_all.append(entry)
            error[a & ~b] = [255, 200, 0, 255]  # cyan: missed
            error[b & ~a] = [200, 0, 255, 255]  # magenta: extra
        for j, p in enumerate(pred["segments"]):
            if p["mask_id"] not in display:
                display[p["mask_id"]] = len(sample["segments"]) + j + 1
                error[pred_masks[j]] = [200, 0, 255, 255]
        for i, s in enumerate(sample["segments"]):
            if i not in {a for a, _ in matches}:
                error[truth_masks[i]] = [255, 200, 0, 255]
        record = {"id": name, "placement_id": sample["placement_id"], "frame_index": sample["frame_index"],
                  "gt_count": len(truth_masks), "pred_count": len(pred_masks), "pairs": details,
                  "unmatched_gt": [s["mask_id"] for i, s in enumerate(sample["segments"]) if i not in {a for a, _ in matches}],
                  "unmatched_pred": [s["mask_id"] for j, s in enumerate(pred["segments"]) if j not in {b for _, b in matches}],
                  "unmatched_prediction_details": [
                      {**p, "display_id": display[p["mask_id"]], "area": int(pred_masks[j].sum()),
                       "component_areas": sorted(components(pred_masks[j])[1], reverse=True)}
                      for j, p in enumerate(pred["segments"]) if j not in {b for _, b in matches}]}
        records.append(record)
        ids = {s["mask_id"]: s["mask_id"] for s in sample["segments"]}
        page_samples.append({**record, "raw": "data:image/png;base64," + base64.b64encode(rgb_bytes).decode(),
                             "gt": overlay(gt, sample["segments"], ids), "pred": overlay(pm, pred["segments"], display),
                             "gt_outline": overlay(gt, sample["segments"], ids, True),
                             "pred_outline": overlay(pm, pred["segments"], display, True), "error": encoded(error)})
    for t, values in counts.items():
        for v in values:
            v["f1"] = 2 * v["tp"] / max(2 * v["tp"] + v["fp"] + v["fn"], 1)
    require(empty_count == report["empty_images"] and empty_fp == report["empty_false_positive_images"], "empty metrics differ")
    summary = {"source_report": report_name, "split": split, "best_epoch": complete["best_epoch"],
               "score_threshold": report["score_threshold"], "images": len(samples), "groups": sorted(groups),
               "lineage_passed": True, "empty_images": empty_count, "empty_false_positive_images": empty_fp,
               "matching": {t: {c["name"]: v for c, v in zip(CLASSES, values, strict=True)} for t, values in counts.items()},
               "class_errors_at_class_agnostic_iou_0_5": sum(p["class_id"] != p["pred_class_id"] for p in pairs_all),
               "matched_instances": len(pairs_all),
               "true_instances": sum(s["gt_count"] for s in records),
               "unmatched_true_instances": sum(len(s["unmatched_gt"]) for s in records),
               "unmatched_predicted_instances": sum(len(s["unmatched_pred"]) for s in records),
               "iou_definition": "Geometry-only matched pairs grouped by true class; includes wrong-class matches, excludes unmatched predictions.",
               "iou": {c["name"]: {"mean": float(np.mean(values)), "median": float(np.median(values)), "min": float(min(values))}
                       for c in CLASSES if (values := [p["iou"] for p in pairs_all if p["class_id"] == c["id"]])},
               "worst_instances": sorted(pairs_all, key=lambda p: p["iou"])[:10], "samples": records,
               "scope": f"Saved {split} predictions; review does not run inference. IoU summaries use matched pairs only; inspect unmatched counts. Not physical grasp performance.",
               "component_definition": "8-connected; counts include tiny regions; per-component recall uses matched prediction ID",
               "hidden_interior_definition": "Reviewed footprint minus visible mask, distance to its boundary >2 pixels"}
    return summary, page_samples


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>竹条预测轮廓复核</title>
<style>body{font:16px system-ui;background:#131923;color:#e8edf4;margin:24px}select,button,label{font:inherit;margin:6px;padding:6px}a{color:#84c8ff}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}canvas{width:100%;background:#222}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #54606e}pre{white-space:pre-wrap}button{cursor:pointer}.note{color:#b6c9db}</style>
<h1>__TITLE__</h1><p>__CAPTION__</p>
<p>IoU（交并比）=预测与真值重叠面积÷两者合并面积。PNG像素ID是实例身份，类别来自JSON segments。</p>
<div id="summary"></div><select id="group"></select><select id="frame"></select><button id="previous">上一张</button><button id="next">下一张</button>
<label><input type="checkbox" id="crop" checked>放大料堆区域（原图坐标 x=280～640，y=200～480，仅显示裁切）</label>
<label><input type="checkbox" id="outline" checked>只画轮廓</label><button id="worst">最低IoU样本</button><button id="anomaly">类别／数量异常</button>
<h2 id="name"></h2><p class="note">同色／显示编号表示匹配到同一根；T=top_strip，C=covered_strip。预测原始ID见表，不代表跨帧追踪。青色=漏分，洋红=多分；误差图按实例比较，像素归错条也计入。</p>
<div class="views"><div>原始RGB<canvas id="raw"></canvas></div><div>人工真值<canvas id="gt"></canvas></div><div>模型预测<canvas id="pred"></canvas></div><div>轮廓差异<canvas id="error"></canvas></div></div>
<p id="counts"></p><table><thead><tr><th>真值→预测ID</th><th>类别 真值→预测</th><th>置信度／IoU</th><th>漏分／多分像素</th><th>连通块面积 真值→预测</th><th>各真值块召回率</th><th>遮挡区误占／面积（内部误占）</th></tr></thead><tbody id="rows"></tbody></table>
<pre id="unmatched"></pre><p class="note">连通块指互相连接的一片像素；同一实例可以有多块。遮挡内部排除了距离遮挡区边缘≤2像素的区域；该统计不替代目视检查。IoU均值只描述匹配轮廓的几何重合，包含类别判错的配对；额外预测另列，空图不计入均值。</p>
<script>const data=__DATA__;const summary=__SUMMARY__;const $=id=>document.getElementById(id);let current=0,renderToken=0;
$('summary').textContent='匹配实例平均IoU：top_strip '+(summary.iou.top_strip?.mean.toFixed(4)??'无匹配')+'，covered_strip '+(summary.iou.covered_strip?.mean.toFixed(4)??'无匹配')+'；匹配 '+summary.matched_instances+'/'+summary.true_instances+'，空图误检 '+summary.empty_false_positive_images+'/'+summary.empty_images+'。完整数值见同目录 contour_review.json。';
for(const g of summary.groups)$('group').add(new Option(g,g));for(let i=0;i<4;i++)$('frame').add(new Option(i===0?'空场景':'加入 '+i+' 根',i));
const load=src=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src;});
async function show(){const token=++renderToken,s=data[current];$('group').value=s.placement_id;$('frame').value=s.frame_index;$('name').textContent=s.id;
const suffix=$('outline').checked?'_outline':'';const images=await Promise.all([load(s.raw),load(s['gt'+suffix]),load(s['pred'+suffix]),load(s.error)]);if(token!==renderToken)return;
['raw','gt','pred','error'].forEach((id,i)=>{const c=$(id),ctx=c.getContext('2d'),r=$('crop').checked?[280,200,360,280]:[0,0,640,480];c.width=r[2]*2;c.height=r[3]*2;ctx.imageSmoothingEnabled=false;ctx.drawImage(images[0],...r,0,0,c.width,c.height);if(i)ctx.drawImage(images[i],...r,0,0,c.width,c.height);});
$('counts').textContent='真值 '+s.gt_count+' 个，预测 '+s.pred_count+' 个；未匹配真值 '+JSON.stringify(s.unmatched_gt)+'，未匹配预测 '+JSON.stringify(s.unmatched_pred);
$('unmatched').textContent=s.unmatched_prediction_details.map(p=>'额外预测：原始ID='+p.mask_id+'，显示编号='+p.display_id+'，类别='+(p.class_id===0?'top_strip':'covered_strip')+'，置信度='+p.score.toFixed(4)+'，面积='+p.area+'像素，连通块面积='+p.component_areas.join(',')).join('\n');
$('rows').replaceChildren();for(const p of s.pairs){const tr=document.createElement('tr');const cls=x=>x===0?'top_strip':'covered_strip';for(const value of [p.gt_mask_id+' → '+p.pred_mask_id,cls(p.class_id)+' → '+cls(p.pred_class_id),p.score.toFixed(3)+' / '+p.iou.toFixed(4),p.missed_pixels+' / '+p.extra_pixels,p.gt_component_areas.join(',')+' → '+p.pred_component_areas.join(','),p.component_recall.map(x=>(100*x).toFixed(1)+'%').join(', '),p.predicted_hidden_pixels+'/'+p.hidden_pixels+' ('+p.predicted_hidden_interior_pixels+')']){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('rows').append(tr);}location.hash=s.id;}
function select(){current=data.findIndex(s=>s.placement_id===$('group').value&&s.frame_index===Number($('frame').value));show();}
$('group').onchange=select;$('frame').onchange=select;$('crop').onchange=show;$('outline').onchange=show;$('previous').onclick=()=>{current=(current+data.length-1)%data.length;show();};$('next').onclick=()=>{current=(current+1)%data.length;show();};$('worst').onclick=()=>{current=data.findIndex(s=>s.id===summary.worst_instances[0].sample_id);show();};
const anomalies=data.map((s,i)=>s.unmatched_gt.length||s.unmatched_pred.length||s.pairs.some(p=>p.class_id!==p.pred_class_id)?i:-1).filter(i=>i>=0);
$('anomaly').disabled=!anomalies.length;$('anomaly').onclick=()=>{current=anomalies.find(i=>i>current)??anomalies[0];show();};
const linked=data.findIndex(s=>s.id===location.hash.slice(1));if(linked>=0)current=linked;show();</script></html>'''


def render_page(summary, samples):
    test = summary["split"] == "test"
    title = "独立测试预测轮廓复核" if test else "验证集预测轮廓复核"
    scope = "独立新摆放测试，未用于训练或选模型；不代表抓取成功率。" if test else "验证集参与选模型，不能当独立测试或抓取成功率。"
    caption = f"固定 best，第{summary['best_epoch']}轮；{summary['images']}图／{len(summary['groups'])}组。{scope}"
    page = HTML.replace("__TITLE__", title).replace("__CAPTION__", caption)
    page = page.replace("__DATA__", json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace("__SUMMARY__", json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False))
    if not summary["worst_instances"]:
        page = page.replace('<button id="worst">', '<button id="worst" disabled>')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/short_strip_segmentation"))
    parser.add_argument("--result", type=Path, default=Path("artifacts/short_strip_training/result"))
    parser.add_argument("--review-root", type=Path, default=Path("artifacts/placement_sequences/review"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    outputs = [args.result / "prediction_review.html", args.result / "contour_review.json"]
    require(not any(p.exists() for p in outputs), "review outputs already exist; preserve the existing review")
    summary, samples = build_review(args.dataset, args.result, args.review_root, split=args.split)
    page = render_page(summary, samples)
    outputs[0].write_text(page)
    outputs[1].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("samples", "worst_instances")}, indent=2, ensure_ascii=False))
    print(outputs[0].resolve())


if __name__ == "__main__":
    main()
