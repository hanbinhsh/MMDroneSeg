"""Keep colleague results separate from reproducible local experiments."""
import csv
import json
from pathlib import Path
from .data import ROOT, CLASSES

HISTORICAL = [
    {"model": "U-Net", "miou": 0.75187, "f1": 0.83934, "pixel_accuracy": 0.93450,
     "class_iou": [0.5752, 0.7894, 0.9164, 0.5650, 0.9134]},
    {"model": "DeepLabV3+", "miou": 0.80538, "f1": 0.87844, "pixel_accuracy": 0.94513,
     "class_iou": [0.6509, 0.8300, 0.9204, 0.7014, 0.9243]},
    {"model": "MMDroneSeg (original)", "miou": 0.78238, "f1": 0.87010, "pixel_accuracy": 0.93250,
     "class_iou": [0.5830, 0.8736, 0.9174, 0.6297, 0.9082]},
]


def generate():
    root = ROOT / "runs"
    root.mkdir(exist_ok=True)
    historical = [{**r, "source": "provided course presentation", "protocol": "historical_other_computers_not_rerun"} for r in HISTORICAL]
    (root / "historical_results.json").write_text(json.dumps(historical, indent=2), encoding="utf-8")
    local = []
    for p in sorted(root.glob("*_seed42/evaluation.json")):
        result = json.loads(p.read_text(encoding="utf-8"))
        if result.get("pretrained") is False or result.get("train_limit") or result.get("val_limit"):
            continue
        config = json.loads((p.parent/"config.json").read_text(encoding="utf-8"))
        status = json.loads((p.parent/"status.json").read_text(encoding="utf-8"))
        if status["state"] == "running":
            result["result_state"] = "previous completed stage; continuation running"
        else:
            result["result_state"] = status["state"]
        local.append({**result, "directory": str(p.parent), "epochs_completed": status["epoch"]})
    fields = ["model", "seed", "epoch", "epochs_completed", "miou", "f1", "pixel_accuracy", "parameters", "protocol", "result_state"] + CLASSES
    with (root/"comparison_local.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in local:
            row = {k: r.get(k, "") for k in fields}
            row.update(dict(zip(CLASSES, r["class_iou"])))
            writer.writerow(row)
    lines = ["# 实验结果", "", "## 同学提供的历史结果（保留、不重跑）", "",
             "原始 PPT 指标；训练环境、骨干、预处理和评估分辨率未完整保存。不能与新协议结果直接归因比较。", "",
             "| 方法 | mIoU (%) | F1 (%) | PixAcc (%) |", "|---|---:|---:|---:|"]
    for r in historical:
        lines.append(f"| {r['model']} | {100*r['miou']:.3f} | {100*r['f1']:.3f} | {100*r['pixel_accuracy']:.3f} |")
    lines += ["", "## 本机统一协议，种子 42", "", "原始 320/80 划分；512 裁剪训练；512/384 滑窗；960×736 全图评估；无 TTA。", "",
              "仅有验证集，所有数字均为验证结果。单个种子不报告标准差。", "",
              "| 方法 | 最优轮次 | 已跑轮次 | mIoU (%) | 状态 |", "|---|---:|---:|---:|---|"]
    for r in local:
        lines.append(f"| {r['model']} | {r['epoch']} | {r['epochs_completed']} | {100*r['miou']:.3f} | {r['result_state']} |")
    if not local:
        lines += ["", "正式实验尚未完成，冒烟检查结果不计入论文表格。"]
    (root / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return local


if __name__ == "__main__":
    generate()
