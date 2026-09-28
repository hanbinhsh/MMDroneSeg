"""Compare archived results at one shared epoch budget without mixing later runs."""
import argparse
import csv
import json

from .data import ROOT, CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=30)
    args = parser.parse_args()
    root = ROOT / "runs"
    records = []
    for path in sorted(root.glob(f"*_seed42/stage_{args.stage}/evaluation.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("train_limit") or result.get("val_limit") or result.get("pretrained") is False:
            continue
        folder = path.parent.parent
        history = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        stage_history = [row for row in history if row["epoch"] <= args.stage]
        if len(stage_history) != args.stage or len({r["epoch"] for r in stage_history}) != args.stage:
            raise ValueError(f"Incomplete or duplicated stage history: {folder}")
        best = max(stage_history, key=lambda r: r["miou"])
        if result["epoch"] != best["epoch"] or abs(result["miou"] - best["miou"]) > 1e-7:
            raise ValueError(f"Archived evaluation does not match this stage: {folder}")
        records.append(result)
    if not records:
        raise RuntimeError("Archive each completed stage before generating its comparison")
    fields = ["model", "seed", "epoch", "miou", "f1", "pixel_accuracy", "parameters"] + CLASSES
    with (root / f"comparison_stage_{args.stage}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in records:
            row = {k: result.get(k, "") for k in fields}
            row.update(dict(zip(CLASSES, result["class_iou"])))
            writer.writerow(row)
    lines = [f"# 统一 {args.stage} 轮预算的阶段结果", "",
             "固定种子 42、原始 320/80 划分、512 裁剪训练及全分辨率滑窗验证。每个方法选取阶段内验证 mIoU 最佳检查点。", "",
             "这些是验证集结果，非独立测试结果；只运行一个种子，不能据小幅差异声称统计显著。各方法保留不同骨干、预训练数据和优化配置。", "",
             "| 方法 | 最佳轮次 | mIoU (%) | F1 (%) | 参数量 (M) |",
             "|---|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda x: -x["miou"]):
        lines.append(f"| {r['model']} | {r['epoch']} | {r['miou']*100:.3f} | {r['f1']*100:.3f} | {r['parameters']/1e6:.2f} |")
    lines += ["", "## 各类别 IoU (%)", "", "| 方法 | " + " | ".join(CLASSES) + " |",
              "|---|" + "---:|" * len(CLASSES)]
    for r in records:
        lines.append("| " + r["model"] + " | " + " | ".join(f"{value*100:.3f}" for value in r["class_iou"]) + " |")
    by_name = {r["model"]: r for r in records}
    if "ours" in by_name:
        ours = by_name["ours"]
        lines += ["", "## 当前可支持的结论", ""]
        for name in ["d2ls", "logcan", "afenet", "ours_rgb", "ours_no_detail", "ours_no_text", "ours_no_boundary"]:
            if name in by_name:
                diff = 100 * (ours["miou"] - by_name[name]["miou"])
                lines.append(f"- 完整模型相对 `{name}` 的 mIoU 差值为 {diff:+.3f} 个百分点。")
        weakest = min(range(len(CLASSES)), key=lambda i: ours["class_iou"][i])
        lines += ["", f"完整模型当前最弱类别是 `{CLASSES[weakest]}`，IoU 为 {ours['class_iou'][weakest]*100:.3f}%。",
                  "新增模块的阶段性贡献较小，不能把全部性能提升归因于文本或细节融合。应先比较相同最终预算的消融结果，再决定后续优化。",
                  "续跑上限为 50 轮并使用既定早停规则；运行中的后续结果不混入本表。历史 U-Net、DeepLabV3+ 的评估协议不同，另表保留。"]
    output = root / f"STAGE_{args.stage}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
