"""Report only our refinement study, keeping completed baselines unchanged."""
import json
from pathlib import Path

from .data import ROOT, CLASSES


def main():
    root = ROOT / "runs"
    names = ["ours", "ours_no_text", "ours_sched50", "ours_refine"]
    labels = {"ours": "原完整模型", "ours_no_text": "原无文本模型",
              "ours_sched50": "无文本 + 50轮学习率计划", "ours_refine": "原学习率计划 + 细节修正 + 难像素监督"}
    results, pending = {}, []
    for name in names:
        folder = root / f"{name}_seed42"
        if not (folder / "evaluation.json").exists():
            pending.append(name)
            continue
        state = json.loads((folder / "status.json").read_text())
        if state["epoch"] < 50 or state["state"] != "stage_complete":
            pending.append(name)
            continue
        result = json.loads((folder / "evaluation.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if config.get("train_limit") or config.get("val_limit"):
            raise ValueError("Limited-data smoke results must never enter this report")
        result["horizon"] = config["horizon"]
        results[name] = result
    lines = ["# 我们模型的第二轮改进实验", "",
             "仅新增我们模型的实验，其他对比方法结果不变。原始320/80划分、种子42、每个模型最多50轮、每5轮保存完整检查点；验证分辨率、滑窗、数据增强、骨干预训练均不变，无TTA。", "",
             "两组新实验都从同一ImageNet预训练骨干开始，没有使用旧分割检查点继续增加训练轮数。", "",
             "| 配置 | 调度器时间尺度 | 最佳轮次 | mIoU (%) | 参数量 (M) |",
             "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {labels[name]} | {r['horizon']} | {r['epoch']} | {r['miou']*100:.3f} | {r['parameters']/1e6:.3f} |")
    if pending:
        lines += ["", "尚未完成：" + "、".join(labels[n] for n in pending) + "。"]
    lines += ["", "## 各类别IoU (%)", "", "| 配置 | " + " | ".join(CLASSES) + " |",
              "|---|" + "---:|" * len(CLASSES)]
    for name, r in results.items():
        lines.append(f"| {labels[name]} | " + " | ".join(f"{v*100:.3f}" for v in r["class_iou"]) + " |")
    if len(results) == 4:
        control, candidate, previous = results["ours_sched50"], results["ours_refine"], results["ours_no_text"]
        lines += ["", "## 结果解释", "",
                  f"- 新学习率计划相对原无文本模型：{100*(control['miou']-previous['miou']):+.3f} 个百分点。",
                  f"- 细节修正和难像素监督组合相对原无文本模型（相同150轮调度时间尺度、实际均训练50轮）：{100*(candidate['miou']-previous['miou']):+.3f} 个百分点。",
                  "- 快速衰减对照表现不佳后，细节修正实验改为沿用原150轮调度时间尺度，停止轮次仍为50；本轮存在基于验证结果的自适应方案选择。",
                  "- 组合实验同时修改了修正头和训练损失，不能把其收益单独归因于修正头。",
                  "- 单种子、同一验证集上进一步调优，不代表独立测试集收益或统计显著性。其他对比方法没有重新调参。"]
        best = max(results, key=lambda n: results[n]["miou"])
        lines += ["", f"本轮所有已测自有版本中验证mIoU最高：**{labels[best]} ({100*results[best]['miou']:.3f}%)**。",
                  f"检查点：`{root / (best+'_seed42') / 'best.pt'}`。"]
    tta_path = root / "ours_no_text_seed42/evaluation_tta4.json"
    if tta_path.exists():
        tta = json.loads(tta_path.read_text())
        lines += ["", "## 单独的推理增强检查", "",
                  f"原无文本模型普通推理：{100*tta['ordinary_inference']['miou']:.3f}%；四向翻转logits平均：{100*tta['four_flip_tta']['miou']:.3f}%。",
                  "该结果需要约4倍网络前向次数，没有重新训练。它不混入上述无TTA表格，也不能作为结构改进或与无TTA基线公平比较的证据。"]
    full_path = root / "ours_no_text_seed42/evaluation_fullframe.json"
    if full_path.exists():
        full = json.loads(full_path.read_text())
        lines += ["", f"另测原无文本模型整图直接推理（无滑窗、无TTA）：{100*full['miou']:.3f}%。该协议同样单独列出，不替换普通滑窗评估。"]
    output = root / "OURS_IMPROVEMENT.md"
    output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
