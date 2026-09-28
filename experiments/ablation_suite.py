"""Matched no-text 2x2 ablations and one separate sampling experiment, 22 classes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import ROOT, dataset_spec
from .models import ABLATION_FLAGS
from .prepare_semantic import DEST, RUNS as REFERENCE_RUNS
from .train import atomic_json

RUNS = ROOT / 'runs/semantic22_ablation'
JOBS = [
    ('rgb', 'ours_ablation_rgb', False, False, 0.0),
    ('detail', 'ours_ablation_detail', True, False, 0.0),
    ('boundary', 'ours_ablation_boundary', False, True, 0.0),
    ('classaware', 'ours_no_text', True, True, 0.5),
]
REFERENCE = REFERENCE_RUNS / 'ours_no_text_seed42'


def rare_ids_from_train(audit, threshold=0.01):
    counts = audit['prepared_class_pixel_counts']['train']
    total = sum(counts)
    if total <= 0:
        raise ValueError('No valid training pixels')
    return [i for i, n in enumerate(counts) if 0 < n / total < threshold]


def plan():
    audit = json.loads((REFERENCE_RUNS / 'dataset_audit.json').read_text(encoding='utf-8'))
    spec = dataset_spec(DEST)
    ids = rare_ids_from_train(audit)
    return {'seed': 42, 'max_epochs': 50, 'horizon': 150,
            'data_root': str(DEST), 'manifest_sha256': spec['manifest_sha256'],
            'reference': str(REFERENCE), 'jobs': [list(j) for j in JOBS],
            'rare_threshold_train_valid_pixels': 0.01, 'rare_class_ids': ids,
            'rare_class_names': [spec['classes'][i] for i in ids],
            'sampling': 'Per image: probability 0.5 choose a present rare class uniformly, then a pixel and a crop containing it; otherwise uniform crop. Same 320 images/epoch.',
            'controls': 'All no text; common seeded initial weights match full DGFF-Net. Same loss, optimizer, image split and evaluation. Full reference reused; no other methods rerun.',
            'limitations': 'Single seed, validation-selected checkpoints; optimization motivated by this validation set; no independent test set.'}


def command_for(job, study):
    key, model, _, _, probability = job
    folder = RUNS / f'{key}_seed42'
    command = [sys.executable, '-B', '-u', '-m', 'experiments.train', '--model', model,
               '--data-root', str(DEST), '--output', str(folder), '--seed', '42',
               '--epochs', '50', '--max-epochs', '50', '--horizon', '150',
               '--checkpoint-every', '5', '--workers', '2', '--patience', '50']
    if probability:
        command += ['--class-aware-crop-prob', str(probability), '--rare-class-ids',
                    *map(str, study['rare_class_ids'])]
    if (folder / 'config.json').exists():
        command.append('--resume')
    return command


def report():
    RUNS.mkdir(parents=True, exist_ok=True)
    classes = dataset_spec(DEST)['classes']
    baseline = json.loads((REFERENCE / 'evaluation.json').read_text())
    lines = ['# DGFF-Net 22类消融与采样优化', '',
             '同一320/80划分、960×736图像、512裁剪、512/384滑窗验证、种子42、50轮上限、150轮调度尺度、无TTA。各模型从ImageNet骨干初始化，公共模块初始权重与完整模型在相同种子下相同。全部不含文本分支。', '',
             '完整模型复用此前50轮结果。前三项为2×2结构消融的缺失组合；最后一项仅改变训练裁剪策略。全量22类mIoU、Macro F1和有效像素PixAcc，不删除难分类别。', '',
             '| 实验 | 细节门控 | 边界监督 | 稀有类采样概率 | 状态 | 已完成轮数 | 最佳轮数 | mIoU (%) | F1 (%) | PixAcc (%) | ΔmIoU vs 完整模型 (pp) |',
             '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    records = []
    entries = [('full_reference', 'ours_no_text', True, True, 0.0), *JOBS]
    for key, model, detail, boundary, probability in entries:
        folder = REFERENCE if key == 'full_reference' else RUNS / f'{key}_seed42'
        state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        row = {'experiment': key, 'detail': detail, 'boundary': boundary,
               'crop_probability': probability, 'state': state.get('state', 'pending'),
               'epoch': state.get('epoch', 0)}
        prefix = f"| {key} | {detail} | {boundary} | {probability} | {row['state']} | {row['epoch']} |"
        if row['state'] == 'stage_complete' and (folder / 'evaluation.json').exists():
            result = json.loads((folder / 'evaluation.json').read_text())
            assert result['num_classes'] == 22
            row.update({'best_epoch': result['epoch'], 'miou': result['miou'], 'f1': result['f1'],
                        'pixel_accuracy': result['pixel_accuracy'], 'delta_miou_pp': 100 * (result['miou'] - baseline['miou'])})
            row.update(dict(zip(classes, result['class_iou'])))
            lines.append(prefix + f" {result['epoch']} | {100*result['miou']:.3f} | {100*result['f1']:.3f} | {100*result['pixel_accuracy']:.3f} | {row['delta_miou_pp']:+.3f} |")
        else:
            lines.append(prefix + ' — | — | — | — | — |')
        records.append(row)
    lines += ['', '采样策略：只用训练集有效像素频率选取占比低于1%的类别。50%的概率尝试包含稀有类的裁剪，图中不存在候选类时保持均匀裁剪。每图每轮仍仅一个裁剪，不增加训练步数。验证集完全不使用此采样。', '',
              '消融解读：detail−rgb衡量无边界监督时的细节分支作用；boundary−rgb衡量无细节时的边界监督作用；full−boundary和full−detail衡量另一模块存在时的增益。该消融尚不能单独证明门控优于直接相加，后续需相加替代门控实验。', '',
              '限制：单种子；最佳模型按同一验证集选择；采样优化由此前验证结果启发，不能作为独立测试集结论。微小差异不代表统计显著。', '',
              '停止：在本目录创建STOP_QUEUE，并在当前运行的子目录创建STOP，可在当前轮结束后保存并停止。续跑需先确认进程已退出，再归档这些停止标记，重新运行同一队列。每5轮保存完整恢复检查点，最佳权重仅改善时保存。']
    (RUNS / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (RUNS / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['experiment', 'detail', 'boundary', 'crop_probability',
            'state', 'epoch', 'best_epoch', 'miou', 'f1', 'pixel_accuracy', 'delta_miou_pp', *classes])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    study = plan()
    path = RUNS / 'plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != study:
        raise ValueError('Study plan changed; use a new experiment directory')
    if not path.exists():
        atomic_json(path, study)
    report()
    if args.report_only:
        return
    if args.smoke_only:
        from .smoke_semantic import main as smoke
        smoke([*ABLATION_FLAGS, 'ours_no_text'], RUNS / 'smoke_results.json',
              {'class_aware_crop_prob': 0.5, 'rare_class_ids': study['rare_class_ids']})
        return
    frozen = RUNS / 'prior_result_hashes.json'
    if not frozen.exists():
        paths = [*(ROOT / 'runs').glob('*_seed42/evaluation*.json'),
                 *REFERENCE_RUNS.glob('*_seed42/evaluation*.json')]
        atomic_json(frozen, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    saved = json.loads(frozen.read_text())
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in saved.items()):
        raise ValueError('A prior result changed since this study began')
    lock = ROOT / 'runs/queue.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        for job in JOBS:
            key = job[0]
            folder = RUNS / f'{key}_seed42'
            if (RUNS / 'STOP_QUEUE').exists() or (folder / 'STOP').exists():
                atomic_json(RUNS / 'queue_status.json', {'state': 'stopped', 'experiment': key})
                return
            state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
            if state.get('state') == 'stage_complete' and state.get('epoch', 0) >= 50:
                continue
            command = command_for(job, study)
            with (RUNS / f'{key}_seed42.console.log').open('a', encoding='utf-8') as log:
                log.write('\nCOMMAND ' + subprocess.list2cmdline(command) + '\n')
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(RUNS / 'queue_status.json', {'state': 'running', 'experiment': key,
                    'model': job[1], 'pid': os.getpid(), 'child_pid': child.pid, 'target_epochs': 50})
                print('Training', key, 'PID', child.pid, flush=True)
                code = child.wait()
            report()
            state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
            if code or state.get('state') != 'stage_complete' or (RUNS / 'STOP_QUEUE').exists():
                atomic_json(RUNS / 'queue_status.json', {'state': 'failed' if code else 'stopped',
                            'experiment': key, 'exit_code': code})
                return
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in saved.items())
        atomic_json(RUNS / 'queue_status.json', {'state': 'complete'})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
