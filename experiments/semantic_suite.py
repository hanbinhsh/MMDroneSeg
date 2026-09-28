"""Isolated four-method fine-label experiment; existing five-class runs are read-only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import ROOT
from .prepare_semantic import DEST, RUNS
from .train import atomic_json

MODELS=['ours_no_text','afenet','logcan','d2ls']
LABELS={'ours_no_text':'DGFF-Net','afenet':'AFENet','logcan':'LOGCAN++','d2ls':'D2LS'}


def report():
    audit=json.loads((RUNS/'dataset_audit.json').read_text(encoding='utf-8'))
    classes=audit['spec']['classes']
    lines=['# 原始细粒度标签实验（22类）','',
           '使用本地原始标签ID 1–22，0(unlabeled)与23(conflicting)作为ignore=255。沿用原始320/80图像划分及现有960×736 RGB；标签从6000×4000以最近邻插值缩放。不是原始分辨率评测，也未合并语义类别。','',
           '四个方法均从各自原有骨干预训练权重开始，分类头重新训练；种子42、512裁剪、批量2/累积至8、最多50轮、沿用各自优化器和150轮调度时间尺度、普通512/384滑窗验证、无TTA。此协议只固定数据与训练预算，不假设不同预训练来源和模型计算量相同。','',
           'mIoU与F1均在全部22个类别上等权平均；若整个验证集中某类union为0，则其IoU记0（沿用既有实现）。PixAcc只统计有效标签像素。每5轮保存完整恢复检查点，最佳权重在改善时保存。所有结果都是验证集结果，没有独立测试集。','',
           '| Method | State | Epoch | Best epoch | PixAcc | mIoU | Macro F1 |','|---|---|---:|---:|---:|---:|---:|']
    rows=[]
    for name in MODELS:
        folder=RUNS/f'{name}_seed42'
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        path=folder/'evaluation.json'
        if path.exists() and state.get('state')=='stage_complete':
            r=json.loads(path.read_text());assert r['num_classes']==22
            rows.append(r)
            lines.append(f"| {LABELS[name]} | complete | {state['epoch']} | {r['epoch']} | {r['pixel_accuracy']:.5f} | {r['miou']:.5f} | {r['f1']:.5f} |")
        else:
            lines.append(f"| {LABELS[name]} | {state.get('state','pending')} | {state.get('epoch',0)} | — | — | — | — |")
    lines+=['','## 类别IoU','', '| Method | '+' | '.join(classes)+' |','|---|'+'---:|'*22]
    for r in rows:lines.append('| '+LABELS[r['model']]+' | '+' | '.join(f'{v:.5f}' for v in r['class_iou'])+' |')
    lines+=['','类别缺失情况：'+json.dumps(audit['missing_classes'],ensure_ascii=False),
            '', '五类实验保存在上一级runs目录，未覆盖；五类与22类的mIoU任务定义不同，不应直接用差值判断模型退化。']
    (RUNS/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (RUNS/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','best_epoch','pixel_accuracy','miou','macro_f1',*classes])
        for r in rows:w.writerow([LABELS[r['model']],r['epoch'],r['pixel_accuracy'],r['miou'],r['f1'],*r['class_iou']])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();report()
    if args.report_only:return
    frozen=RUNS/'five_class_result_hashes.json'
    if not frozen.exists():
        atomic_json(frozen,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'runs').glob('*_seed42/evaluation*.json')})
    lock=ROOT/'runs/queue.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    failures=[]
    try:
        for name in MODELS:
            if (RUNS/'STOP_QUEUE').exists():
                atomic_json(RUNS/'queue_status.json',{'state':'stopped','failures':failures});return
            folder=RUNS/f'{name}_seed42'
            path=folder/'status.json';state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('epoch',0)>=50 and state.get('state')=='stage_complete':continue
            if (folder/'STOP').exists():continue
            command=[sys.executable,'-B','-u','-m','experiments.train','--model',name,'--data-root',str(DEST),
                     '--output',str(folder),'--seed','42','--epochs','50','--max-epochs','50',
                     '--horizon','150','--checkpoint-every','5','--workers','2','--patience','50']
            if (folder/'config.json').exists():command.append('--resume')
            with (RUNS/f'{name}_seed42.console.log').open('a',encoding='utf-8') as log:
                log.write('\nCOMMAND '+subprocess.list2cmdline(command)+'\n');log.flush()
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(RUNS/'queue_status.json',{'state':'running','model':name,'target_epochs':50,'pid':os.getpid(),'child_pid':child.pid,'failures':failures})
                print('Training',name,'PID',child.pid,flush=True)
                code=child.wait()
            if code:
                failures.append({'model':name,'exit_code':code});print('FAILED',name,code,flush=True)
            report()
        saved=json.loads(frozen.read_text())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in saved.items())
        atomic_json(RUNS/'queue_status.json',{'state':'complete_with_failures' if failures else 'complete','failures':failures})
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
