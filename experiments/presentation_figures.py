"""Create classroom figures from saved results only; no training or inference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "classroom_figures"
DATA = ROOT.parent / "Drone/classes_dataset/classes_dataset"
TEAL, BLUE, ORANGE, INK = "#087F8C", "#4275BD", "#DB8351", "#203247"
LABELS = {"afenet": "AFENet (2025)", "logcan": "LOGCAN++ (2025)",
          "d2ls": "D2LS (2025)", "ours_no_text": "本文模型（无文本）",
          "ours": "完整模型", "ours_rgb": "仅 RGB", "ours_no_detail": "去掉细节分支",
          "ours_no_boundary": "去掉边界监督", "ours_sched50": "50轮学习率计划",
          "ours_refine": "细节修正 + 难像素监督"}
CLASSES = ["障碍物", "水体", "软质表面", "移动物体", "着陆区域"]
COLORS = np.array([[155,38,182], [14,135,204], [124,252,0], [255,20,147], [169,169,169]])
MAIN = ["afenet", "logcan", "d2ls", "ours_no_text"]
PROTOCOL = "320/80 训练/验证划分 · seed=42 · 各50轮 · 512裁剪训练 · 960×736滑窗验证 · 无TTA"
SOURCES, PAGES = {}, []


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def canvas(title, subtitle, footer=PROTOCOL):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.055, .925, title, fontsize=26, weight="bold", color=INK)
    fig.text(.055, .867, subtitle, fontsize=14, color="#54657A")
    fig.text(.055, .045, footer, fontsize=11, color="#627185")
    return fig


def clean(ax, axis="x"):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis=axis, color="#E4EAF0", linewidth=.8)
    ax.set_axisbelow(True)


def save(fig, name, title, notes, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    pdf.savefig(fig)
    PAGES.append(dict(name=name, title=title, notes=notes))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 14,
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    results = {}
    for name in LABELS:
        folder = RUNS / f"{name}_seed42"
        r, config, status = [read(folder / file) for file in ("evaluation.json", "config.json", "status.json")]
        assert status["epoch"] == 50 and status["state"] == "stage_complete"
        assert r["seed"] == 42 and not config.get("train_limit") and not config.get("val_limit")
        assert r["pretrained"] and r["protocol"] == "native_960x736_crop512_sliding_no_tta"
        cm = np.array(r["confusion_matrix"], dtype=float)
        ious = cm.diagonal() / (cm.sum(0) + cm.sum(1) - cm.diagonal())
        assert np.allclose(ious, r["class_iou"]) and np.isclose(ious.mean(), r["miou"])
        results[name] = r
    score = lambda name: results[name]["miou"] * 100
    with PdfPages(OUT / "课堂展示图表合集.pdf") as pdf:
        fig = canvas("01  方法对比：本文模型与三种2025年方法", "统一数据与评估协议；各方法沿用各自的骨干、预训练来源与优化配置。")
        ax = fig.add_axes([.23,.24,.67,.54])
        values = [score(n) for n in MAIN]
        ax.barh(range(4), values, color=[BLUE]*3+[TEAL], height=.55)
        ax.set_yticks(range(4), [LABELS[n] for n in MAIN]); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel("验证集 mIoU（%）"); clean(ax)
        for i,v in enumerate(values): ax.text(v+1,i,f"{v:.3f}",va="center",weight="bold",color=INK)
        gap = score("ours_no_text") - score("d2ls")
        fig.text(.23,.135,f"本文模型：{score('ours_no_text'):.3f}%   |   相对 D2LS：+{gap:.3f} 个百分点",fontsize=18,color=TEAL,weight="bold")
        save(fig,"01_method_comparison","方法对比", "结果取50轮预算内最佳验证检查点。本文模型指ConvNeXt-Tiny+FPN、DoG/阈值细节分支和边界监督的无文本版本，并非旧图中的三流ResNet模型。AFENet采用SWSL预训练。不同方法参数量、预训练和优化器并不完全相同；单种子的小差距不能声称统计显著。",pdf)

        fig = canvas("02  收敛过程：验证集 mIoU 随训练轮次变化", "展示逐轮原始记录，不平滑；右侧放大后半程，观察小幅变化。")
        axs = [fig.add_axes([.075,.22,.49,.57]),fig.add_axes([.64,.22,.30,.57])]
        for n,c in zip(MAIN,["#A4B2C2",ORANGE,BLUE,TEAL]):
            path = RUNS/f"{n}_seed42/metrics.jsonl"
            SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['epoch'] for r in rows] == list(range(1,51))
            for ax in axs: ax.plot([r['epoch'] for r in rows],[r['miou']*100 for r in rows],color=c,label=LABELS[n],lw=2.2)
        for ax in axs: clean(ax, "y"); ax.set_xlabel("训练轮次")
        axs[0].set(xlim=(1,50),ylim=(0,100),ylabel="验证集 mIoU（%）")
        axs[1].set(xlim=(25,50),ylim=(80,91),title="后半程放大（纵轴80–91%）")
        axs[0].legend(loc="lower right",frameon=False,fontsize=12)
        save(fig,"02_validation_curves","训练收敛曲线","各线是每轮验证结果，不是最佳值累计曲线。50轮相同不等于计算开销相同。右图为局部放大，不能据此夸大性能差异。",pdf)

        fig = canvas("03  各类别表现：障碍物仍是主要短板", "每格为该类别 IoU（%）；mIoU 对五个类别等权平均。")
        ax = fig.add_axes([.23,.24,.65,.52])
        matrix = np.array([results[n]['class_iou'] for n in MAIN])*100
        im=ax.imshow(matrix,cmap="YlGnBu",vmin=60,vmax=100,aspect="auto")
        ax.set_xticks(range(5),CLASSES); ax.set_yticks(range(4),[LABELS[n] for n in MAIN]); ax.tick_params(length=0,pad=12)
        for i in range(4):
            for j in range(5): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=18,color="white" if matrix[i,j]>86 else INK)
        fig.colorbar(im,cax=fig.add_axes([.90,.24,.016,.52]),label="类别 IoU（%）")
        save(fig,"03_class_iou","类别IoU对比","本文模型障碍物IoU为75.83%，低于其他类别。软质表面等中文名用于展示，原始类别名为obstacles、water、soft-surfaces、moving-objects、landing-zones。颜色量程固定60–100%。",pdf)

        names=["ours_rgb","ours","ours_no_detail","ours_no_boundary","ours_no_text"]
        fig = canvas("04  模块消融：文本分支未带来可验证的增益", "同一骨干与训练设置；完整模型作为消融参照，当前选用无文本版本。")
        ax=fig.add_axes([.055,.23,.52,.53]); ax.axis("off")
        rows=[[LABELS[n],*flags,f"{score(n):.3f}"] for n,flags in zip(names,[("—","—","—"),("有","有","有"),("—","有","有"),("有","有","—"),("有","—","有")])]
        table=ax.table(cellText=rows,colLabels=["配置","细节","文本","边界","mIoU%"],bbox=[0,0,1,1],cellLoc="center",colWidths=[.38,.13,.13,.13,.23])
        table.auto_set_font_size(False); table.set_fontsize(13)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor("white");cell.set_facecolor("#DBE7ED" if i==0 else ("#E4F3F0" if i==5 else "#F3F6F9"))
        ax=fig.add_axes([.65,.23,.28,.53]); deltas=[score(n)-score("ours") for n in names]
        ax.axvline(0,color="#AAB7C2",lw=1)
        ax.scatter(deltas,range(5),s=95,color=[BLUE]*4+[TEAL]); ax.set_yticks(range(5),[""]*5); ax.invert_yaxis()
        for i,v in enumerate(deltas): ax.annotate(f"{v:+.3f}",(v,i),xytext=(9,0),textcoords="offset points",va="center",fontsize=13)
        # Reserve one header-height row so points align with table body rows.
        ax.set(xlim=(-.10,.18),ylim=(4.5,-1.5),xlabel="相对完整模型（百分点）");clean(ax)
        fig.text(.055,.13,"各版本差距较小；单种子消融仅支持趋势观察，不能证明模块收益稳定。",fontsize=15,color=INK)
        save(fig,"04_ablation","模块消融","仅RGB版本同时移除了细节、文本和边界模块，因此它与无文本版本的差异不能单独归因于细节分支。其余去掉单模块的配置均相对完整模型。点图横轴表示百分点差值。",pdf)

        tta=read(RUNS/"ours_no_text_seed42/evaluation_tta4.json")
        fig = canvas("05  改进尝试：结构改动未超过原版本，TTA有小幅收益", "左侧为重新训练的模型；右侧为同一检查点的推理增强，分开解读。", "均在同一80张验证集上评估 · 单种子 · TTA约需4倍网络前向次数 · 无独立测试集")
        ax=fig.add_axes([.25,.26,.25,.49])
        ns=["ours_no_text","ours_sched50","ours_refine"]
        ds=[score(n)-score("ours_no_text") for n in ns]
        ax.barh(range(3),ds,color=[TEAL,ORANGE,ORANGE],height=.5);ax.scatter([0],[0],color=TEAL,zorder=3)
        ax.set_yticks(range(3),["原无文本模型","缩短学习率计划","修正头 + 难像素"]);ax.invert_yaxis();ax.axvline(0,color="#AAB7C2")
        for i,v in enumerate(ds): ax.text(.03,i,f"{v:+.3f}",va="center",fontsize=13)
        ax.set(xlim=(-1.1,.3),xlabel="相对原无文本模型（百分点）",title="重新训练：实际均为50轮");clean(ax)
        ax=fig.add_axes([.64,.26,.28,.49]);vals=[score("ours_no_text"),tta['four_flip_tta']['miou']*100]
        ax.bar([0,1],vals,color=[BLUE,TEAL],width=.5);ax.set_xticks([0,1],["普通推理","四向翻转TTA"])
        for i,v in enumerate(vals):ax.text(i,v+2,f"{v:.3f}%",ha="center",fontsize=16,weight="bold")
        ax.set(ylim=(0,100),ylabel="验证集 mIoU（%）",title=f"仅推理增强：+{vals[1]-vals[0]:.3f} 个百分点");clean(ax,"y")
        save(fig,"05_improvement_and_tta","改进实验与推理增强","原无文本与修正头实验使用150轮学习率调度时间尺度、实际训练50轮；快速衰减实验的调度时间尺度为50。修正头方案是在快速衰减结果不佳后调整，存在验证集上的自适应选择。修正头同时改变结构和损失，无法隔离因果。TTA不属于结构收益，不能直接与无TTA基线宣称公平优势。",pdf)

        fig=canvas("06  错误分析：障碍物容易被预测为着陆区域", "本文无文本模型的行归一化混淆矩阵；每一行总计100%，表示该真实类别的预测去向。")
        cm=np.array(results['ours_no_text']['confusion_matrix'],dtype=float); norm=cm/cm.sum(1,keepdims=True)*100
        ax=fig.add_axes([.19,.21,.53,.57]);im=ax.imshow(norm,cmap="Blues",vmin=0,vmax=100,aspect="auto")
        ax.set_xticks(range(5),CLASSES);ax.set_yticks(range(5),CLASSES);ax.set(xlabel="预测类别",ylabel="真实类别");ax.tick_params(length=0,pad=10)
        for i in range(5):
            for j in range(5):ax.text(j,i,f"{norm[i,j]:.1f}%",ha="center",va="center",fontsize=14,color="white" if norm[i,j]>55 else INK)
        fig.colorbar(im,cax=fig.add_axes([.74,.21,.014,.57]))
        diag=read(RUNS/"ours_diagnostics/error_analysis.json")
        fig.text(.80,.64,f"{norm[0,4]:.1f}%",fontsize=33,color=ORANGE,weight="bold")
        fig.text(.80,.54,"真实障碍物像素\n被预测为着陆区域",fontsize=14,linespacing=1.6)
        fig.text(.80,.37,f"{diag['error_fraction_near_boundary']*100:.1f}%",fontsize=33,color=TEAL,weight="bold")
        fig.text(.80,.23,"错误像素位于\n真值边界5像素邻域\n（该邻域占全图15.1%）",fontsize=12,linespacing=1.6)
        save(fig,"06_confusion_and_errors","混淆矩阵与错误分析","行归一化混淆矩阵的对角线是各类别召回率，不是IoU。边界错误占比来自已有诊断记录，不能据此断言边界损失一定有效；大量边界误差和区域内部错分仍然存在。",pdf)

        fig=canvas("07  分割效果：相同样本、相同配色直观比较", "固定展示已保存预测中的前两张验证图（476、478），未依据模型胜负挑选。")
        gs=fig.add_gridspec(2,6,left=.025,right=.975,bottom=.20,top=.77,wspace=.035,hspace=.12)
        columns=["原图","真值标签",*map(LABELS.get,MAIN)]
        for row,name in enumerate(["476.png","478.png"]):
            paths=[DATA/"val_original"/name,DATA/"val_label"/name]+[RUNS/f"{n}_seed42/predictions"/name for n in MAIN]
            for col,path in enumerate(paths):
                SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
                ax=fig.add_subplot(gs[row,col]);ax.imshow(Image.open(path));ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(columns[col],fontsize=12,pad=12)
                if col==0:ax.set_ylabel(name,fontsize=11)
        fig.legend(handles=[Patch(color=c/255,label=n) for c,n in zip(COLORS,CLASSES)],loc="lower center",bbox_to_anchor=(.5,.105),ncol=5,frameon=False,fontsize=13)
        save(fig,"07_segmentation_examples","分割可视化","仅使用各方法已有最佳检查点保存的预测图，没有重新训练或推理。两张图用于直观展示，不代表完整验证集表现。推荐放大原PNG查看边界。",pdf)

        fig=canvas("08  历史结果：保留作为前期工作的记录", "这些数值来自原始课堂PPT；不与新协议结果拼成统一排行榜。", "历史结果未重跑 · 原评估分辨率与当前不同 · 训练环境和预处理记录不完整 · 不用于归因结构改进")
        history=(RUNS/'RESULTS.md').read_text(encoding='utf-8')
        historical_names=['U-Net','DeepLabV3+','MMDroneSeg (original)']
        hist=[float(next(line for line in history.splitlines() if line.startswith('| '+name+' |')).split('|')[2]) for name in historical_names]
        ax=fig.add_axes([.23,.26,.66,.49])
        ax.barh(range(3),hist,color=["#A4B2C2",BLUE,ORANGE],height=.55)
        ax.set_yticks(range(3),["U-Net","DeepLabV3+","原 MMDroneSeg"]);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel="历史记录 mIoU（%）");clean(ax)
        for i,v in enumerate(hist):ax.text(v+1,i,f"{v:.3f}",va="center",fontsize=17,weight="bold")
        SOURCES['runs/RESULTS.md']=hashlib.sha256((RUNS/'RESULTS.md').read_bytes()).hexdigest()
        save(fig,"08_historical_results","历史结果（单独列示）","原始报告：报告/第六组-贺锶函-论文-无人机图像分割.pptx，数值由runs/RESULTS.md中的既有整理记录读取参照。本页是前期记录而不是受控实验。不得用当前89.271减去历史78.238并将差值归因于模型结构。",pdf)

    with (OUT/"实验数据.csv").open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['run','展示名称','最佳轮次','mIoU%','F1%','像素准确率%','参数量',*CLASSES])
        for n,r in results.items():writer.writerow([n,LABELS[n],r['epoch'],r['miou']*100,r['f1']*100,r['pixel_accuracy']*100,r['parameters'],*[v*100 for v in r['class_iou']]])
    (OUT/'manifest.json').write_text(json.dumps({'pages':PAGES,'source_sha256':SOURCES},ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 课堂展示图表','', '共8张16:9中文图表；PNG为2880×1620，PDF适合无损缩放。合集PDF可直接放映，另有PPTX。', '', '推荐讲述顺序：01 → 02 → 03 → 04 → 05 → 06 → 07；08作为历史补充。', '', '“本文模型”始终指当前ConvNeXt-Tiny + FPN无文本版本，不是原三路ResNet结构。所有新实验数值均为单种子验证集结果，没有独立测试集。', '']
    for p in PAGES:lines += [f"## {p['name']} · {p['title']}", '',p['notes'],'']
    lines += ['## 重建','', '`python -B -m experiments.presentation_figures`', '', '本脚本仅读取已保存结果和图片，不调用训练、模型或推理。来源SHA256见manifest.json，数值表见实验数据.csv。']
    (OUT/'展示说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
