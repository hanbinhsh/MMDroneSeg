"""Paper tables and the user's original five examples, using saved outputs only."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/paper_comparison'
DATA=ROOT.parent/'Drone/classes_dataset/classes_dataset'
NAMES=['478.png','480.png','479.png','484.png','476.png']
MODELS=['ours_no_text','afenet','d2ls','logcan']
DISPLAY=['Ours (no text)','AFENet','D2LS','LOGCAN++']
CLASSES=['Obstacles','Water','Soft-surfaces','Moving-objects','Landing-zones']
COLORS=[(155,38,182),(14,135,204),(124,252,0),(255,20,147),(169,169,169)]
SOURCES={}


def source(path):
    SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def three_line(ax, headers, names, values, digits):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    n=len(headers); centers=[.15]+list(np.linspace(.36,.94,n-1))
    ax.plot([0,1],[.985,.985],color='black',lw=2.0)
    ax.plot([0,1],[.735,.735],color='black',lw=1.2)
    ax.plot([0,1],[.045,.045],color='black',lw=1.2)
    for x,h in zip(centers,headers):ax.text(x,.86,h,ha='center',va='center',fontsize=18,weight='bold',linespacing=1.2)
    best=np.max(values,axis=0)
    for i,(name,row) in enumerate(zip(names,values)):
        y=.735-(i+.5)*(.69/len(names))
        ax.text(centers[0],y,name,ha='center',va='center',fontsize=18)
        for j,v in enumerate(row):ax.text(centers[j+1],y,f'{v:.{digits}f}',ha='center',va='center',fontsize=18,weight='bold' if v==best[j] else 'normal')


def tables(names, overall, class_iou, stem, proportions=None, historical=False):
    # Keep text editable in SVG, embed TrueType fonts in PDF.
    fig=plt.figure(figsize=(16,8.1),facecolor='white')
    three_line(fig.add_axes([.17,.57,.66,.38]),['Model Name','PixAcc','mIoU','F1 (macro)' if not historical else 'F1'],names,overall,5)
    headers=['Model Name']+[name+(f'\n({p*100:.2f}%)' if proportions is not None else '') for name,p in zip(CLASSES,proportions if proportions is not None else [0]*5)]
    three_line(fig.add_axes([.025,.13,.95,.37]),headers,names,class_iou,4)
    note=('Historical results from the original presentation; not rerun. Protocol differs from the current experiments.' if historical else
          'Validation set: 80 images; seed 42; 50 epochs; no TTA. Bold: best in each column. Scores are in [0, 1].')
    fig.text(.5,.071,note,ha='center',fontsize=12)
    if not historical:fig.text(.5,.037,'Class percentages are pixel proportions in the full validation set. F1 is the unweighted mean across five classes.',ha='center',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=300,facecolor='white')
    plt.close(fig)


def latex_table(headers,names,values,digits):
    best=np.max(values,axis=0)
    lines=['\\begin{tabular}{l'+'c'*len(best)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    for name,row in zip(names,values):
        cells=[name]+[(r'\textbf{'+f'{v:.{digits}f}'+'}') if v==best[j] else f'{v:.{digits}f}' for j,v in enumerate(row)]
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])


def gallery():
    # Preserve every original pixel: no resizing, cropping, smoothing or recoloring.
    iw,ih,gap,margin,top,bottom=960,736,16,70,135,180
    width=2*margin+6*iw+5*gap
    height=top+5*ih+4*gap+bottom
    out=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(out)
    font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',49)
    bold=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',49)
    small=ImageFont.truetype('C:/Windows/Fonts/times.ttf',30)
    headings=['Input Image',*DISPLAY,'Ground Truth']
    for j,label in enumerate(headings):draw.text((margin+j*(iw+gap)+iw/2,60),label,font=bold if j==1 else font,anchor='mm',fill='black')
    for i,name in enumerate(NAMES):
        paths=[DATA/'val_original'/name]+[ROOT/f'runs/{n}_seed42/predictions'/name for n in MODELS]+[DATA/'val_label'/name]
        draw.text((margin-10,top+i*(ih+gap)+ih/2),name.removesuffix('.png'),font=small,anchor='rm',fill='#555555')
        for j,path in enumerate(paths):
            with Image.open(source(path)) as im:
                assert im.size==(iw,ih)
                image=im.convert('RGB')
                if j>0:
                    colors=set(image.getdata())
                    assert colors.issubset(set(COLORS)),(path,colors-set(COLORS))
                out.paste(image,(margin+j*(iw+gap),top+i*(ih+gap)))
    # Highlight our column outside the actual mask pixels.
    x=margin+iw+gap
    draw.rectangle([x-6,top-6,x+iw+5,top+5*ih+4*gap+5],outline='#BD3333',width=4)
    y=height-115
    for i,(name,color) in enumerate(zip(CLASSES,COLORS)):
        x=440+i*1040
        draw.rectangle([x,y-18,x+64,y+18],fill=color)
        draw.text((x+82,y),name,font=font,anchor='lm',fill='black')
    draw.text((width/2,height-42),'Same five examples as the original figure; native 960 x 736 images; saved best-checkpoint predictions; no TTA.',font=small,anchor='mm',fill='#444444')
    out.save(OUT/'five_image_comparison_native.png',dpi=(300,300))
    preview=out.copy();preview.thumbnail((1800,1800));preview.save(OUT/'five_image_comparison_preview.jpg',quality=94)
    return [width,height]


def main():
    OUT.mkdir(exist_ok=True)
    for name in ['times.ttf','timesbd.ttf']:font_manager.fontManager.addfont('C:/Windows/Fonts/'+name)
    plt.rcParams.update({'font.family':'Times New Roman','svg.fonttype':'none','pdf.fonttype':42})
    results=[]
    for model in MODELS:
        folder=ROOT/f'runs/{model}_seed42'
        r=json.loads(source(folder/'evaluation.json').read_text())
        status=json.loads(source(folder/'status.json').read_text())
        assert status['epoch']==50 and status['state']=='stage_complete'
        assert r['seed']==42 and r['protocol']=='native_960x736_crop512_sliding_no_tta'
        cm=np.array(r['confusion_matrix'],dtype=np.float64)
        tp=cm.diagonal();iou=tp/(cm.sum(0)+cm.sum(1)-tp);f1=2*tp/(cm.sum(0)+cm.sum(1))
        assert np.isclose(iou.mean(),r['miou']) and np.isclose(f1.mean(),r['f1']) and np.isclose(tp.sum()/cm.sum(),r['pixel_accuracy'])
        results.append(r)
    totals=np.array([r['confusion_matrix'] for r in results]).sum(axis=2)
    assert np.all(totals==totals[0])
    proportions=totals[0]/totals[0].sum()
    overall=np.array([[r[k] for k in ['pixel_accuracy','miou','f1']] for r in results])
    perclass=np.array([r['class_iou'] for r in results])
    tables(DISPLAY,overall,perclass,'paper_tables_current',proportions)
    historical=json.loads(source(ROOT/'runs/historical_results.json').read_text())
    historical.sort(key=lambda r:['MMDroneSeg (original)','U-Net','DeepLabV3+'].index(r['model']))
    historical_names=['MMDroneSeg (original)','U-Net','DeepLabV3+']
    ho=np.array([[r[k] for k in ['pixel_accuracy','miou','f1']] for r in historical]);hc=np.array([r['class_iou'] for r in historical])
    tables(historical_names,ho,hc,'paper_tables_historical',historical=True)
    with (OUT/'metrics_current.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Model Name','PixAcc','mIoU','F1 (macro)',*CLASSES])
        for n,o,c in zip(DISPLAY,overall,perclass):writer.writerow([n,*o,*c])
    tex='% Requires \\usepackage{booktabs}. Current protocol only; values in [0,1].\n\n'+latex_table(['Model Name','PixAcc','mIoU','F1 (macro)'],DISPLAY,overall,5)+'\n\n'+latex_table(['Model Name',*CLASSES],DISPLAY,perclass,4)+'\n'
    (OUT/'paper_tables_current.tex').write_text(tex,encoding='utf-8')
    size=gallery()
    docdata={'models':DISPLAY,'overall':overall.tolist(),'class_iou':perclass.tolist(),'class_names':CLASSES,'class_proportions':proportions.tolist(),'historical_models':historical_names,'historical_overall':ho.tolist(),'historical_class_iou':hc.tolist()}
    (OUT/'table_data.json').write_text(json.dumps(docdata,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'selected_images':NAMES,'selection_basis':'Matched original screenshot visually against the five original image files retained in project root; fixed-selection plotting script not present in current workspace.','models':dict(zip(MODELS,DISPLAY)),'gallery_size':size,'source_sha256':SOURCES}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 论文三线表与五张图可视化','', '本次仅读取已有结果和预测，无训练、无新推理。','', '## 表格','', 'paper_tables_current：当前四种方法的PixAcc、mIoU、宏平均F1及五类别IoU；PNG、PDF、SVG三种格式。SVG保留可编辑文本。','', '数值范围为0–1。每列最大值分别加粗：不能因为本文方法的mIoU最高，就把它的PixAcc也加粗。类别百分比重新根据80张验证图的真值像素数计算，与历史表中的占比不同。','', 'F1为五类F1的算术平均；每类F1=2TP/(2TP+FP+FN)。PixAcc为全部有效像素的正确率。指标已由保存的混淆矩阵复核。','', 'paper_tables_historical：原MMDroneSeg、U-Net、DeepLabV3+的历史结果，单独保存；不与新协议做直接归因比较。历史表保留原报告F1定义，未假设与当前实现完全一致。','', '## 五张图','', '从上到下：478.png、480.png、479.png、484.png、476.png，与用户提供的旧图顺序一致。当前代码目录未找到固定五张选择脚本，通过项目中保留的原图与参考截图逐一核对确定。','', '从左到右：原图、Ours (no text)、AFENet、D2LS、LOGCAN++、GT。所有模型使用各自50轮预算内最佳检查点的已保存预测，普通滑窗推理，无TTA。红框仅标示本文方法列，不代表每个样例均最优。','', f'原尺寸合成图：{size[0]}×{size[1]}；每个子图保留960×736像素，无裁剪、无插值。PNG和PDF可用于论文与放大查看，JPG为快速预览。','', '当前五图对比不加入旧U-Net/DeepLabV3+截图中的低分辨率预测，也没有重新运行这些模型。','', '本文模型是ConvNeXt-Tiny+FPN无文本版本，与旧图中的原MMDroneSeg不同。单种子验证集结果，不代表统计显著优势。','', '## 可直接使用的图注','', '图X 不同方法在无人机俯视图语义分割任务中的定性比较。自左至右依次为输入图像、本文无文本模型、AFENet、D2LS、LOGCAN++和真值标签。五个样例沿用原报告选取的图像及顺序；各方法采用50轮训练预算内最佳验证检查点的普通滑窗预测。颜色分别表示障碍物、水体、软质表面、移动物体和着陆区域。','', 'Figure X. Qualitative comparison on five UAV-view images. Columns show the input, our no-text model, AFENet, D2LS, LOGCAN++, and ground truth. The examples and their order follow the original report. Predictions use the best validation checkpoint within a 50-epoch budget, with sliding-window inference and no test-time augmentation.','', '## 文件用途','', '- paper_tables_current.docx：可直接复制到Word论文的原生可编辑三线表。','- paper_tables_current.tex：LaTeX表格源码，依赖booktabs。','- metrics_current.csv：未四舍五入的数值。','- manifest.json：固定图片顺序和所有输入文件SHA256。']
    (OUT/'使用说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Created',OUT,'gallery size',size)


if __name__=='__main__':main()
