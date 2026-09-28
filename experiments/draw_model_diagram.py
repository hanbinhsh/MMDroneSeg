"""Editable SVG and vector PDF of the selected ours_no_text implementation.

Run with the bundled document Python (reportlab + pypdfium2).
No training, inference, raster tracing, or Illustrator-private formats required.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import hashlib
import json
import math

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/model_diagram'
W, H = 1800, 1170
INK = '#263541'
RGB = '#DEEAF4'
DETAIL = '#E1EFE7'
GATE = '#D9ECEB'
FPN = '#EAE2F0'
HEAD = '#F5E8D9'
GRAY = '#F3F5F7'
GREEN = '#43846A'
PURPLE = '#85709A'
ORANGE = '#AD7950'


class Drawing:
    def __init__(self, language):
        self.lang = language
        self.name = 'model_architecture_' + language
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                      '<title>DroneSegV2: selected no-text architecture</title>',
                      '<desc>Editable vector drawing of experiments/ours.py; detail=True, text=False, boundary=True, refine=False.</desc>']
        self.pdf = canvas.Canvas(str(OUT / (self.name+'.pdf')), pagesize=(W,H))
        self.pdf.setTitle('DroneSegV2 — selected no-text architecture')
        self.pdf.setAuthor('Model diagram generated from the implemented architecture')
        self.font = 'CN' if language == 'zh' else 'EN'
        self.family = 'Microsoft YaHei' if language == 'zh' else 'Times New Roman'
        self.rect(0,0,W,H,'#FFFFFF',stroke=None)

    def tr(self, en, zh):
        return zh if self.lang == 'zh' else en

    def group(self, name):
        self.parts.append(f'<g id="{name}" data-name="{name}">')

    def end(self): self.parts.append('</g>')

    def rect(self,x,y,w,h,fill=GRAY,stroke=INK,r=8,dash=False,lw=1.4):
        dashsvg=' stroke-dasharray="7 5"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{lw}"{dashsvg}/>')
        c=self.pdf;c.saveState();c.setFillColor(HexColor(fill));c.setLineWidth(lw)
        if stroke:c.setStrokeColor(HexColor(stroke))
        if dash:c.setDash(7,5)
        c.roundRect(x,H-y-h,w,h,r,stroke=int(bool(stroke)),fill=1);c.restoreState()

    def text(self,x,y,lines,size=20,bold=False,anchor='middle',color=INK,gap=None):
        if isinstance(lines,str):lines=[lines]
        gap=gap or size*1.28
        for i,line in enumerate(lines):
            yy=y+i*gap
            self.parts.append(f'<text x="{x}" y="{yy}" font-family="{self.family}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" fill="{color}">{escape(line)}</text>')
            c=self.pdf;c.setFillColor(HexColor(color));c.setFont(self.font+('B' if bold else ''),size)
            getattr(c,{'middle':'drawCentredString','start':'drawString','end':'drawRightString'}[anchor])(x,H-yy,line)

    def line(self,points,color=INK,dash=False,arrow=True,lw=1.8):
        d='M '+' L '.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
        c=self.pdf;c.saveState();c.setStrokeColor(HexColor(color));c.setLineWidth(lw)
        if dash:c.setDash(7,5)
        path=c.beginPath();path.moveTo(points[0][0],H-points[0][1])
        for x,y in points[1:]:path.lineTo(x,H-y)
        c.drawPath(path);c.restoreState()
        if arrow:
            x,y=points[-1];px,py=points[-2];angle=math.atan2(y-py,x-px)
            pts=[(x,y),(x-9*math.cos(angle)+4*math.sin(angle),y-9*math.sin(angle)-4*math.cos(angle)),(x-9*math.cos(angle)-4*math.sin(angle),y-9*math.sin(angle)+4*math.cos(angle))]
            self.parts.append('<polygon points="'+' '.join(f'{xx},{yy}' for xx,yy in pts)+f'" fill="{color}"/>')
            c.saveState();c.setFillColor(HexColor(color));p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
            for xx,yy in pts[1:]:p.lineTo(xx,H-yy)
            p.close();c.drawPath(p,fill=1,stroke=0);c.restoreState()

    def box(self,x,y,w,h,lines,fill=GRAY,size=20,dash=False):
        self.rect(x,y,w,h,fill,dash=dash)
        if isinstance(lines,str):lines=[lines]
        self.text(x+w/2,y+h/2-(len(lines)-1)*size*.64+size*.32,lines,size=size)

    def panel(self,x,y,w,h,letter,title):
        self.rect(x,y,w,h,'#FFFFFF',r=12,lw=1.8)
        self.text(x+18,y+34,f'({letter})  '+title,size=24,bold=True,anchor='start')

    def finish(self):
        self.parts.append('</svg>')
        (OUT/(self.name+'.svg')).write_text('\n'.join(self.parts),encoding='utf-8')
        self.pdf.showPage();self.pdf.save()
        doc=pdfium.PdfDocument(str(OUT/(self.name+'.pdf')))
        doc[0].render(scale=1.6).to_pil().save(OUT/(self.name+'.png'))


def make(lang):
    d=Drawing(lang);t=d.tr
    d.group('a-overall-architecture')
    d.panel(20,20,1760,515,'a',t('Overall architecture of the selected model','当前选用模型的整体结构'))
    d.text(1760,53,t('ConvNeXt-Tiny + gated detail injection + FPN','ConvNeXt-Tiny + 细节门控注入 + FPN'),size=18,anchor='end',color='#697681')
    # All arrows are placed explicitly; no SVG marker definitions or raster assets.
    d.line([(185,252),(210,252),(210,175),(240,175)])
    d.line([(210,252),(210,367),(240,367)],GREEN)
    for a,b in [(480,540),(740,800),(1050,1120)]:d.line([(a,175),(b,175)])
    for a,b in [(480,540),(740,800)]:d.line([(a,367),(b,367)],GREEN)
    d.line([(925,320),(925,230)],GREEN)
    d.text(941,276,['D1, D2',t('128 channels','128 通道')],size=17,anchor='start',color=GREEN)
    d.line([(1210,230),(1210,320)],PURPLE)
    d.text(1225,266,['R1–R4'],size=18,anchor='start',color=PURPLE)
    d.line([(1300,367),(1335,367),(1335,267),(1380,267)])
    d.line([(1335,367),(1335,425),(1380,425)],ORANGE,dash=True)
    d.line([(1525,267),(1585,267)])
    d.line([(1525,425),(1585,425)],ORANGE,dash=True)
    d.box(50,205,135,94,[t('RGB image','RGB 图像'),'3 × H × W'],RGB,21)
    d.box(240,120,240,110,['ConvNeXt-Tiny',t('ImageNet pretrained','ImageNet 预训练'),t('4-stage encoder','四阶段编码器')],RGB,21)
    d.text(360,258,['C1: 96 @ 1/4; C2: 192 @ 1/8','C3: 384 @ 1/16; C4: 768 @ 1/32'],size=16)
    d.box(540,120,200,110,[t('Lateral projection','侧向通道投影'),'1 × 1 Conv → 128','C1 / C2 / C3 / C4'],RGB,18)
    d.text(640,95,'P1, P2, P3, P4',size=19)
    d.box(800,120,250,110,[t('Gated detail injection','细节门控注入'),t('P1, P2: see (b)','P1、P2：详见 (b)'),t('P3, P4: unchanged','P3、P4：直接传递')],GATE,20)
    d.box(1120,120,180,110,[t('Top-down FPN','自顶向下 FPN'),t('+ smoothing','+ 平滑卷积'),t('See (c)','详见 (c)')],FPN,21)
    d.box(240,320,240,94,[t('Grayscale → DoG + Otsu','灰度 → DoG + Otsu'),t('Concatenate two maps','拼接为双通道输入')],DETAIL,20)
    d.text(360,447,t('Signed DoG; binary threshold map','有符号高斯差分；二值阈值图'),size=17,color=GREEN)
    d.box(540,320,200,94,[t('Detail CNN','细节卷积分支'),'2 → 32 → 64 → 128',t('Stride 2 per block','每个卷积块步长为2')],DETAIL,18)
    d.text(640,446,t('Take 64 @ 1/4, 128 @ 1/8','提取 64@1/4、128@1/8'),size=16,color=GREEN)
    d.box(800,320,250,94,[t('Two lateral projections','两级侧向通道投影'),'1 × 1 Conv → 128'],DETAIL,20)
    d.box(1120,320,180,94,[t('Resize + concat','对齐尺寸并拼接'),t('512 → 128 Conv','512 → 128 卷积'),'Dropout2d 0.1'],FPN,18)
    d.text(1210,447,'F: 128 × H/4 × W/4',size=16,color=PURPLE)
    d.box(1380,225,145,84,[t('Seg. head','分割预测头'),'1 × 1 Conv','128 → 5'],HEAD,18)
    d.box(1585,225,165,84,[t('Bilinear ↑4×','双线性上采样4×'),t('5-class logits','五类分割 logits')],HEAD,18)
    d.text(1667,341,t('Argmax → label map','Argmax → 分割标签图'),size=17)
    d.box(1380,380,145,90,[t('Boundary head','边界预测头'),'128 → 32 → 1'],HEAD,18,dash=True)
    d.box(1585,380,165,90,[t('Bilinear ↑4×','双线性上采样4×'),t('Boundary logits','边界 logits')],HEAD,18,dash=True)
    d.text(1520,503,t('Dashed: auxiliary boundary supervision during training','虚线：训练阶段的边界辅助监督'),size=17,color=ORANGE)
    d.line([(56,494),(100,494)]);d.text(111,500,t('RGB / semantic flow','RGB / 语义特征流'),size=17,anchor='start')
    d.line([(390,494),(434,494)],GREEN);d.text(445,500,t('Derived detail flow','派生细节特征流'),size=17,anchor='start',color=GREEN)
    d.text(797,500,t('All upsampling: bilinear, align_corners=False','上采样均为双线性插值，align_corners=False'),size=16,anchor='start')
    d.end()

    d.group('b-gated-detail-injection')
    d.panel(20,555,850,365,'b',t('Gated detail injection (i = 1, 2)','细节门控注入（i = 1, 2）'))
    d.box(48,625,140,58,['Pi: 128 ch'],RGB,20)
    d.box(48,734,140,58,['Di: 128 ch'],DETAIL,20)
    d.line([(188,654),(272,654),(272,680),(290,680)])
    d.line([(188,763),(272,763),(272,710),(290,710)],GREEN)
    d.box(290,666,110,58,t('Concat','拼接'),GRAY,20)
    d.line([(400,695),(435,695)])
    d.box(435,655,145,80,['1 × 1 Conv','256 → 128'],GATE,20)
    d.line([(580,695),(610,695)])
    d.box(610,666,105,58,'Sigmoid',GATE,19)
    d.line([(715,695),(761,695),(761,751)],GREEN)
    d.line([(188,763),(716,763)],GREEN)
    d.box(716,751,90,38,'×',DETAIL,22)
    d.line([(118,625),(118,611),(782,611),(782,625)])
    d.line([(806,770),(836,770),(836,735),(782,735),(782,669)],GREEN)
    d.box(760,625,44,44,'+',GATE,24)
    d.line([(804,647),(846,647)])
    d.text(830,631,'P’i',size=20)
    d.text(445,831,'Gi = sigmoid(Conv1×1([Pi, Di]));   P’i = Pi + Gi × Di',size=23)
    d.text(445,881,t('Spatial and channel-wise gate; bias initialized to −2.','门控权重随空间位置和通道变化；门控卷积偏置初始化为 −2。'),size=18,color='#697681')
    d.end()

    d.group('c-top-down-fpn')
    d.panel(890,555,890,365,'c',t('Top-down pyramid and multi-scale fusion','自顶向下特征金字塔与多尺度融合'))
    xs=[1000,1200,1400,1600]
    for j,x in enumerate(xs):
        i=4-j
        label='P'+str(i)+('’' if i<=2 else '')
        d.text(x,635,label,size=23,bold=True)
        d.line([(x,645),(x,675)],PURPLE)
        d.box(x-65,675,130,46,'Q'+str(i),FPN,20)
        if j<3:
            d.line([(x+65,698),(xs[j+1]-65,698)],PURPLE)
            d.text(x+100,676,'↑2× +',size=17,color=PURPLE)
        d.line([(x,721),(x,753)],PURPLE)
        d.box(x-80,753,160,55,['3 × 3 Conv','GN + GELU'],FPN,17)
        d.text(x,836,'R'+str(i)+' @ 1/'+str(2**(i+1)),size=18)
    d.text(1335,879,t('R1–R4 → resize to H/4 × W/4 → concat (512 ch) → fuse','R1–R4 → 对齐至 H/4 × W/4 → 拼接（512通道）→ 融合'),size=19)
    d.end()

    d.group('d-training-objective')
    d.panel(20,940,1760,210,'d',t('Training objective and implementation notes','训练目标与实现说明'))
    d.text(900,1020,'L = LCE(S, Y) + LDice(S, Y) + 0.1 LBCE(B, E(Y))',size=30,bold=True)
    d.text(900,1063,t('S: segmentation logits   ·   B: boundary logits   ·   Y: ground-truth labels   ·   E(Y): adjacent-label boundaries','S：分割 logits   ·   B：边界 logits   ·   Y：真值标签   ·   E(Y)：相邻像素标签变化生成的边界'),size=20)
    d.text(900,1105,t('Conv blocks: Conv3×3 + GroupNorm(8) + GELU. Ignore label: 255. Q4 = P4; Qi = Pi* + ↑Qi+1 (i = 3, 2, 1).','卷积块：Conv3×3 + GroupNorm(8) + GELU。忽略标签：255。Q4 = P4；Qi = Pi* + ↑Qi+1（i = 3, 2, 1）。'),size=19)
    d.text(900,1135,t('Pi* = P’i for i = 1, 2; otherwise Pi. Smoothing is applied after all top-down additions.','i = 1、2 时 Pi* = P’i，其余为 Pi。所有自顶向下相加完成后，再分别进行平滑卷积。'),size=17,color='#697681')
    d.end();d.finish()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,path in [('EN','times.ttf'),('ENB','timesbd.ttf'),('CN','msyh.ttc'),('CNB','msyhbd.ttc')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+path))
    for language in ['en','zh']:make(language)
    manifest={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/ours.py','experiments/models.py','experiments/data.py','experiments/losses.py']}
    (OUT/'code_sources.json').write_text(json.dumps({'variant':'ours_no_text','flags':{'detail':True,'text':False,'boundary':True,'refine':False},'sha256':manifest},indent=2),encoding='utf-8')
    print('Created editable SVG, vector PDF and PNG previews in',OUT)


if __name__ == '__main__':main()
