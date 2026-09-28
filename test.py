#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@xiaohong 2020.04
"""
import time
import copy
import torch
from torch import optim, nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
from IPython import display
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def set_figsize(figsize=(3.5, 2.5)):
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# 显示图片
def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

# 将输入图像和标签读进内存
def read_voc_images(root="../dataset/VOC2012", is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split() # 拆分成一个个名字组成list
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # 读入数据并且转为RGB的 PIL image
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels # PIL image 0-255

voc_dir = r"..\dataset\VOC2012"
# train_features, train_labels = read_voc_images(voc_dir, max_num=10)
#
# n = 5
# imgs = train_features[0:n] + train_labels[0:n] # PIL image
# show_images(imgs, 2, n)


# 标签中每个RGB颜色的值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 颜色与标签的映射：colormap2label
colormap2label = torch.zeros(256**3, dtype=torch.uint8) # torch.Size([16777216])
for i, colormap in enumerate(VOC_COLORMAP):
    # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap.convert("RGB")).astype('int32')      # (281, 500, 3)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])  # (281, 500)
    return colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标

# 将图像裁剪成固定尺寸而不是缩放
def voc_rand_crop(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    随机裁剪feature(PIL image) 和 label(PIL image).
    为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
    """
    i,j,h,w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    """
    Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label

# 双线性插值的上采样，用来初始化转置卷积层的卷积核
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size+1)//2
    if kernel_size%2 == 1:
        center = factor-1
    else:
        center = factor-0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1-abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
    weight = np.zeros((in_channels,out_channels, kernel_size,kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    weight = torch.Tensor(weight)
    weight.requires_grad = True   ######
    return weight

# 显示n张随机裁剪的图像和标签
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300) # PIL.Image.Image
# show_images(imgs[::2] + imgs[1::2], 2, n);

# 自定义VOC2012数据集
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        self.crop_size = crop_size # (h, w)
        features, labels = read_voc_images(root=voc_dir, is_train=is_train,  max_num=max_num)
# 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features) # PIL image
        self.labels = self.filter(labels)     # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
                                # float32 tensor           uint8 tensor (b,h,w)
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)



batch_size = 32 # 根据内存设定
# 指定随机裁剪的输出图像的形状为(320,480)
crop_size = (320, 480)
# 最多读取多少张图片
max_num = 20000

# 创建训练集和测试集的实例，查看训练集和测试集所保留的样本个数
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

# 设批量大小为32，分别定义【训练集】和【测试集】的数据迭代器
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                            num_workers=num_workers)

# 封装，把训练集和验证集保存在dict里
dataloaders = {'train':train_iter, 'val':test_iter}
dataset_sizes = {'train':len(voc_train), 'val':len(voc_test)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使⽤⼀个基于 ImageNet 数据集预训练的 ResNet-18 模型来抽取图像特征
num_classes = 21 # 类别
model_ft = resnet18(pretrained=True) # 设置True，表明使用训练好的参数

# 特征提取器，不更新参数
for param in model_ft.parameters():
    param.requires_grad = False

# 去掉最后两层的Globalpool和fc
model_ft = nn.Sequential(*list(model_ft.children())[:-2],
                    nn.Conv2d(512,num_classes,kernel_size=1),
                    nn.ConvTranspose2d(num_classes,num_classes, kernel_size=64, padding=16, stride=32)).to(device)

# 可以对model_ft做一个测试
# x = torch.rand((3,3,320,480), device=device)
# print(net(x).shape) # 依然是torch.Size([3, 21, 320, 480])

# 打印第一个小批量的类型和形状。不同于图像分类和目标识别，这里的标签是一个三维数组
# for X, Y in train_iter:
#     print(X.dtype, X.shape)
#     print(Y.dtype, Y.shape)
    # break

# 初始化model_ft的最后两层参数
nn.init.xavier_normal_(model_ft[-2].weight.data, gain=1)
model_ft[-1].weight.data = bilinear_kernel(num_classes, num_classes, 64).to(device)

# 训练函数
def train_model(model:nn.Module, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 每个epoch都有一个训练和验证阶段
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            runing_loss = 0.0
            runing_corrects = 0.0
            # 迭代一个epoch
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    logits = model(inputs)  # [5, 21, 320, 480]
                    loss = criteon(logits, labels.long())
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                runing_loss += loss.item()*inputs.size(0)
                runing_corrects += torch.sum((torch.argmax(logits.data,1))==labels.data)/(480*320)

            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since;
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model


epochs = 7
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 测试时需要把预测像素类别映射回在数据集中标注的颜色
def label2image(pred):
    # pred: [320,480]
    colormap = torch.tensor(VOC_COLORMAP,device=device,dtype=int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()

mean=torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1).to(device)
std=torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1).to(device)

# 测试函数
def visualize_model(model:nn.Module, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    n, imgs = num_images, []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device) # [b,3,320,480]
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1) # [b,320,480]
            inputs_nd = (inputs*std+mean).permute(0,2,3,1)*255 # 变回去

            for j in range(num_images):
                images_so_far += 1
                pred1 = label2image(pred[j]) # numpy.ndarray (320, 480, 3)
                imgs += [inputs_nd[j].data.int().cpu().numpy(), pred1, label2image(labels[j])]
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
                    return model.train(mode=was_training)


# 训练
model_ft = train_model(model_ft, criteon, optimizer, exp_lr_scheduler, num_epochs=5)
# 测试
visualize_model(model_ft)


# 另一种预测方法如下
# 预测前将图像标准化，并转换成(b,c,h,w)的tensor
def predict(img, model):
    tsf = transforms.Compose([
            transforms.ToTensor(), # 好像会自动转换channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    x = tsf(img).unsqueeze(0).to(device) # (3,320,480) -> (1,3,320,480)
    pred = torch.argmax(model(x), dim=1) # (1,320,480)
    return pred.reshape(pred.shape[1],pred.shape[2]) # (320,480)

def evaluate(model:nn.Module):
    model.eval()
    test_images, test_labels = read_voc_images(voc_dir, is_train=False, max_num=10) # PIL
    n, imgs = 4, []
    for i in range(n):
        xi, yi = voc_rand_crop(test_images[i], test_labels[i], 320, 480) # Image
        pred = label2image(predict(xi, model)) # 里面torch.Tensor [320, 480]
        # numpy.ndarray (320, 480, 3)
        imgs += [xi, pred, yi]
    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)