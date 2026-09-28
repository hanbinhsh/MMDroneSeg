# 无人机五类分割实验

## 当前约定

- 仅使用种子 **42**。不重新训练 U-Net、DeepLabV3+，保留 PPT 的同学实验结果。
- 新方法：D2LS（ICCV 2025）、AFENet（TGRS 2025）、LOGCAN++（TGRS 2025）。使用作者网络源码，放在 `external/`。
- 改进方法：`ours`，ConvNeXt-Tiny + 128 通道 FPN + 轻量 DoG/阈值细节分支 + 全局文本 FiLM + 边界辅助监督。
- 消融：`ours_rgb`、`ours_no_detail`、`ours_no_text`、`ours_no_boundary`。只使用 RGB 的变体不加载文本缓存。
- 没有独立测试集；所有本地结果均为验证集结果，不声称跨场景泛化或单种子统计显著性。

## 数据和协议

保留 `../Drone/classes_dataset/classes_dataset` 中原始 320/80 图像划分。所有图像为 960×736。
训练集像素比例已从原始标签统计：obstacles 9.005%、water 2.973%、soft-surfaces 33.530%、moving-objects 2.160%、landing-zones 52.332%。
完整文件清单见 `runs/data_audit.json`。没有训练/验证间的像素完全相同图像，尚未验证近似重复或场景级独立性。

训练从原图随机裁剪 512×512，每轮每图一个裁剪，使用水平/垂直翻转、90°旋转及轻度亮度扰动。
图像使用 ImageNet 归一化。标签精确转换颜色，忽略填充值 255，不对类别 ID 使用双线性插值。
验证使用 512 窗口、384 步长滑窗，平均重叠 logits，在 960×736 原图标签上累计混淆矩阵。
无 TTA；mIoU 为全部五类 IoU 的算术平均，全集中分母为零的类别记 0。

历史 U-Net 75.187%、DeepLabV3+ 80.538%、MMDroneSeg 78.238% 单列保留，不与新协议结果直接作结构收益归因。
原训练代码是 368×480 输入；旧基线具体骨干和执行环境未保存。不能因为新协议超过历史 80.538% 就声称同条件击败 DeepLabV3+。

## 环境和来源

当前可用环境：Python 3.12.2，PyTorch 2.2.1+cu121，torchvision 0.17.1+cu121，timm 1.0.22，einops 0.8.1。
GPU：RTX 3080 Ti Laptop 16 GB。没有升级或改动系统安装的 PyTorch。
新适配器不依赖 Lightning、MMCV 或 MMSeg；直接调用作者网络模块，避免旧项目的框架版本冲突。
实际完整环境记录在 `runs/environment.txt`，每个训练目录保存源码哈希、仓库提交、配置和设备信息。

| 模型 | 官方来源 | 采用配置 |
|---|---|---|
| D2LS | https://github.com/XavierJiezou/D2LS | ConvNeXt-Base / ImageNet-1K，l=3，5 个类别 token |
| AFENet | https://github.com/oucailab/AFENet | 官方默认 SWSL ResNet18，包含额外预训练数据；不能声称所有骨干预训练数据相同 |
| LOGCAN++ | https://github.com/xwmaxwma/rssegmentation | 当前作者配置的 RepViT-M2.3，450 epoch 蒸馏预训练，head=96、8 heads |

固定版本及下载权重 SHA256 分别记录在每次运行的 `config.json`、`weights/manifest.json`。
D2LS 大型展示附件导致 Git 检出失败，使用同一提交的 GitHub 原始文件恢复 50 个代码/配置文件；每个文件与 Git blob SHA1 核对，见 `external/d2ls_snapshot.json`。
未修改作者网络源码。兼容处理：本地权重加载、AFENet 旧 timm 名称映射、FFT 固定 FP32、五分类输出适配。
D2LS 原配置缺少训练脚本检查的 `has_contrastive_loss` 开关，但模型默认返回对比损失；适配器显式解析返回值并计入该损失。
LOGCAN++ 使用作者的冻结骨干 BN 统计策略；CUDA grid_sample 反向有非确定性，固定种子不等于逐位确定。

## 训练预算和损失

- 单 GPU 顺序执行，batch=2，梯度累积到有效 batch=8，AMP；每轮验证，有提升时保存最优权重；完整恢复检查点改为每5轮保存，阶段结束、早停和收到STOP时也保存。
- 2026-09-22 按用户要求，将每个模型的总训练上限统一改为 **50 轮**，包括已完成轮次；先完成30轮阶段再续至50轮。队列和单模型入口默认 `--max-epochs 50`，队列会截断超过上限的阶段，单模型入口会拒绝超限参数。
- 保留现有学习率调度器的150轮时间尺度和优化器状态，仅缩短停止轮次，不重置已训练模型的学习率。故本次50轮结果是原训练计划的提前截断，未宣称重新针对50轮预算优化调度器。
- 最少 30 轮后，验证 mIoU 连续 30 轮无提升早停；不根据历史基线分数提前结束。
- D2LS：作者 CE(平滑0.05)+Dice + 0.4×辅助CE + 对比损失；AdamW+Lookahead，lr=1e-4，wd=0.01，cosine。
- AFENet：作者 CE(平滑0.05)+Dice；AdamW+Lookahead，head lr=6e-4、backbone=6e-5，wd=0.01，cosine warm restarts(15,2)。
- LOGCAN++：CE+0.8×辅助CE；AdamW，lr=1e-4，wd=1e-4，poly0.9。
- Ours：CE+SoftDice+0.1×边界BCE；AdamW，head lr=3e-4、backbone=3e-5，wd=0.01；5轮warmup后cosine。
- 相比作者原论文，训练数据、裁剪、有效batch、训练轮数等发生适配，均明确记录。保留了网络和特有损失，并非宣称原论文基准复现。

## 命令

在本目录执行（Windows 多进程 DataLoader 需要能创建系统管道的普通终端）：

```powershell
python -m experiments.prepare --audit
python -m experiments.prepare --weights convnext_tiny afenet logcan d2ls
python -m pytest experiments/test_pipeline.py -q

# 正式队列：不包含 U-Net 或 DeepLabV3+
python -u -m experiments.run_suite --models ours afenet logcan d2ls ours_rgb --stages 30 50 --max-epochs 50

# 单独运行/续跑
python -u -m experiments.train --model d2ls --epochs 30
python -u -m experiments.train --model d2ls --epochs 50 --resume

# 消融，可在首批模型完成后顺序运行
python -u -m experiments.run_suite --models ours_no_detail ours_no_text ours_no_boundary --stages 30 50 --max-epochs 50

# 从最优检查点重新评估并输出预测图
python -m experiments.train --model ours --evaluate
python -m experiments.predict --checkpoint runs/ours_seed42/best.pt --image ../Drone/classes_dataset/classes_dataset/val_original/476.png --output runs/prediction_476.png
python -m experiments.report
python -m experiments.plot_results

# 从已归档的 30 轮评估生成同预算对比（不混入续跑结果）
python -m experiments.analyze_stage --stage 30
```

已有队列运行时可加 `--wait-for-queue` 排在其后，避免同时占用 GPU。

权重下载失败时可以显式传入 Windows 已配置的 `--proxy http://127.0.0.1:7890`；不需要更改全局代理。
新图片的文本模型预测需先生成对应 BLIP/CLIP 缓存，缺少缓存会报错，绝不以零向量静默替代。
比较模型 FPS 时应分别报告网络推理和包含 BLIP/CLIP 的端到端成本；当前日志 `seconds` 是包含读取、滑窗和指标的整套验证用时，不是模型 FPS。

## 输出与停止

`runs/<model>_seed42/` 包含 `config.json`、`metrics.jsonl`、`best.pt`、`last.pt`、`status.json`、阶段结束时的 `evaluation.json`、`per_image.json` 和前六张固定验证图的预测。
汇总文件为 `runs/RESULTS.md`、`runs/comparison_local.csv`、`runs/historical_results.json`。
首批全部方法的 30 轮评估已保存在各目录的 `stage_30/`，同预算对比为 `runs/STAGE_30.md` 和 `runs/comparison_stage_30.csv`。
`experiments.plot_results` 根据已完成阶段导出训练曲线 PNG/PDF 和固定验证图对照，保存到 `runs/figures/`；多个方法的损失定义不同，不能直接以损失数值高低评价优劣。
查看 `runs/queue_status.json` 和 `runs/<model>_seed42.console.log` 判断当前任务，不把 `smoke_*` 的检查分数视为论文结果。

在模型运行目录创建空文件 `STOP`，会在当前完整轮次保存后停止该模型。
在 `runs/` 创建 `STOP_QUEUE`，队列不会启动下一个任务。继续前删除相应停止标记。
队列使用 `runs/queue.lock` 防止误开多个完整训练任务；异常断电后先检查记录的 PID 是否仍存活，再移除失效锁。

检查点、JSON 状态和训练指标现在会 flush/fsync 强制写盘；替换检查点时保留 `last.prev.pt`、`best.prev.pt` 作为上一版备份。此措施降低断电损坏风险，不保证硬件故障下零损失。恢复前仍需检查文件可读性及优化器状态。

按用户减少硬盘写入的要求，`--checkpoint-every` 默认5。模型恢复文件实测：D2LS约1390 MiB、LOGCAN++约290 MiB、AFENet约309 MiB、Ours约338 MiB。周期性完整检查点写入次数约减少80%，突然断电可能回退最多约5轮。最优权重另存，单份约77–359 MiB；`*.prev.pt` 通过重命名轮换，不额外复制大文件。最多保留当前和上一版，各两份完整恢复/最优权重文件，不逐轮累积。`status.json` 的 `checkpoint_epoch` 表示实际可恢复轮次，可能小于已完成的 `epoch`。

2026-09-22 断电使 LOGCAN++ 的 `last.pt` 和 `status.json` 变成零字节填充内容，无法精确恢复优化器。旧目录保存在 `runs/logcan_seed42_powerloss_20260922_1102/`；第105轮最佳权重完整、验证 mIoU 为90.110%。该结果仅作故障前归档，不能与重新训练的结果择优合并。LOGCAN++ 按相同种子42从头重跑，其余七个方法已验证完整检查点并从30轮续跑。恢复记录见 `runs/powerloss_recovery_20260922.json`。原 `STAGE_30.md` 保留故障前的阶段对比，后续正式汇总使用恢复后的活动目录。

## 旧代码修复

保留旧网络与入口，修正实际预训练权重加载、编码器微调、可微 Dice/边界损失、忽略标签、路径、验证文本随机性和水域频率。
旧 `main.py` 默认 batch 调为2，保留原有 lr_gamma 和 edge_weight 的用户修改。
后续论文实验以 `experiments` 入口为准，旧报告数字不自动覆盖。自定义 VAE 预训练权重来源未核实，主实验不使用它。

## 仅针对我们模型的后续改进（2026-09-22）

用户要求其他对比方法不再实验。仅新增以下两组，均从同一ConvNeXt-Tiny预训练骨干开始，种子42、50轮、每5轮完整保存，不在旧50轮检查点上再追加50轮：

- `ours_sched50`：与原 `ours_no_text` 结构、损失完全相同，仅把学习率调度器时间尺度从150改为50轮，含5轮warmup。这是短预算训练对照。
- `ours_refine`：在原无文本模型基础上增加1/2分辨率RGB+DoG/阈值卷积分支，结合FPN语义特征、粗分割logits和预测边界，预测五类logit残差；边界门控系数为0.25+0.75×sigmoid(edge)，保留内部区域修正能力。残差末层零初始化，初始输出与原模型一致。参数从29,405,766增至29,450,347。快速衰减对照表现不佳后，正式修正版改为沿用原150轮调度时间尺度，实际训练仍上限50轮；原本等待中的50轮调度修正版任务已在正式训练前取消。
- 修正版损失为 `CE + SoftDice + 0.1*边界BCE + 0.25*困难像素CE + 0.2*粗分割CE`。困难像素项取有效像素中交叉熵最高的25%，忽略255标签。该项是通用困难样本挖掘思路的独立实现，参考[HRNet官方实现中的OHEM使用](https://github.com/HRNet/HRNet-Semantic-Segmentation)，并非声称发明OHEM或复刻其阈值选择规则。

动机来自原无文本模型的验证误差：71.594%的错误发生在标注边界附近5像素内，而该区域仅占全部像素的15.071%；最弱类别obstacles的IoU为75.833%。诊断记录为 `runs/ours_diagnostics/error_analysis.json`。这是验证集调优依据，不是额外测试集证据。

两个新版本的数据增强、裁剪、滑窗、无TTA评估和预训练均相同；组合实验同时修改结构及损失，只能评价组合效果。旧模型及其他方法的结果保留，不能将调度器变化的收益归因于结构创新。

```powershell
python -u -m experiments.run_suite --models ours_sched50 --stages 50 --max-epochs 50 --horizon 50 --checkpoint-every 5
python -u -m experiments.run_suite --models ours_refine --stages 50 --max-epochs 50 --horizon 150 --checkpoint-every 5
python -m experiments.report_ours_improvement
```

队列已运行时不要再次启动上述命令；结果说明保存在 `runs/OURS_IMPROVEMENT.md`。

另作无需训练的四向翻转集成推理检查，输出仅写入 `evaluation_tta4.json`，不覆盖普通评估或混入原对比表。命令为 `python -m experiments.evaluate_ours_tta --model ours_no_text`，前向次数约为普通推理4倍。
单张图启用该可选策略可使用 `python -m experiments.predict --checkpoint runs/ours_no_text_seed42/best.pt --image ../Drone/classes_dataset/classes_dataset/val_original/476.png --output runs/prediction_476_tta.png --tta4`；默认不加该参数仍为普通推理。
还检查了原无文本模型在原始分辨率直接整图推理的表现（`python -m experiments.evaluate_ours_fullframe`）。此项不使用滑窗或TTA，结果单独记录在 `evaluation_fullframe.json`；未改变普通评估协议。后续可用 `python -m experiments.plot_ours_comparison` 导出预先固定样例的预测对照。
# 原始细粒度标签补充实验（2026-09-23）

用户授权四个方法DGFF-Net（ours_no_text）、AFENet、LOGCAN++、D2LS重新在原始细粒度标签上训练；确认使用960×736，沿用原320/80图像划分。旧五类结果不覆盖、不重跑。

原始路径为`../Drone/semantic_drone_dataset/semantic_drone_dataset/label_images_semantic`，400张6000×4000单通道索引标签；类别定义直接取本地`../Drone/colormaps.xlsx`。24个ID中，0(unlabeled)和23(conflicting)设为255忽略，其余ID1–22映射为连续22类，不做语义合并。标签使用最近邻缩放，RGB复用旧划分中现有的960×736图像。全部400张标签通过范围和空间对应检查，训练/验证均包含全部22类；缩放后忽略像素占比约0.250%/0.242%。原始标签重新缩放与旧五类标签合并后的最低一致率为98.143%，边界采样并非逐像素完全相同，已记录每张差异。

准备目录：`.cache/semantic22_960x736`；实验输出：`runs/semantic22_960x736`。数据审计、类别计数、标签哈希、RGB对应检查及固定划分存于`dataset_audit.json`。四方法从各自已有骨干预训练权重开始，分类头重新训练，不从五类分割最佳权重微调；seed42，最多50轮，batch2/有效8，512训练裁剪，512/384滑窗，原有优化器和150轮调度时间尺度，AMP，完整checkpoint每5轮，最佳权重改善时保存，保留上一个版本。所有22类等权计算mIoU和macro-F1；PixAcc忽略255。初始AMP梯度溢出由GradScaler正常回退，冒烟检查要求至少一次有限梯度实际更新。

命令：`python -B -u -m experiments.prepare_semantic`；`python -B -u -m experiments.smoke_semantic`；`python -B -u -m experiments.semantic_suite`。报告更新：`python -B -m experiments.semantic_suite --report-only`。`runs/semantic22_960x736/STOP_QUEUE`阻止后续模型，当前模型目录的`STOP`在当前轮结束后保存并停止；恢复前检查状态及移走相应STOP标记。共享`runs/queue.lock`防止与其他训练队列竞争GPU。用户要求启动后结束本轮，不主动轮询或创建定时监控。
