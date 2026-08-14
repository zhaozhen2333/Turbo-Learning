<div align="center">

# A Turbo-Inference Strategy for Object Detection and Instance Segmentation

### 检测与实例分割之间的免训练迭代优化

[Zhen Zhao](https://github.com/zhaozhen2333) · Gang Zhang · Xiaolin Hu · Liang Tang

北京林业大学 · 清华大学 · 中国脑科学研究院

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.3.0-4B8BBE.svg)](https://github.com/open-mmlab/mmdetection)
[![arXiv](https://img.shields.io/badge/arXiv-2606.12371-b31b1b.svg)](https://arxiv.org/abs/2606.12371)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cviu.2026.104827-blue.svg)](https://doi.org/10.1016/j.cviu.2026.104827)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[论文](https://doi.org/10.1016/j.cviu.2026.104827) · [arXiv](https://arxiv.org/abs/2606.12371) · [PDF](https://arxiv.org/pdf/2606.12371) · [代码](https://github.com/zhaozhen2333/Turbo-Learning)

> **官方实现**：“A Turbo-Inference Strategy for Object Detection and Instance Segmentation”，发表于 *Computer Vision and Image Understanding*（CVIU），2026。

[English](README.md) | **简体中文**

</div>

## Overview

传统的 top-down 实例分割通常采用单向的“先检测、后分割”流程。Turbo-Inference 将它改造成闭环：利用粗粒度实例 mask 中的像素级定位信息和质量信息反向优化检测框与分类分数，再用优化后的检测框重新预测更准确的 mask。

该方法**无需重新训练**，直接复用预训练 mask head，不改变原模型的训练过程即可接入现有 top-down 实例分割方法。

<p align="center">
  <img src="resources/turbo_inference/Turbo.jpg" width="100%" alt="Turbo-Inference 总览">
</p>

反馈路径包含三个主要操作：

1. **检测框优化（Box Refinement）**：将 RoI mask 映射回图像坐标，并根据前景区域生成更紧致的检测框。
2. **Maskness 重打分**：根据 mask 前景概率分布估计 mask 质量，并与原始分类分数融合。
3. **Turbo 分割**：根据优化后的检测框重新提取 RoI 特征，复用原始 mask head 获得更准确的 mask。

```text
检测 → 粗 mask → 优化检测框与分数 → 精细 mask
                 ↖__________________|
```

## 主要特点

- **无需训练**：所有优化只发生在推理阶段。
- **即插即用**：通过 RoI head 配置共享的 `MaskBoxRefiner` 模块。
- **联合提升**：分割结果反向帮助检测，同时提高 bbox AP 和 mask AP。
- **适用范围广**：提供 Mask R-CNN、Cascade Mask R-CNN/HTC、QueryInst/Sparse R-CNN 和 RTMDet-Ins 的实验路径。
- **结果可复现**：发布的 Mask R-CNN 实现已在完整的 5,000 张 COCO 2017 验证集图像上重新评测。

## 可视化结果

<p align="center">
  <img src="resources/turbo_inference/coco_intro.jpg" width="92%" alt="与原始 Mask R-CNN 的对比">
</p>

Turbo-Inference 能够生成更紧致的检测框、抑制低质量重复预测，并改善实例 mask。

<p align="center">
  <img src="resources/turbo_inference/coco_result.jpg" width="100%" alt="COCO 可视化结果">
</p>

## 实验结果

### COCO 主要结果

Turbo-Inference 在两阶段、级联、基于 query 和单阶段实例分割框架上均能带来稳定提升。下面的精简表展示论文中的主要结果；FPS 使用单张 RTX 2080 Ti、batch size 2 测量。

| 方法 | Backbone | bbox AP | bbox AP + Turbo | segm AP | segm AP + Turbo | FPS → Turbo FPS |
|:--|:--|--:|--:|--:|--:|--:|
| Mask R-CNN | R50-FPN | 39.2 | **40.3 (+1.1)** | 35.4 | **36.7 (+1.3)** | 15.7 → 12.0 |
| HTC | R50-FPN | 43.3 | **43.7 (+0.4)** | 38.3 | **39.2 (+0.9)** | 5.5 → 4.5 |
| RTMDet-m | CSPX-PAFPN | 48.8 | **49.3 (+0.5)** | 42.1 | **42.4 (+0.3)** | 3.7 → 2.7 |
| QueryInst | R50-FPN | 42.0 | **42.8 (+0.8)** | 37.5 | **38.7 (+1.2)** | 7.5 → 6.0 |

<details>
<summary><b>更多 Backbone 和模型规模的结果</b></summary>

| 方法 | Backbone | bbox AP | bbox AP + Turbo | segm AP | segm AP + Turbo | FPS → Turbo FPS |
|:--|:--|--:|--:|--:|--:|--:|
| Mask R-CNN | R101-FPN | 40.8 | **41.8** | 36.6 | **37.9** | 13.5 → 9.8 |
| Mask R-CNN | X101-FPN | 42.8 | **43.9** | 38.4 | **39.7** | 6.8 → 5.5 |
| Mask R-CNN | ConvNeXt-T-FPN | 46.2 | **47.5** | 41.7 | **42.8** | 14.5 → 10.8 |
| Mask R-CNN | Swin-T | 46.0 | **47.5** | 41.7 | **42.9** | 9.8 → 7.4 |
| Mask R-CNN | Swin-S | 48.2 | **49.4** | 43.2 | **44.5** | 9.3 → 6.3 |
| Mask R-CNN | ViT-B | 51.5 | **52.0** | 45.7 | **46.8** | 3.9 → 3.3 |
| Mask R-CNN | ConvNeXt-v2-FPN | 52.9 | **53.4** | 46.4 | **47.6** | 5.4 → 5.2 |
| HTC | R101-FPN | 44.8 | **45.1** | 39.6 | **40.5** | 5.2 → 4.3 |
| RTMDet-l | CSPX-PAFPN | 51.1 | **51.4** | 43.7 | **44.0** | 3.4 → 2.5 |
| RTMDet-x | CSPX-PAFPN | 52.4 | **52.7** | 44.6 | **44.9** | 2.8 → 2.2 |
| QueryInst | R101-FPN | 49.0 | **49.8** | 42.9 | **44.1** | 3.5 → 2.5 |

</details>

这些结果说明，该反馈策略不依赖特定的检测器、mask head 或 backbone。额外的优化阶段会以一定的推理速度为代价换取精度提升。

### 与 Training-free 方法的兼容性

下表所有结果均使用同一份公开的 Mask R-CNN R50-FPN 2x 权重，在完整的 5,000 张 COCO 2017 验证集图像上重新评测。Turbo-Inference 和 Soft-NMS 都不需要重新训练。

| 方法 | Turbo-Inference | Soft-NMS | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
|:--|:--:|:--:|--:|--:|--:|--:|--:|--:|
| 原始 Mask R-CNN |  |  | 39.993 | 59.556 | 43.577 | 35.175 | 56.346 | 37.719 |
| Mask R-CNN + Soft-NMS |  | ✓ | 40.618 | **59.600** | 44.620 | 35.423 | 56.371 | 38.126 |
| Mask R-CNN + Turbo | ✓ |  | 40.279 | 59.482 | 43.901 | 36.400 | **56.498** | 39.277 |
| Mask R-CNN + Turbo + Soft-NMS | ✓ | ✓ | **40.901** | 59.510 | **44.945** | **36.643** | 56.487 | **39.688** |

#### 可叠加的 Training-free 优化

Turbo-Inference 和 Soft-NMS 作用于预测流程中的不同信息。Soft-NMS 根据检测框之间的重叠关系调整分数，而 Turbo-Inference 利用实例 mask 的反馈优化检测框、分数和 mask，因此二者的收益具有互补性：

- 单独使用 Turbo-Inference，相比原始推理流程提升 **0.286 bbox AP** 和 **1.225 segm AP**。
- 在 Turbo-Inference 上继续加入 Soft-NMS，可再提升 **0.622 bbox AP** 和 **0.243 segm AP**。
- 两种 training-free 方法组合后达到 **40.901 bbox AP** 和 **36.643 segm AP**，相比原始模型分别提升 **0.908** 和 **1.468**。

这说明 Turbo-Inference 可以在不更新预训练模型权重的情况下，与其他兼容的 training-free 优化或后处理方法叠加，从而获得更好的结果。

验证所用 checkpoint 的 SHA-256 为：

```text
3e542a40ccc2952293c56bddc05e2002d681b9f6f20fde01f5908a5540b582b3
```

> Turbo-Inference 通过增加检测头和分割头计算换取精度。Backbone 不会重复计算，但优化后的 RoI 会再次经过 mask head。

## 安装

本仓库基于 MMDetection 3.3.0。典型安装方式如下：

```bash
conda create -n turbo-inference python=3.8 -y
conda activate turbo-inference

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
pip install -v -e .
```

如果 CUDA 或 PyTorch 版本不同，请参考 [MMDetection 安装文档](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html)。

## 模型权重

大多数权重超过 GitHub 单文件 100 MB 限制，因此本仓库不直接保存权重文件。请从各模型的原始发布地址下载。

完整复现中默认使用的 Mask R-CNN R50-FPN 权重为：

| 模型 | 配置 | 权重 | SHA-256 |
|:--|:--|:--|:--|
| Mask R-CNN R50-FPN 2x | [配置](configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth) | `3e542a40…b582b3` |

```bash
mkdir -p checkpoints
wget -P checkpoints \
  https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth
```

<details>
<summary><b>实验使用的完整 Checkpoint Zoo</b></summary>

| 模型 | 配置/来源 | 官方权重 |
|:--|:--|:--|
| Cascade Mask R-CNN ConvNeXt-S | [配置](configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth) |
| Cascade Mask R-CNN ConvNeXt-T | [配置](configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth) |
| HTC R101-FPN 20e | [配置](configs/htc/htc_r101_fpn_20e_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth) |
| HTC R50-FPN 1x | [配置](configs/htc/htc_r50_fpn_1x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth) |
| Mask R-CNN ConvNeXt-v2-B | [配置](projects/ConvNeXt-V2/configs/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v3.0/convnextv2/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth) |
| Mask R-CNN ConvNeXt-T | [配置](configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth) |
| Mask R-CNN R101-FPN 2x | [配置](configs/mask_rcnn/mask-rcnn_r101_fpn_2x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth) |
| Mask R-CNN R50-FPN 2x | [配置](configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth) |
| Mask R-CNN Swin-S | [配置](configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth) |
| Mask R-CNN Swin-T | [配置](configs/swin/mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth) |
| Mask R-CNN X101-64x4d-FPN | [配置](configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_1x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) |
| QueryInst R101-FPN，300 proposals | [配置](configs/queryinst/queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth) |
| QueryInst R101-FPN，100 proposals | [配置](configs/queryinst/queryinst_r101_fpn_ms-480-800-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_mstrain_480-800_3x_coco/queryinst_r101_fpn_mstrain_480-800_3x_coco_20210904_104048-91f9995b.pth) |
| QueryInst R50-FPN 1x | [配置](configs/queryinst/queryinst_r50_fpn_1x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth) |
| QueryInst R50-FPN，300 proposals | [配置](configs/queryinst/queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth) |
| QueryInst R50-FPN，100 proposals | [配置](configs/queryinst/queryinst_r50_fpn_ms-480-800-3x_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco/queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643-7837af86.pth) |
| RTMDet-Tiny | [配置](configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py) | [下载](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth) |
| ViTDet Mask R-CNN ViT-B | [配置](projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py) | [下载](https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth) |
| YOLOv12-M Seg | [来源](https://github.com/sunsmarterjie/yolov12) | [下载](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt) |
| YOLOv12-L Seg | [来源](https://github.com/sunsmarterjie/yolov12) | [下载](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt) |
| YOLOv12-X Seg | [来源](https://github.com/sunsmarterjie/yolov12) | [下载](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt) |

</details>

## 快速开始

下载上面的默认权重，然后运行：

```bash
python demo/image_demo.py \
  demo/demo.jpg \
  configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py \
  --weights checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth \
  --device cuda:0 \
  --out-dir outputs/turbo_mask_rcnn
```

单卡 COCO 评测：

```bash
python tools/test.py \
  configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py \
  /path/to/mask_rcnn_r50_fpn_2x_coco.pth
```

8 卡完整回归：

```bash
./tools/dist_test.sh \
  configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py \
  /path/to/mask_rcnn_r50_fpn_2x_coco.pth 8 \
  --work-dir work_dirs/turbo_mask_rcnn_r50_8gpu
```

评测前请按照 MMDetection 的标准目录结构准备 COCO：

```text
data/coco/
├── annotations/instances_val2017.json
└── val2017/
```

## 配置说明

`MaskBoxRefiner` 注册在 [`mmdet/models/utils/mask_box_refiner.py`](mmdet/models/utils/mask_box_refiner.py)，通过 RoI head 配置启用：

```python
roi_head=dict(
    type='StandardRoIHead',
    mask_box_refiner=dict(
        type='MaskBoxRefiner',
        box_threshold=0.20,
        score_threshold=0.35,
        mask_score_weight=0.35,
        empty_mask_fallback=False))
```

| 参数 | 含义 |
|:--|:--|
| `box_threshold` | 从 mask 前景生成紧致框时使用的阈值。 |
| `score_threshold` | 计算 maskness 时使用的前景阈值。 |
| `mask_score_weight` | mask 感知分数在分数融合中的权重。 |
| `empty_mask_fallback` | 当前景区域为空时是否回退到输入检测框。 |

设置 `mask_box_refiner=None` 即可恢复 MMDetection 官方推理流程。提供的 Mask R-CNN R50-FPN 配置还启用了 IoU 阈值为 0.5 的 Soft-NMS。

## 分阶段优化

<p align="center">
  <img src="resources/turbo_inference/stage.jpg" width="88%" alt="Turbo-Inference 分阶段优化">
</p>

Mask R-CNN 和 HTC 默认使用四个阶段：检测、原始分割、Turbo 检测与 Turbo 分割。QueryInst 使用三个阶段，因为再次运行 mask head 可能破坏 proposal feature 与优化后 RoI feature 之间的交互。

## 实现进度

下列所有模型系列均已完整实现 Turbo-Inference，可以直接使用现有预训练权重，无需重新训练。

| 模型系列 | Turbo-Inference |
|:--|:--:|
| Mask R-CNN | ✓ |
| Cascade Mask R-CNN / HTC | ✓ |
| QueryInst / Sparse R-CNN | ✓ |
| RTMDet-Ins | ✓ |

### Roadmap

- [x] Mask R-CNN 的 training-free Turbo-Inference。
- [x] Cascade Mask R-CNN / HTC 的 training-free Turbo-Inference。
- [x] QueryInst / Sparse R-CNN 的 training-free Turbo-Inference。
- [x] RTMDet-Ins 的 training-free Turbo-Inference。
- [x] 与 Soft-NMS 等其他 training-free 优化方法兼容。
- [ ] 在训练阶段引入 Turbo 反馈闭环，学习更强的 Turbo-aware 权重。

当前版本聚焦于使用已有 checkpoint 的 training-free 推理。本次重新评测的 Mask R-CNN Turbo-only 和 Turbo + Soft-NMS 日志分别保存在 `work_dirs/turbo_mask_rcnn_r50_no_softnms_8gpu/` 和 `work_dirs/turbo_mask_rcnn_r50_readme_rerun_8gpu/`。目前唯一剩余的研究方向是在训练阶段加入 Turbo 反馈，从而获得更好的模型权重。本次发布不包含 CondInst 实验。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{zhao2026turboinference,
  title   = {A turbo-inference strategy for object detection and instance segmentation},
  author  = {Zhao, Zhen and Zhang, Gang and Hu, Xiaolin and Tang, Liang},
  journal = {Computer Vision and Image Understanding},
  volume  = {270},
  pages   = {104827},
  year    = {2026},
  doi     = {10.1016/j.cviu.2026.104827}
}
```

预印本：[arXiv:2606.12371](https://arxiv.org/abs/2606.12371)。

## 致谢

本项目基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 实现。感谢 OpenMMLab 社区以及相关实例分割方法的作者。

## 开源协议

代码采用 [Apache 2.0](LICENSE) 开源协议。使用时还需遵循 MMDetection、相关依赖、数据集和预训练模型各自的许可证。
