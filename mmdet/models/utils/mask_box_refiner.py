# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class MaskBoxRefiner(BaseModule):
    """Refine detection boxes and scores using instance-mask confidence.

    The module is inference-only and can be shared by different instance
    segmentation heads. It supports masks predicted in a box-relative ROI
    coordinate system and maps their foreground extent back to image space.

    Args:
        box_threshold: Foreground threshold used to derive a tight box.
        score_threshold: Foreground threshold used to compute maskness.
        mask_score_weight: Weight of ``box_score * maskness`` in the final
            detection score. The box-score weight is ``1 - weight``.
        empty_mask_fallback: Keep the input box when a mask has no foreground.
        init_cfg: Initialization config accepted by ``BaseModule``.
    """

    def __init__(self,
                 box_threshold: float = 0.20,
                 score_threshold: float = 0.35,
                 mask_score_weight: float = 0.35,
                 empty_mask_fallback: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if not 0 <= mask_score_weight <= 1:
            raise ValueError('mask_score_weight must be in [0, 1], but got '
                             f'{mask_score_weight}')
        self.box_threshold = box_threshold
        self.score_threshold = score_threshold
        self.mask_score_weight = mask_score_weight
        self.empty_mask_fallback = empty_mask_fallback

    def rescore(self, scores: Tensor,
                mask_probs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse box scores with mean confidence over mask foreground."""
        probs = self._as_single_channel(mask_probs).squeeze(1)
        foreground = (probs >= self.score_threshold).to(probs.dtype)
        foreground_area = foreground.sum((1, 2)).clamp_min(1e-6)
        maskness = (probs * foreground).sum((1, 2)) / foreground_area
        maskness = torch.nan_to_num(maskness, nan=0.0)
        mask_scores = scores * maskness
        fused_scores = (self.mask_score_weight * mask_scores +
                        (1 - self.mask_score_weight) * scores)
        return fused_scores, mask_scores, maskness

    def refine_boxes(self, mask_probs: Tensor, boxes: Tensor,
                     img_shape: Tuple[int, int]) -> Tensor:
        """Map ROI masks to image space and return their tight boxes."""
        mask_probs = self._as_single_channel(mask_probs)
        img_h, img_w = img_shape[:2]
        device = mask_probs.device
        num_masks = mask_probs.shape[0]
        if num_masks == 0:
            return boxes.clone()

        x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
        img_y = torch.arange(
            0, img_h, device=device, dtype=torch.float32) + 0.5
        img_x = torch.arange(
            0, img_w, device=device, dtype=torch.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        img_x = torch.nan_to_num(img_x, nan=0.0, posinf=0.0, neginf=0.0)
        img_y = torch.nan_to_num(img_y, nan=0.0, posinf=0.0, neginf=0.0)

        grid_x = img_x[:, None, :].expand(num_masks, img_h, img_w)
        grid_y = img_y[:, :, None].expand(num_masks, img_h, img_w)
        grid = torch.stack([grid_x, grid_y], dim=3)
        image_masks = F.grid_sample(
            mask_probs.float(), grid, align_corners=False).squeeze(1)
        foreground = image_masks >= self.box_threshold
        x_any = foreground.any(dim=1)
        y_any = foreground.any(dim=2)

        refined = boxes.clone() if self.empty_mask_fallback else boxes.new_zeros(
            (num_masks, 4))
        for index in range(num_masks):
            xs = torch.where(x_any[index])[0]
            ys = torch.where(y_any[index])[0]
            if xs.numel() and ys.numel():
                refined[index] = torch.stack(
                    (xs[0], ys[0], xs[-1] + 1, ys[-1] + 1)).to(boxes)

        refined[:, [0, 2]].clamp_(0, img_w)
        refined[:, [1, 3]].clamp_(0, img_h)
        return refined

    def forward(self,
                mask_probs: Tensor,
                boxes: Tensor,
                img_shape: Tuple[int, int],
                scores: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Run box refinement and optional score refinement."""
        output = dict(
            bboxes=self.refine_boxes(mask_probs, boxes, img_shape))
        if scores is not None:
            fused_scores, mask_scores, maskness = self.rescore(
                scores, mask_probs)
            output.update(
                scores=fused_scores,
                mask_scores=mask_scores,
                maskness=maskness)
        return output

    @staticmethod
    def _as_single_channel(mask_probs: Tensor) -> Tensor:
        if mask_probs.ndim == 3:
            mask_probs = mask_probs[:, None]
        if mask_probs.ndim != 4 or mask_probs.shape[1] != 1:
            raise ValueError('mask_probs must have shape (N, 1, H, W) or '
                             f'(N, H, W), but got {tuple(mask_probs.shape)}')
        return mask_probs
