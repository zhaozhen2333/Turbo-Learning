# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from mmengine.structures import InstanceData
from mmcv.ops.nms import batched_nms


@MODELS.register_module()
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, mask_box_refiner=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_box_refiner = MODELS.build(mask_box_refiner)

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = MODELS.build(mask_head)

    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        return mask_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list

    def predict_(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = False

        proposals = [res.bboxes for res in rpn_results_list]
        num_imgs = len(proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='all',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                mask_thr_binary=0.5,
                score_per_cls=self.test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in batch_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in batch_img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in batch_img_metas)

        ms_scores = []
        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)
        bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        ms_scores.append(cls_scores)

        scale_factors = [
            torch.tensor([scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]]).to(rois[0].device)
            for scale_factor in scale_factors
        ]

        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        results_list = []
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        segm_results = []
        mask_scores = []
        for i in range(len(proposals)):
            results = InstanceData()
            # some loss (Seesaw loss..) may have custom activation
            cfg = self.test_cfg
            scores = F.softmax(
                cls_score[i], dim=-1) if cls_score[i] is not None else None
            num_classes = scores.size(1) - 1
            scores = scores[:, :-1]  # (1000, 80)

            det_label = torch.arange(num_classes, dtype=torch.long, device=scores.device)  # (80,)
            det_label = det_label.expand_as(scores)  # (1000, 80)
            scores = scores.reshape(-1)  # 1000*80
            det_label = det_label.reshape(-1)  # 1000*80
            valid_mask = scores > cfg.score_thr  # (N,)
            # multiply score_factor after threshold to preserve more bboxes, improve
            # mAP by 1% for YOLOv3
            # NonZero not supported  in TensorRT
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)  # (N,)

            rois = rois[i]  # (N, 5)
            bboxes = self.bbox_head.bbox_coder.decode(
                rois[..., 1:], bbox_preds[i], max_shape=img_shapes[i])
            bboxes = bboxes.view(bbox_preds[i].size(0), -1, 4)  # (1000, 80, 4)
            bboxes = bboxes.reshape(-1, 4)  # (1000*80, 4)

            bboxes, scores, det_label = bboxes[inds], scores[inds], det_label[inds]  # (N, 4) (N,) (N,)

            rois = bbox2roi([bboxes])
            if rois.shape[0] == 0:
                # There is no proposal in the single image
                return empty_instances([batch_img_metas[i]],
                                       rois.device,
                                       task_type='all',
                                       instance_results=[results],
                                       box_type=self.bbox_head.predict_box_type,
                                       use_box_type=False,
                                       num_classes=self.bbox_head.num_classes,
                                       mask_thr_binary=0.5,
                                       score_per_cls=self.test_cfg is None)[0]
            else:
                _, bboxes = torch.split(rois, (1, 4), dim=1)
                sort_inds = torch.argsort(scores, descending=True)
                bboxes = bboxes[sort_inds, :]
                det_label = det_label[sort_inds]
                scores = scores[sort_inds]

                # # bboxes is torch.Size([140, 4])
                # # scores is torch.Size([140])
                # # labels is torch.Size([140])
                det_bbox, keep = batched_nms(bboxes, scores, det_label, cfg.nms)
                bboxes, scores = torch.split(det_bbox, 4, dim=1)
                scores = scores.squeeze(1)
                # scores = scores[keep]
                det_label = det_label[keep]
                if cfg.max_per_img > 0:
                    bboxes = bboxes[:cfg.max_per_img]
                    det_label = det_label[:cfg.max_per_img]
                    scores = scores[:cfg.max_per_img]

                rois = bbox2roi([bboxes])
                if rois.shape[0] == 0:
                    # There is no proposal in the single image
                    return empty_instances([batch_img_metas[i]],
                                           rois.device,
                                           task_type='all',
                                           instance_results=[results],
                                           box_type=self.bbox_head.predict_box_type,
                                           use_box_type=False,
                                           num_classes=self.bbox_head.num_classes,
                                           mask_thr_binary=0.5,
                                           score_per_cls=self.test_cfg is None)[0]
                else:
                    mask_roi_extractor = self.mask_roi_extractor
                    mask_head = self.mask_head
                    mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
                    mask_pred_ = mask_head(mask_feats)

                    N = len(mask_pred_)
                    mask_pred = mask_pred_[range(N), det_label][:, None]

                    # maskness.
                    mask_pred = mask_pred.sigmoid()  # (N, 1, h, w)
                    seg_masks = (mask_pred.squeeze(1) >= 0.35).to(dtype=torch.float32)  # (N, h, w)
                    sum_masks = seg_masks.sum((1, 2)).float() + 1e-6
                    seg_scores = (mask_pred.squeeze(1) * seg_masks.float()).sum((1, 2)) / sum_masks
                    if torch.isnan(seg_scores).any():
                        inds = torch.where(torch.isnan(seg_scores))
                        seg_scores[inds] = 0
                    mask_score = scores * seg_scores
                    scores = mask_score * 0.5 + scores * 0.5

                    mask_pred__ = mask_pred
                    bboxes_ = bboxes
                    bboxes_ = self.mask_processing(
                        mask_pred__, bboxes_, img_shapes[i][0], img_shapes[i][1], threshold=0.20)

                    if img_shapes[i] is not None:
                        bboxes_[:, [0, 2]].clamp_(min=0, max=img_shapes[i][1])
                        bboxes_[:, [1, 3]].clamp_(min=0, max=img_shapes[i][0])

                    bboxes = bboxes_
                    mask_rois = bbox2roi([bboxes_])

                    # mask_rois = bbox2roi([bboxes])
                    mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], mask_rois)
                    mask_pred = self.mask_head(mask_feats)  # (N, 80, 28, 28)
                    # mask_pred_[inds] = mask_pred
                    mask_pred_ = mask_pred

                    # segm_result = self.mask_head.get_seg_masks(
                    #     mask_pred_, bboxes, det_label,
                    #     self.test_cfg, ori_shapes[i], scale_factors[i],
                    #     rescale)
                    segm_result = self.mask_head._predict_by_feat_single(
                        mask_pred_, bboxes, det_label, batch_img_metas[i],
                        self.test_cfg,
                        rescale)

                    if rescale and bboxes.size(0) > 0:
                        scale_factor = bboxes.new_tensor(scale_factors[i])
                        bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                            bboxes.size()[0], -1)

                    # det_bbox = torch.cat([bboxes, scores[:, None]], dim=1)
                    mask_score = [mask_score[det_label == i] for i in range(num_classes)]

                    results.bboxes = bboxes
                    results.scores = scores
                    results.labels = det_label
                    results.masks = segm_result

                results_list.append(results)
                # det_bboxes.append(det_bbox)
                # det_labels.append(det_label)
                # segm_results.append(segm_result)
                # mask_scores.append(mask_score)
            # segm_results = list(zip(segm_results, mask_scores))
            #
            # bbox_results = [
            #     bbox2result(det_bboxes[i], det_labels[i],
            #                 self.bbox_head.num_classes)
            #     for i in range(len(det_bboxes))
            # ]
            # return list(zip(bbox_results, segm_results))
        return results_list

        ##########################################################

        # # don't need to consider aug_test.
        # bboxes = [res.bboxes for res in results_list]
        # mask_rois = bbox2roi(bboxes)
        # if mask_rois.shape[0] == 0:
        #     results_list = empty_instances(
        #         batch_img_metas,
        #         mask_rois.device,
        #         task_type='mask',
        #         instance_results=results_list,
        #         mask_thr_binary=self.test_cfg.mask_thr_binary)
        #     return results_list
        #
        # mask_results = self._mask_forward(x, mask_rois)
        # mask_preds = mask_results['mask_preds']
        # # split batch mask prediction back to each image
        # num_mask_rois_per_img = [len(res) for res in results_list]
        # mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        #
        # # TODO: Handle the case where rescale is false
        # results_list = self.mask_head.predict_by_feat(
        #     mask_preds=mask_preds,
        #     results_list=results_list,
        #     batch_img_metas=batch_img_metas,
        #     rcnn_test_cfg=self.test_cfg,
        #     rescale=rescale)
        #
        # return results_list

    def mask_processing(self, mask_pred, rois, img_h, img_w, threshold=0.40):
        # mask_pred = mask_pred.sigmoid()  # (N, 1, h, w)
        device = mask_pred.device
        # img_w = img_shape[1]
        # img_h = img_shape[0]
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
        # rois = torch.round(rois)
        x0, y0, x1, y1 = torch.split(rois, 1, dim=1)  # each is Nx1

        N = mask_pred.shape[0]

        img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
        img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1

        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

        gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
        gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
        grid = torch.stack([gx, gy], dim=3)
        del gx, gy, img_x, img_y
        torch.cuda.empty_cache()

        mask_pred = F.grid_sample(
            mask_pred.to(dtype=torch.float32), grid, align_corners=False).squeeze(1)  # (N, H, W)
        del grid
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------------------------
        mask_pred_decode = (mask_pred >= threshold).to(dtype=torch.bool)  # (N, h, w)
        # mask_pred_decode_ = (mask_pred >= threshold).to(dtype=torch.bool)  # (N, h, w)
        # seg_masks = mask_pred_decode.float()  # (N, h, w)
        # sum_masks = seg_masks.sum((1, 2)).float()

        del mask_pred
        torch.cuda.empty_cache()

        # valid_mask = seg_scores > scores_thr  # (N,)
        # inds = valid_mask.nonzero(as_tuple=False).squeeze(1)  # (N,)
        # mask_pred_decode = mask_pred_decode[inds]
        x_any = torch.any(mask_pred_decode, dim=1)  # (N, 600)
        y_any = torch.any(mask_pred_decode, dim=2)
        del mask_pred_decode
        torch.cuda.empty_cache()
        bbox_change = torch.zeros(N, 4, dtype=torch.float,
                                  device=device)  # list[tensor(x0, y0, x1, y1), ...]
        for idx in range(N):
            x_ = torch.where(x_any[idx, :])[0]
            y_ = torch.where(y_any[idx, :])[0]
            if len(x_) > 0 and len(y_) > 0:
                bbox_change[idx, :] = torch.as_tensor(
                    [x_[0], y_[0], x_[-1] + 1, y_[-1] + 1], dtype=torch.float32
                )
        # rois[inds] = bbox_change
        # bbox_change = (bbox_change + rois) / 2
        # return bbox_change, seg_masks, sum_masks, seg_scores  # (N, H, W)
        # # 获取非零元素的索引
        # x_idx = torch.nonzero(x_any, as_tuple=True)
        # y_idx = torch.nonzero(y_any, as_tuple=True)
        #
        # # 通过切片获取矩阵的最小和最大值
        # x_min = x_idx[1][x_idx[0] == 0].min()
        # x_max = x_idx[1][x_idx[0] == 0].max()
        # y_min = y_idx[1][y_idx[0] == 0].min()
        # y_max = y_idx[1][y_idx[0] == 0].max()
        #
        # # 构建 bbox_change 矩阵
        # bbox_change = torch.zeros(N, 4, dtype=torch.float, device=device)
        # bbox_change[:, 0] = x_min
        # bbox_change[:, 1] = y_min
        # bbox_change[:, 2] = x_max + 1
        # bbox_change[:, 3] = y_max + 1

        # # 获取每一行/列的第一个和最后一个非零元素的索引
        # sum_x_any = torch.cumsum(x_any, dim=1)
        # sum_y_any = torch.cumsum(y_any, dim=1)
        # x_any = sum_x_any * x_any
        # y_any = sum_y_any * y_any
        # x_max = torch.argmax(x_any, dim=1)
        # # x_min = torch.argmax(x_any.flip(1), dim=1)
        # x_min = torch.argmax((x_any != 0).to(torch.float), dim=1)
        # y_max = torch.argmax(y_any, dim=1)
        # # y_min = torch.argmax(y_any.flip(1), dim=1)
        # y_min = torch.argmax((y_any != 0).to(torch.float), dim=1)
        #
        # # 构建 bbox_change 矩阵
        # # bbox_change = torch.stack([x_min, y_min, x_max.flip([0]) + 1, y_max.flip([0]) + 1], dim=1).to(torch.float)
        #
        # bbox_change = torch.stack([x_min, y_min, x_max + 1, y_max + 1], dim=1).to(torch.float)

        return bbox_change  # (N, H, W)

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.mask_box_refiner is None:
            return BaseRoIHead.predict(
                self, x, rpn_results_list, batch_data_samples, rescale)

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False

        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=self.test_cfg is None)
            results_list = empty_instances(
                batch_img_metas,
                rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
        else:
            bbox_results = self._bbox_forward(x, rois)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            bbox_preds = bbox_preds.split(num_proposals_per_img, 0)

            ####
            assert len(cls_scores) == len(bbox_preds)
            results_list = []

            for img_id in range(len(batch_img_metas)):
                img_meta = batch_img_metas[img_id]
                results = self.bbox_head._predict_by_feat_single(
                    roi=rois[img_id],
                    cls_score=cls_scores[img_id],
                    bbox_pred=bbox_preds[img_id],
                    img_meta=img_meta,
                    rescale=bbox_rescale,
                    rcnn_test_cfg=self.test_cfg)

                results_list.append(results)
            ####

            # don't need to consider aug_test.
            bboxes = [res.bboxes for res in results_list]
            mask_rois = bbox2roi(bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_preds = mask_results['mask_preds']
            # split batch mask prediction back to each image
            num_mask_rois_per_img = [len(res) for res in results_list]
            mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

            # TODO: Handle the case where rescale is false
            ###
            assert len(mask_preds) == len(results_list) == len(batch_img_metas)

            for img_id in range(len(batch_img_metas)):
                img_meta = batch_img_metas[img_id]
                results = results_list[img_id]
                bboxes = results.bboxes
                if bboxes.shape[0] == 0:
                    results_list[img_id] = empty_instances(
                        [img_meta],
                        bboxes.device,
                        task_type='mask',
                        instance_results=[results],
                        mask_thr_binary=self.test_cfg.mask_thr_binary)[0]
                else:
                    mask_pred_ori = mask_preds[img_id]
                    mask_pred_ori = bboxes.new_tensor(mask_pred_ori)
                    # print("mask_pred_ori: ", mask_pred_ori.shape)
                    det_label = results.labels
                    scores = results.scores
                    img_h, img_w = img_meta['img_shape'][:2]

                    N = len(mask_pred_ori)
                    mask_pred = mask_pred_ori[range(N), det_label][:, None]  # [13, 1, 28, 28]

                    # # # maskness.
                    mask_pred = mask_pred.sigmoid()  # (N, 1, h, w)

                    refinement = self.mask_box_refiner(
                        mask_pred, bboxes, (img_h, img_w), scores)
                    results.scores = refinement['scores']
                    results.mask_scores = refinement['mask_scores']
                    bboxes_ = refinement['bboxes']

                    ##################
                    boxes_prev = bboxes        # stage 1
                    boxes_curr = bboxes_ # stage 3

                    if boxes_prev.numel() > 0:
                        # ΔIoU
                        ious = bbox_iou_xyxy(boxes_prev, boxes_curr)
                        mean_iou = ious.mean().item()

                        # ΔB (L2 norm)
                        delta_b = torch.norm(boxes_curr - boxes_prev, dim=1)
                        mean_delta_b = delta_b.mean().item()

                        N = len(results.bboxes)

                        results.mean_iou_s1_s3 = boxes_prev.new_full((N,), mean_iou)
                        results.mean_delta_b_s1_s3 = boxes_prev.new_full((N,), mean_delta_b)

                    # bboxes = (bboxes_ + bboxes) / 2
                    bboxes = bboxes_
                    results.bboxes = bboxes

                    mask_rois = bbox2roi([bboxes])
                    mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], mask_rois)
                    mask_pred = self.mask_head(mask_feats)  # (N, 80, 28, 28)
                    # print("mask_pred: ", mask_pred.shape)

                    # stage 5
                    mask_pred_stage_5 = mask_pred
                    N = len(mask_pred_stage_5)
                    mask_pred_5 = mask_pred_stage_5[range(N), det_label][:, None]  # [13, 1, 28, 28]
                    # print("mask_pred_5: ", mask_pred_5.shape)
                    # # # maskness.
                    mask_pred_5 = mask_pred_5.sigmoid()  # (N, 1, h, w)
                    bboxes_ = self.mask_box_refiner.refine_boxes(
                        mask_pred_5, bboxes, (img_h, img_w))

                    ##################
                    boxes_prev = bboxes        # stage 1
                    boxes_curr = bboxes_ # stage 3

                    if boxes_prev.numel() > 0:
                        # ΔIoU
                        ious = bbox_iou_xyxy(boxes_prev, boxes_curr)
                        mean_iou = ious.mean().item()

                        # ΔB (L2 norm)
                        delta_b = torch.norm(boxes_curr - boxes_prev, dim=1)
                        mean_delta_b = delta_b.mean().item()

                        N = len(results.bboxes)

                        results.mean_iou_s3_s5 = boxes_prev.new_full((N,), mean_iou)
                        results.mean_delta_b_s3_s5 = boxes_prev.new_full((N,), mean_delta_b)

                    bboxes = bboxes_
                    results.bboxes = bboxes
                    im_mask = self.mask_head._predict_by_feat_single(
                        mask_preds=mask_pred,
                        bboxes=bboxes,
                        labels=results.labels,
                        img_meta=img_meta,
                        rcnn_test_cfg=self.test_cfg,
                        rescale=rescale)
                    results.masks = im_mask
            ###

        return results_list

def bbox_iou_xyxy(boxes1, boxes2, eps=1e-6):
    """
    boxes1, boxes2: Tensor [N, 4] in xyxy
    return: Tensor [N] IoU for each matched pair
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1 + area2 - inter + eps
    return inter / union
