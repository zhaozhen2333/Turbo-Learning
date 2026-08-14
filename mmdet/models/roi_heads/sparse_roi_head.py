# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from ..utils.misc import empty_instances, unpack_gt_instances
from .cascade_roi_head import CascadeRoIHead
import numpy as np
import torch.nn.functional as F

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@MODELS.register_module()
class SparseRoIHead(CascadeRoIHead):
    r"""The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_
    and `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (:obj:`ConfigDict` or dict): Config of box
            roi extractor.
        mask_roi_extractor (:obj:`ConfigDict` or dict): Config of mask
            roi extractor.
        bbox_head (:obj:`ConfigDict` or dict): Config of box head.
        mask_head (:obj:`ConfigDict` or dict): Config of mask head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in train stage. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in test stage. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_stages: int = 6,
                 stage_loss_weights: Tuple[float] = (1, 1, 1, 1, 1, 1),
                 proposal_feature_channel: int = 256,
                 bbox_roi_extractor: ConfigType = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor: OptConfigType = None,
                 bbox_head: ConfigType = dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN and QueryInst only support `PseudoSampler`'

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  results_list: InstanceList, object_feats: Tensor,
                  batch_img_metas: List[dict],
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            results_list (List[:obj:`InstanceData`]) : List of region
                proposals.
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

            - `cls_score` (Tensor): Classification scores.
            - `bbox_pred` (Tensor): Box energies / deltas.
            - `bbox_feats` (Tensor): Extract bbox RoI features.
            - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        proposal_list = [res.bboxes for res in results_list]
        rois = bbox2roi(proposal_list)
        bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                          batch_img_metas)
        imgs_whwh = torch.cat(
            [res.imgs_whwh[None, ...] for res in results_list])
        cls_pred_list = bbox_results['detached_cls_scores']
        proposal_list = bbox_results['detached_proposals']

        sampling_results = []
        bbox_head = self.bbox_head[stage]
        for i in range(len(batch_img_metas)):
            pred_instances = InstanceData()
            # TODO: Enhance the logic
            pred_instances.bboxes = proposal_list[i]  # for assinger
            pred_instances.scores = cls_pred_list[i]
            pred_instances.priors = proposal_list[i]  # for sampler

            assign_result = self.bbox_assigner[stage].assign(
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                gt_instances_ignore=None,
                img_meta=batch_img_metas[i])

            sampling_result = self.bbox_sampler[stage].sample(
                assign_result, pred_instances, batch_gt_instances[i])
            sampling_results.append(sampling_result)

        bbox_results.update(sampling_results=sampling_results)

        cls_score = bbox_results['cls_score']
        decoded_bboxes = bbox_results['decoded_bboxes']
        cls_score = cls_score.view(-1, cls_score.size(-1))
        decoded_bboxes = decoded_bboxes.view(-1, 4)
        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score,
            decoded_bboxes,
            sampling_results,
            self.train_cfg[stage],
            imgs_whwh=imgs_whwh,
            concat=True)
        bbox_results.update(bbox_loss_and_target)

        # propose for the new proposal_list
        proposal_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            results.imgs_whwh = results_list[idx].imgs_whwh
            results.bboxes = bbox_results['detached_proposals'][idx]
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results

    def _bbox_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor,
                      object_feats: Tensor,
                      batch_img_metas: List[dict]) -> dict:
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
                Each dimension means (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
            Containing the following results:

            - cls_score (Tensor): The score of each class, has
              shape (batch_size, num_proposals, num_classes)
              when use focal loss or
              (batch_size, num_proposals, num_classes+1)
              otherwise.
            - decoded_bboxes (Tensor): The regression results
              with shape (batch_size, num_proposal, 4).
              The last dimension 4 represents
              [tl_x, tl_y, br_x, br_y].
            - object_feats (Tensor): The object feature extracted
              from current stage
            - detached_cls_scores (list[Tensor]): The detached
              classification results, length is batch_size, and
              each tensor has shape (num_proposal, num_classes).
            - detached_proposals (list[tensor]): The detached
              regression results, length is batch_size, and each
              tensor has shape (num_proposal, 4). The last
              dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(batch_img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats)

        fake_bbox_results = dict(
            rois=rois,
            bbox_targets=(rois.new_zeros(len(rois), dtype=torch.long), None),
            bbox_pred=bbox_pred.view(-1, bbox_pred.size(-1)),
            cls_score=cls_score.view(-1, cls_score.size(-1)))
        fake_sampling_results = [
            InstanceData(pos_is_gt=rois.new_zeros(object_feats.size(1)))
            for _ in range(len(batch_img_metas))
        ]

        results_list = bbox_head.refine_bboxes(
            sampling_results=fake_sampling_results,
            bbox_results=fake_bbox_results,
            batch_img_metas=batch_img_metas)
        proposal_list = [res.bboxes for res in results_list]
        bbox_results = dict(
            cls_score=cls_score,
            decoded_bboxes=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detached_cls_scores=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detached_proposals=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor,
                      attn_feats) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            attn_feats (Tensot): Intermediate feature get from the last
                diihead, has shape
                (batch_size*num_proposals, feature_dimensions)

        Returns:
            dict: Usually returns a dictionary with keys:

            - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_preds = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def mask_loss(self, stage: int, x: Tuple[Tensor], bbox_results: dict,
                  batch_gt_instances: InstanceList,
                  rcnn_train_cfg: ConfigDict) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            bbox_results (dict): Results obtained from `bbox_loss`.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: Usually returns a dictionary with keys:

            - `mask_preds` (Tensor): Mask prediction.
            - `loss_mask` (dict): A dictionary of mask loss components.
        """
        attn_feats = bbox_results['attn_feats']
        sampling_results = bbox_results['sampling_results']

        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])

        attn_feats = torch.cat([
            feats[res.pos_inds]
            for (feats, res) in zip(attn_feats, sampling_results)
        ])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_loss_and_target = self.mask_head[stage].loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        mask_results.update(mask_loss_and_target)

        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: a dictionary of loss components of all stage.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        results_list = rpn_results_list
        losses = {}
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                object_feats=object_feats,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            if self.with_mask:
                mask_results = self.mask_loss(
                    stage=stage,
                    x=x,
                    bbox_results=bbox_results,
                    batch_gt_instances=batch_gt_instances,
                    rcnn_train_cfg=self.train_cfg[stage])

                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            object_feats = bbox_results['object_feats']
            results_list = bbox_results['results_list']
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x(tuple[Tensor]): Feature maps of all scale level.
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
        proposal_list = [res.bboxes for res in rpn_results_list]
        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              batch_img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detached_proposals']

        num_classes = self.bbox_head[-1].num_classes

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        topk_inds_list = []
        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = proposal_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)
            if rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 4) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)
        if self.with_mask:
            for img_id in range(len(batch_img_metas)):
                # add positive information in InstanceData to predict
                # mask results in `mask_head`.
                proposals = bbox_results['detached_proposals'][img_id]
                topk_inds = topk_inds_list[img_id]
                attn_feats = bbox_results['attn_feats'][img_id]

                results_list[img_id].proposals = proposals
                results_list[img_id].topk_inds = topk_inds
                results_list[img_id].attn_feats = attn_feats
        return results_list

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
                each image. Each item usually contains following keys:

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - proposal (Tensor): Bboxes predicted from bbox_head,
                  has a shape (num_instances, 4).
                - topk_inds (Tensor): Topk indices of each image, has
                  shape (num_instances, )
                - attn_feats (Tensor): Intermediate feature get from the last
                  diihead, has shape (num_instances, feature_dimensions)

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
        proposal_list = [res.pop('proposals') for res in results_list]
        topk_inds_list = [res.pop('topk_inds') for res in results_list]
        attn_feats = torch.cat(
            [res.pop('attn_feats')[None, ...] for res in results_list])

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        last_stage = self.num_stages - 1
        mask_results = self._mask_forward(last_stage, x, rois, attn_feats)

        num_imgs = len(batch_img_metas)
        mask_results['mask_preds'] = mask_results['mask_preds'].reshape(
            num_imgs, -1, *mask_results['mask_preds'].size()[1:])
        num_classes = self.bbox_head[-1].num_classes

        mask_preds = []
        for img_id in range(num_imgs):
            topk_inds = topk_inds_list[img_id]
            masks_per_img = mask_results['mask_preds'][img_id].flatten(
                0, 1)[topk_inds]
            masks_per_img = masks_per_img[:, None,
                                          ...].repeat(1, num_classes, 1, 1)
            mask_preds.append(masks_per_img)
        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds,
            results_list,
            batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)

        return results_list

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

        proposal_list = [res.bboxes for res in rpn_results_list]
        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              batch_img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detached_proposals']

        num_classes = self.bbox_head[-1].num_classes

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        topk_inds_list = []
        results_list = []

        proposal_list_new = []
        topk_inds_list_new = []
        attn_list = []

        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = proposal_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)
            if bbox_rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 4) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)

            proposal_list_new.append(bbox_results['detached_proposals'][img_id])
            topk_inds_list_new.append(topk_inds_list[img_id])
            attn_list.append(bbox_results['attn_feats'][img_id][None, ...])

        attn_feats = torch.cat(attn_list)
        topk_inds_list = topk_inds_list_new
        proposal_list = proposal_list_new

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        last_stage = self.num_stages - 1
        mask_results = self._mask_forward(last_stage, x, rois, attn_feats)

        num_imgs = len(batch_img_metas)
        mask_results['mask_preds'] = mask_results['mask_preds'].reshape(
            num_imgs, -1, *mask_results['mask_preds'].size()[1:])
        num_classes = self.bbox_head[-1].num_classes

        mask_preds = []
        for img_id in range(num_imgs):
            topk_inds = topk_inds_list[img_id]
            masks_per_img = mask_results['mask_preds'][img_id].flatten(
                0, 1)[topk_inds]
            masks_per_img = masks_per_img[:, None,
                                        ...].repeat(1, num_classes, 1, 1)
            mask_preds.append(masks_per_img)

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
                det_label = results.labels
                scores = results.scores
                img_h, img_w = img_meta['img_shape'][:2]

                N = len(mask_pred_ori)
                mask_pred = mask_pred_ori[range(N), det_label][:, None]  # [13, 1, 28, 28]

                # # # # maskness.
                mask_pred = mask_pred.sigmoid()  # (N, 1, h, w)

                # # seg_masks = (mask_pred.squeeze(1) >= 0.20).to(dtype=torch.float32)  # (N, h, w)
                # # sum_masks = seg_masks.sum((1, 2)).float() + 1e-6
                # # seg_scores_0 = (mask_pred.squeeze(1) * seg_masks.float()).sum((1, 2)) / sum_masks
                # # print("seg_scores_0: ", seg_scores_0)
                # seg_masks = (mask_pred.squeeze(1) >= 0.15).to(dtype=torch.float32)  # (N, h, w)
                # sum_masks = seg_masks.sum((1, 2)).float() + 1e-6
                # seg_scores = (mask_pred.squeeze(1) * seg_masks.float()).sum((1, 2)) / sum_masks
                # # print("seg_scores_1: ", (seg_scores*0.65 + 0.45) / seg_scores_0)
                # # print("seg_scores_1: ", torch.mean(((seg_scores_0 - 1) / (seg_scores - 1)), dim=0))

                # if torch.isnan(seg_scores).any():
                #     inds = torch.where(torch.isnan(seg_scores))
                #     seg_scores[inds] = 0
                # mask_score = scores * seg_scores  # 0.1
                # # mask_score_0 = scores * seg_scores_0  # 0.2
                # scores = mask_score * -0.20 + scores * 1.20
                # # scores = mask_score_0 * 1.2 - scores * 0.2

                # # print("seg_scores_1: ", torch.mean((scores - scores_0), dim=0))
                # # scores = scores * 0.03 + mask_score * 0.97

                # # scores = mask_score
                # results.scores = scores
                # results.mask_scores = mask_score

                bboxes_ = self.mask_processing(
                    mask_pred, bboxes, img_h, img_w, threshold=0.31) # 0.23

                bboxes_[:, [0, 2]].clamp_(min=0, max=img_w)
                bboxes_[:, [1, 3]].clamp_(min=0, max=img_h)
                refine_bboxes = (bboxes_ + bboxes) / 2
                # bboxes = bboxes_
                results.bboxes = bboxes
                results.refine_bboxes = refine_bboxes
                # print("bboxes: ", bboxes.shape)
                # print("bboxes: ", bboxes)
                # print("refine_bboxes: ", refine_bboxes.shape)
                # print("refine_bboxes: ", refine_bboxes)

                boxes_prev = results.bboxes        # stage 1
                boxes_curr = results.refine_bboxes # stage 3

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


                # im_mask = self.mask_head[-1]._predict_by_feat_single(
                #     mask_preds=mask_pred_ori,
                #     bboxes=bboxes,
                #     labels=results.labels,
                #     img_meta=img_meta,
                #     rcnn_test_cfg=self.test_cfg,
                #     rescale=rescale)


                scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                    (1, 2))
                img_h, img_w = img_meta['ori_shape'][:2]
                device = bboxes.device

                mask_preds = mask_pred_ori.sigmoid()

                refine_bboxes_ori = refine_bboxes.clone()
                refine_bboxes_paste = refine_bboxes_ori / scale_factor

                bboxes_ori = bboxes.clone()
                bboxes_paste = bboxes_ori / scale_factor

                N = len(mask_preds)

                num_chunks = int(
                    np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                            GPU_MEM_LIMIT))
                assert (num_chunks <=
                        N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
                chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

                threshold = self.test_cfg.mask_thr_binary
                im_mask = torch.zeros(
                    N,
                    img_h,
                    img_w,
                    device=device,
                    dtype=torch.bool if threshold >= 0 else torch.uint8)

                mask_preds = mask_preds[range(N), results.labels][:, None]

                for inds in chunks:
                    masks_chunk, spatial_inds = _do_paste_mask(
                        mask_preds[inds],
                        bboxes_paste[inds],
                        img_h,
                        img_w,
                        skip_empty=device.type == 'cpu')

                    if threshold >= 0:
                        masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
                    else:
                        # for visualization and debugging
                        masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

                    im_mask[(inds, ) + spatial_inds] = masks_chunk

                results.masks = im_mask
                results.bboxes = refine_bboxes_paste
                # results.bboxes = bboxes_paste

        return results_list

    # TODO: Need to refactor later
    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        all_stage_bbox_results = []
        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        results_list = rpn_results_list
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self.bbox_loss(
                    stage=stage,
                    x=x,
                    results_list=results_list,
                    object_feats=object_feats,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances=batch_gt_instances)
                bbox_results.pop('loss_bbox')
                # torch.jit does not support obj:SamplingResult
                bbox_results.pop('results_list')
                bbox_res = bbox_results.copy()
                bbox_res.pop('sampling_results')
                all_stage_bbox_results.append((bbox_res, ))

                if self.with_mask:
                    attn_feats = bbox_results['attn_feats']
                    sampling_results = bbox_results['sampling_results']

                    pos_rois = bbox2roi(
                        [res.pos_priors for res in sampling_results])

                    attn_feats = torch.cat([
                        feats[res.pos_inds]
                        for (feats, res) in zip(attn_feats, sampling_results)
                    ])
                    mask_results = self._mask_forward(stage, x, pos_rois,
                                                      attn_feats)
                    all_stage_bbox_results[-1] += (mask_results, )
        return tuple(all_stage_bbox_results)


def _do_paste_mask(masks: Tensor,
                   boxes: Tensor,
                   img_h: int,
                   img_w: int,
                   skip_empty: bool = True) -> tuple:
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
        is the slice object.

            If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.

            If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


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
