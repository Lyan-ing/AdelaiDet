# -*- coding: utf-8 -*-
import logging
# import os

from skimage import color

import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.ops import roi_align
from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head_for_disdepth import build_dynamic_mask_head
from .mask_branch import build_mask_branch
# from torchvision.utils import
# torch.save()
from adet.utils.comm import aligned_bilinear

# from torchvision.utils import draw_bounding_boxes, save_image
# from skimage.morphology import erosion, disk

__all__ = ["MyCondInst_Dis_Pseudo_depth"]

logger = logging.getLogger(__name__)


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation, sim=0.1):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]
    if images.size(1) == 1:
        depth_sim = similarity * unfolded_weights
        # import torchshow as ts
        edge0 = depth_sim[0,3]
        edge0 = (edge0 < sim)
        # ts.save(edge0, '/home/yl/python/AdelaiDet1/out/e0.png')
        edge1 = depth_sim[0, 4]
        edge1 = (edge1 < sim)
        # ts.save(edge1, '/home/yl/python/AdelaiDet1/out/e1.png')
        edge = edge0 * edge1
        # ts.save(edge, '/home/yl/python/AdelaiDet1/out/e.png')
        return depth_sim, edge[None]

    else:
        return similarity * unfolded_weights


@META_ARCH_REGISTRY.register()
class MyCondInst_Dis_Pseudo_depth(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        # print("we are CondInst and BoxInst")
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())  # fcos
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self.use_depth = cfg.MODEL.BOXINST.PAIRWISE.USE_DEPTH
        self.edge_thresh = cfg.MODEL.BOXINST.EDGE_THRESH
        self.pred_depth = cfg.MODEL.BOXINST.PAIRWISE.PRED_DEPTH
        self.pred_edge = cfg.MODEL.BOXINST.PRED_EDGE
        self.mask_feature = cfg.MODEL.BOXINST.MASK_FEATURE_PARA
        self.tmp = True  # cfg.MODEL.BOXINST.TMP
        self.max = cfg.INPUT.MAX_SIZE_TEST
        self.min = cfg.INPUT.MIN_SIZE_TEST

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [x["image"].to(self.device) for x in batched_inputs]
        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)  # 图像尺度归一化
        # features = self.backbone(images_norm.tensor)
        depth_prediction = None
        if self.use_depth and self.training:
            depth_maps = [x["depth_map"].to(self.device) for x in batched_inputs]
            depth_maps_norm = ImageList.from_tensors(depth_maps, self.backbone.size_divisibility)

            # depth_map = images_norm.tensor[3:4]
            depth_prediction = F.avg_pool2d(
                depth_maps_norm.tensor.float(), kernel_size=4,
                stride=4, padding=0
            ).unsqueeze(dim=1)
        return images_norm, depth_prediction, original_images

    def forward(self, batched_inputs, branch='student'):
        # original_images = [x["image"].to(self.device) for x in batched_inputs]
        if self.use_depth and self.training and branch == 'student':
            images_norm, depth_prediction, original_images = self.preprocess_image(batched_inputs)
            if 'mask_feature' in batched_inputs[-1]:
                teacher_mask_feature = [x["mask_feature"].to(self.device) for x in batched_inputs]
            features = self.backbone(images_norm.tensor)
        elif self.training and branch == 'teacher' and self.use_depth:
            original_images = [x["image"].to(self.device) for x in batched_inputs]
            original_depths = [x["depth_map"].to(self.device) for x in batched_inputs]
            # normalize images
            images_norm = [self.normalizer(x) for x in original_images]
            depth_sims = []
            for im_idx, (images_single, depth_single) in enumerate(zip(images_norm, original_depths)):
                teacher_size = images_single.shape[-2:]
                min_size = self.min
                max_size = max(teacher_size) * self.min / min(teacher_size)
                if max_size > self.max:
                    max_size = self.max
                    min_size = max_size * min(teacher_size) / max(teacher_size)
                max_size = int(max_size)
                min_size = int(min_size)
                if teacher_size[0] > teacher_size[1]:
                    images_norm[im_idx] = F.interpolate(images_single[None], size=(max_size, min_size),
                                                        mode="bilinear", align_corners=False).squeeze(0)

                else:
                    images_norm[im_idx] = F.interpolate(images_single[None], size=(min_size, max_size),
                                                        mode="bilinear", align_corners=False).squeeze(0)
                depth = depth_single[None, None].float()
                image_masks = torch.ones_like(depth_single)
                image_depth_similarity, _ = get_images_color_similarity(
                    depth, image_masks.float(),
                    self.pairwise_size, self.pairwise_dilation, self.edge_thresh
                )
                depth_sims.append(image_depth_similarity[0, 4])
            original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in images_norm]
            original_image_masks = ImageList.from_tensors(original_image_masks,
                                                          self.backbone.size_divisibility, pad_value=0.0)
            images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)  # 图像尺度归一化

            features = self.backbone(images_norm.tensor)
            depth_prediction = None
        else:
            original_images = [x["image"].to(self.device) for x in batched_inputs]
            # import torchshow as ts
            # ts.save(original_images[0] /255, './immm.png')
            # normalize images
            images_norm = [self.normalizer(x) for x in original_images]
            images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)  # 图像尺度归一化
            features = self.backbone(images_norm.tensor)
            depth_prediction = None

        if "instances" in batched_inputs[0] and branch == 'student':
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.boxinst_enabled:
                original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]

                # mask out the bottom area where the COCO dataset probably has wrong annotations
                for i in range(len(original_image_masks)):
                    im_h = batched_inputs[i]["height"]
                    pixels_removed = int(
                        self.bottom_pixels_removed *
                        float(original_images[i].size(1)) / float(im_h)
                    )
                    if pixels_removed > 0:
                        original_image_masks[i][-pixels_removed:, :] = 0

                original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)
                original_image_masks = ImageList.from_tensors(
                    original_image_masks, self.backbone.size_divisibility, pad_value=0.0
                )
                if self.use_depth and self.training:
                    depth_edges = self.add_bitmasks_from_boxes(
                        gt_instances, original_images.tensor, original_image_masks.tensor,
                        original_images.tensor.size(-2), original_images.tensor.size(-1),
                        depth_prediction=depth_prediction
                    )
                else:
                    _, = self.add_bitmasks_from_boxes(
                        gt_instances, original_images.tensor, original_image_masks.tensor,
                        original_images.tensor.size(-2), original_images.tensor.size(-1),
                        depth_prediction=None
                    )
            else:

                self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses, depth_pred = self.mask_branch(features, gt_instances)
        if self.training:
            mask_feats_attention = mask_feats.pow(2).mean(1)
            mask_feats_attention = [single[:(size[0] + 4) // 8, :(size[1] + 4) // 8] for (single, size) in
                                    zip(mask_feats_attention, original_image_masks.image_sizes)]
        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller, branch=branch
        )  # fcos

        if self.training and branch == 'student':
            mask_losses = self._forward_mask_heads_train(proposals, mask_feats, gt_instances,
                                                         depth_map=depth_prediction)
            # controller, mask feature, gt
            losses = {}
            if self.pred_depth:
                depth_pred_loss = {}
                if self.tmp:
                    depth_predictions = F.avg_pool2d(
                        depth_prediction, kernel_size=2,
                        stride=2, padding=0
                    ) / 255.0
                    depth_pred_loss["loss_depth_pred"] = F.l1_loss(depth_pred,
                                                                   depth_predictions) * self.pred_depth
                else:
                    depth_pred = aligned_bilinear(depth_pred, 2)
                    # depth_edges = depth_edges * 255
                    # if self.pred_depth:
                    if self.pred_edge:
                        depth_pred_loss["loss_depth_pred"] = F.l1_loss(depth_edges.float(),
                                                                       depth_pred) * self.pred_edge
                        # print('edge')
                    else:
                        depth_pred_loss["loss_depth_pred"] = F.l1_loss(depth_prediction / 255.,
                                                                       depth_pred) * self.pred_depth
                    # print('depth')
                # print('111')

                losses.update(depth_pred_loss)
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            loss_mask_feature = {}
            if self.mask_feature:
                if self.mask_feature and 'mask_feature' in batched_inputs[-1]:
                    loss_features = []
                    for (teacher_feature, student_feature) in zip(teacher_mask_feature, mask_feats_attention):
                        student_feature = F.interpolate(student_feature[None, None], size=teacher_feature.size(),
                                                        mode="bilinear", align_corners=False)[0, 0]
                        teacher_feature = F.normalize(teacher_feature.view(1, -1)).view(teacher_feature.shape)
                        student_feature = F.normalize(student_feature.view(1, -1)).view(student_feature.shape)
                        loss_feature = (student_feature - teacher_feature).pow(2).mean() * self.mask_feature * 100
                        loss_features.append(loss_feature)
                    loss_features = torch.stack(loss_features).mean()
                    loss_mask_feature['loss_mask_feature'] = loss_features
                else:
                    # loss_feature = 0.0
                    loss_mask_feature['loss_mask_feature'] = torch.stack([i.sum()*0. for i in mask_feats_attention]).sum()
                losses.update(loss_mask_feature)
            return {}, losses

        elif self.training and branch == 'teacher':
            processed_results = []
            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            for input_per_image, proposal, mask_feat in zip(batched_inputs, proposals, mask_feats):
                pred_instances_w_masks = self._forward_mask_heads_test([proposal], mask_feat[None], branch=branch)

                # padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
                # processed_result = []
                # for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image['image'].shape[-2]
                width = input_per_image['image'].shape[-1]

                instances_per_im = pred_instances_w_masks
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })
            # processed_results.append(processed_result)
            return processed_results, mask_feats_attention, depth_sims
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances, depth_map=None):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)
                # 只保留部分预测实例，保留策略与论文里的中心区域不同？？
                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)
            # print(len(pred_instances.gt_inds))

        pred_instances.mask_head_params = pred_instances.top_feats  # controller weights

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances, depth_map=depth_map
        )  # dynamic_mask_head.__call__

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats, branch='teacher'):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances, branch=branch
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else:  # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w, depth_prediction=None):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]]
        image_masks = image_masks[:, start::stride, start::stride]
        depth_edges = []
        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation
            )
            # from torchvision.utils import save_image
            if depth_prediction is not None:
                image_depth_similarity, depth_edge = get_images_color_similarity(
                    depth_prediction[im_i].unsqueeze(0), image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation, self.edge_thresh
                )

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)  # 由bbox生成伪mask

            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

            if per_im_gt_inst.has('pseudo_masks'):
                per_pseudo_masks = per_im_gt_inst.pseudo_masks
                per_pseudo_masks_pad = torch.zeros((per_pseudo_masks.shape[0], im_h, im_w), device=self.device).float()
                per_pseudo_masks_pad[:, :per_pseudo_masks.shape[1], :per_pseudo_masks.shape[2]] = per_pseudo_masks
                # per_pseudo_masks = F.interpolate(per_pseudo_masks_pad[None], size=(im_h // 4, im_w // 4),
                #                                  mode="bilinear", align_corners=False)
                per_pseudo_masks = per_pseudo_masks_pad[:, start::stride, start::stride]
                per_im_gt_inst.pseudo_masks = per_pseudo_masks
            if per_im_gt_inst.has('pseudo_depths'):
                per_pseudo_depths = per_im_gt_inst.pseudo_depths
                per_pseudo_depths_pad = torch.zeros((per_pseudo_depths.shape[0], im_h, im_w), device=self.device).float()
                per_pseudo_depths_pad[:, :per_pseudo_depths.shape[1], :per_pseudo_depths.shape[2]] = per_pseudo_depths
                # per_pseudo_masks = F.interpolate(per_pseudo_masks_pad[None], size=(im_h // 4, im_w // 4),
                #                                  mode="bilinear", align_corners=False)
                per_pseudo_depths = per_pseudo_depths_pad[:, start::stride, start::stride]
                per_im_gt_inst.pseudo_depths = per_pseudo_depths

            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst))
            ], dim=0)
            if depth_prediction is not None:
                per_im_gt_inst.image_depth_similarity = torch.cat([
                    image_depth_similarity for _ in range(len(per_im_gt_inst))
                ], dim=0)

                per_im_gt_inst.depth_edge = torch.cat([
                    depth_edge for _ in range(len(per_im_gt_inst))
                ], dim=0)
            depth_edges.append(depth_edge)
        return torch.stack(depth_edges, 0)

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        # if branch == 'teacher':
        #     scale_x, scale_y = 1.0, 1.0
        # else:
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        # if branch == 'teacher':
        #     output_height, output_width = resized_im_h, resized_im_w
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            if self.training:
                pred_depths = aligned_bilinear(results.pred_depths, factor)
                pred_depths = pred_depths[:, :, :resized_im_h, :resized_im_w]
                pred_depths = F.interpolate(pred_depths, size=(output_height, output_width),
                                           mode="bilinear", align_corners=False)[:, 0, :, :]
                results.pred_depths = pred_depths
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            if self.training:
                results.pred_masks = pred_global_masks
            else:
                results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
