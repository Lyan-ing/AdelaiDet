# from copy import deepcopy

import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
# import torchshow as ts
# from adet.utils.kmean_gpu import kmeans
from adet.modeling.condinst.condinst import unfold_wo_center


def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_all_project_term(mask_scores, depth_cob_box):
    mask_losses_y = dice_coefficient(
        mask_scores.min(dim=2, keepdim=True)[0],
        depth_cob_box.min(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.min(dim=3, keepdim=True)[0],
        depth_cob_box.min(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)
    # (18,8)-->(8,8)-->(8,1) channel change  [18*8=144, 8*8=64, 8*1=8] weight  [8, 8, 1] bias
    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 2:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)  # (n, 144)-->(n*8, 18, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)  # (n, 8)-->(n*8, )
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)  # change the kernel to 3*3
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


def comput_irn_loss(mask_scores, gt_edge, gt_bitmasks, dilation):
    mask_score = mask_scores.clone()
    mask_score[gt_edge] = 10
    mask_diff = unfold_wo_center(mask_score, 3, dilation)
    mask_diff = mask_diff[:, 0]
    left_top = (mask_diff[:, 0] - mask_diff[:, 7]).abs()
    top = (mask_diff[:, 1] - mask_diff[:, 6]).abs()
    right_top = (mask_diff[:, 2] - mask_diff[:, 5]).abs()
    left = (mask_diff[:, 3] - mask_diff[:, 4]).abs()
    diff = torch.stack([left_top, top, right_top, left], 1)
    # diff = (torch.stack([left_top, top, right_top, left], 1)).max(1,True)[0]
    diff[diff == 0] = 0
    diff[diff > 5] = 0
    is_differnet = (mask_scores>0.4 ) * ( mask_scores<0.6)
    edge_diff = diff * (gt_edge * gt_bitmasks) * is_differnet
    mask_edge = edge_diff[edge_diff != 0]
    if len(mask_edge) == 0:
        loss_edge = mask_score.sum() * 0.
        # print(mask_score.sum())
        # print("**************************")
    else:
        # print(mask_edge.mean() / 4)
        loss_edge = - torch.log(mask_edge.mean() + 1e-5)
    return loss_edge


# def comput_irn_loss_tmp(mask_scores, gt_edge, gt_bitmasks, dilation):
#     mask_score = mask_scores.clone()
#     gt_bitmasks = gt_bitmasks * (mask_score > 0.5)
#     mask_score[gt_edge] = 10
#     mask_diff = unfold_wo_center(mask_score, 3, dilation)
#     mask_diff = mask_diff[:, 0]
#     left_top = (mask_diff[:, 0] - mask_diff[:, 7]).abs()
#     top = (mask_diff[:, 1] - mask_diff[:, 6]).abs()
#     right_top = (mask_diff[:, 2] - mask_diff[:, 5]).abs()
#     left = (mask_diff[:, 3] - mask_diff[:, 4]).abs()
#     diff = torch.stack([left_top, top, right_top, left], 1)
#     diff[diff < 1e-4] = 0
#     diff[diff > 5] = 0
#     edge_diff = diff * (gt_edge * gt_bitmasks)
#     mask_edge = edge_diff[edge_diff > 1e-4]
#     if len(mask_edge) == 0:
#         loss_edge = mask_score.sum() * 0.
#         # print(mask_score.sum())
#         # print("**************************")
#     else:
#         # print(mask_edge.mean() / 4)
#         loss_edge = - torch.log(mask_edge.mean() + 1e-5)
#     return loss_edge


MSEL = nn.MSELoss()


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        # self.use_depth = cfg.MODEL.BOXINST.PAIRWISE.USE_DEPTH
        self.depth_thresh = cfg.MODEL.BOXINST.PAIRWISE.DEPTH_THRESH
        self.pairwise_depth_thresh = cfg.MODEL.BOXINST.PAIRWISE.DEPTH_SIM_THRESH
        self.hpyer_parameters = cfg.MODEL.BOXINST.PAIRWISE.HYPERS
        self.depth_with_box = cfg.MODEL.BOXINST.PAIRWISE.DEPTH_WITH_BOX
        self.depth_cob_sim = False
        self.bg_depth = cfg.MODEL.BOXINST.PAIRWISE.DEPTH_BG
        self.edge_dilation = cfg.MODEL.BOXINST.DILATIONS
        self.tmp = cfg.MODEL.BOXINST.TMP
        self.semi_threshold = cfg.MODEL.BOXINST.SEMSUPER_THRESHOLD
        self.tmp2 = cfg.MODEL.BOXINST.CROSS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels)
                weight_nums.append(self.channels)  # change the kernel to double
                bias_nums.append(1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.tmp3 = cfg.MODEL.BOXINST.PSEUDO_DEPTH

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        pad = 0
        depth = None
        for i, (w, b) in enumerate(zip(weights, biases)):  # 1*1卷积
            if i < n_layers - 2:
                x = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
                x = F.relu(x)
            if i == n_layers - 2:
                depth = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
                # depth = F.relu(depth)
            if i == n_layers - 1:
                x = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
                if self.tmp:
                    depth = depth.sigmoid()
                    x = x * depth
                else:
                    x = x + depth
                    depth = depth.sigmoid()
        return x, depth  # mask head output

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)  # (n, c(16+2), h, w) --> (1, n*c, h,w)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits, depth_logits = self.mask_heads_forward(mask_head_inputs, weights, biases,
                                                            n_inst)  # 获得controller的参数

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        depth_logits = depth_logits.reshape(-1, 1, H, W)

        depth_logits = aligned_bilinear(depth_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits, depth_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None, depth_map=None,
                 branch='student'):
        if self.training and branch == 'student':
            self._iter += 1

            gt_inds = pred_instances.gt_inds  # 预测实例对应真实实例的索引（一个batch所有图像的所有实例数）
            im_inds = pred_instances.im_inds  # 实例对应图像索引
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_edge = torch.cat([per_im.depth_edge for per_im in gt_instances])
            gt_edge = gt_edge[gt_inds].unsqueeze(dim=1)
            # 获取预测实例应当对应的mask
            gt_depth = depth_map[im_inds].to(dtype=mask_feats.dtype)  # 每张图对应的gt_depth

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:  # controller weights
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits, depth_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()
                # depth_score = depth_logits.sigmoid()

                # 编码mask的几何特征
                # mask_emb = self.model(mask_scores)

                # update the geo feature

                if self.boxinst_enabled:
                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    # depth_pred

                    if gt_instances[-1].has('pseudo_masks'):
                        pseudo_masks = torch.cat([per_im.pseudo_masks for per_im in gt_instances])
                        pseudo_masks = pseudo_masks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                        pseudo_masks = pseudo_masks * gt_bitmasks
                        pseudo_scores = torch.cat([per_im.pseudo_scores for per_im in gt_instances])
                        pseudo_scores = pseudo_scores[gt_inds].to(dtype=mask_feats.dtype)
                        # if self.tmp:
                        # pseudo_masks = (mask_scores.detach() > 0.5).float() * pseudo_masks
                        loss_pseudo_masks = dice_coefficient(mask_scores, pseudo_masks)
                        # max_matching_mask = [int(((1 - loss_pseudo_masks) * (gt_inds == i)).argmax()) for i in
                        #                      gt_inds.unique()]
                        # pseudo_scores[max_matching_mask] = 1.0
                        # useful_pseudo_masks = pseudo_masks.min(dim=(1,2,3))
                        # usable_masks = [pseudo_scores>0.]
                        if pseudo_scores.max() > self.semi_threshold[0]:
                            if self.tmp2 =='normal':
                                loss_pseudo_masks = loss_pseudo_masks[pseudo_scores > self.semi_threshold[0]].mean()
                            if self.tmp2 =='mask':
                                loss_pseudo_masks = (loss_pseudo_masks* pseudo_scores)[pseudo_scores > self.semi_threshold[0]].mean()
                            if not self.tmp3:
                                pseudo_depths = torch.cat([per_im.pseudo_depths for per_im in gt_instances])
                                pseudo_depths = pseudo_depths[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                                pseudo_depths = pseudo_depths * gt_bitmasks
                                # if self.tmp:
                                # pseudo_masks = (mask_scores.detach() > 0.5).float() * pseudo_masks
                                loss_pseudo_depths = (
                                            ((depth_logits - pseudo_depths) * gt_bitmasks).pow(2).sum(dim=(1, 2, 3)) \
                                            / gt_bitmasks.sum(dim=(1, 2, 3)).clamp(min=1.0))
                                loss_pseudo_depths = loss_pseudo_depths[pseudo_scores > self.semi_threshold[0]].mean()

                        else:
                            loss_pseudo_masks = loss_pseudo_masks.mean() * 0.
                            loss_pseudo_depths = depth_logits.mean() * 0.
                        losses.update({
                            'loss_pseudo_masks': loss_pseudo_masks * self.hpyer_parameters[2],
                        })

                        # if gt_instances[-1].has('pseudo_depths'):
                        if self.tmp3:
                            pseudo_depths = torch.cat([per_im.pseudo_depths for per_im in gt_instances])
                            pseudo_depths = pseudo_depths[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                            pseudo_depths = pseudo_depths * gt_bitmasks
                            # if self.tmp:
                            # pseudo_masks = (mask_scores.detach() > 0.5).float() * pseudo_masks
                            loss_pseudo_depths = ((depth_logits - pseudo_depths)[gt_bitmasks.bool()]).pow(2).mean()
                        # if pseudo_scores.max() > self.semi_threshold[0]:
                        #     loss_pseudo_depths = loss_pseudo_depths[pseudo_scores > self.semi_threshold[0]].mean()
                        # else:
                        #     loss_pseudo_depths = loss_pseudo_depths.mean() * 0.
                        losses.update({
                            'loss_pseudo_depths': loss_pseudo_depths * self.hpyer_parameters[5],
                        })

                    else:
                        losses.update({
                            'loss_pseudo_masks': mask_scores.mean() * 0.,
                        })
                        losses.update({
                            'loss_pseudo_depths': depth_logits.mean() * 0.,
                        })
                    norm_depth = gt_depth / 255.0
                    # loss_depth_pred = MSEL(depth_logits, norm_depth)
                    loss_depth_pred = ((depth_logits - norm_depth)[gt_bitmasks.bool()]).pow(2).mean()

                    losses.update({
                        "loss_ins_depth": loss_depth_pred * self.hpyer_parameters[1],
                    })

                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    # warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor * self.hpyer_parameters[0]

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                    if gt_instances[-1].has('pseudo_masks') and self.hpyer_parameters[4]:
                        loss_edge = 0
                        for dilation in self.edge_dilation:
                            # if self.tmp:
                            #     loss_cur_edge = comput_irn_loss_tmp(mask_scores, gt_edge, gt_bitmasks, dilation)
                            # else:
                            loss_cur_edge = comput_irn_loss(mask_scores, gt_edge, gt_bitmasks, dilation)
                            loss_edge = loss_edge + loss_cur_edge
                        loss_edge = loss_edge / len(self.edge_dilation)
                        # loss_edge = loss.mean()
                        # print(loss_edge)
                        losses.update({
                            "loss_edge": loss_edge * warmup_factor * self.hpyer_parameters[4],
                        })

                    if (self._iter > self._warmup_iters) and self.pairwise_depth_thresh:
                        image_depth_similarity = torch.cat([x.image_depth_similarity for x in gt_instances])
                        image_depth_similarity = image_depth_similarity[gt_inds].to(dtype=mask_feats.dtype)
                        depth_weights = (image_depth_similarity >= self.pairwise_depth_thresh).float()
                        if self.depth_with_box:
                            depth_weights = depth_weights * gt_bitmasks.float()
                        loss_depth_pairwise = (pairwise_losses * depth_weights).sum() / depth_weights.sum().clamp(
                            min=1.0)
                        warmup_factor2 = min((self._iter.item() - self._warmup_iters) / float(self._warmup_iters), 1.0)
                        loss_depth_pairwise = loss_depth_pairwise * warmup_factor2 * self.hpyer_parameters[3]
                        losses.update({
                            'loss_depth_pairwise': loss_depth_pairwise,
                        })

                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        elif self.training and branch == 'teacher':
            if len(pred_instances) > 0:
                mask_logits, mask_depth = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()
                pred_instances.pred_depths = mask_depth

            return pred_instances
        else:
            if len(pred_instances) > 0:
                mask_logits, _ = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances
