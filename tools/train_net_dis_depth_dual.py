# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
# import torchvision.datasets
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
# from detectron2.data import MetadataCatalog
from adet.data.build import build_detection_semisup_train_loader_two_crops
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.env import TORCH_VERSION
from adet.data.my_dataset_mapper_p2m import DatasetMapperWithBasis
from adet.data.my_dataset_mapper_two_crop import DatasetMapperTwoCrop
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer, Checkpoint_With_Interrupt
from adet.evaluation import TextEvaluator
from adet.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from detectron2.engine.defaults import default_writers
from adet.modeling.ts_ensemble import EnsembleTSModel
from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
from adet.utils.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from typing import List, Mapping, Optional
import warnings

warnings.filterwarnings("ignore")


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        # data_loader = self.build_train_loader(cfg)

        # create an teacher model
        model_teacher = self.build_model(cfg)  #
        self.model_teacher = model_teacher

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )  # init trainer

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)  # init lr_scheduler

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        # self.checkpointer = DetectionCheckpointer(
        #     model,
        #     cfg.OUTPUT_DIR,
        #     optimizer=optimizer,
        #     scheduler=self.scheduler,
        # )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.threshold = cfg.MODEL.BOXINST.SEMSUPER_THRESHOLD

        self.register_hooks(self.build_hooks())  # how to use the hook? and how to register?

    # def build_hooks(self):
    #     """
    #     Replace `DetectionCheckpointer` with `AdetCheckpointer`.
    #
    #     Build a list of default hooks, including timing, evaluation,
    #     checkpointing, lr scheduling, precise BN, writing events.
    #     """
    #
    #     ret = super().build_hooks()
    #     for i in range(len(ret)):
    #         if isinstance(ret[i], hooks.PeriodicCheckpointer):
    #             # print("*-*")
    #             self.checkpointer = AdetCheckpointer(
    #                 self.model,
    #                 self.cfg.OUTPUT_DIR,
    #                 optimizer=self.optimizer,
    #                 scheduler=self.scheduler,
    #             )
    #             if self.cfg.MODEL.INTERRUPT:
    #                 ret[i] = Checkpoint_With_Interrupt(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
    #             else:
    #                 ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
    #     return ret

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            if self.cfg.MODEL.INTERRUPT:
                ret.append(Checkpoint_With_Interrupt(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD))
            else:
                ret.append(
                    hooks.PeriodicCheckpointer(
                        self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                    )
                )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]


    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                # self.run_step()
                self.run_step_full_semisup()
                # print(self.iter)
                self.after_step()
            self.after_train()

    # def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    #     if proposal_type == "rpn":
    #         valid_map = proposal_bbox_inst.objectness_logits > thres
    #
    #         # create instances containing boxes and gt_classes
    #         image_shape = proposal_bbox_inst.image_size
    #         new_proposal_inst = Instances(image_shape)
    #
    #         # create box
    #         new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
    #         new_boxes = Boxes(new_bbox_loc)
    #
    #         # add boxes to instances
    #         new_proposal_inst.gt_boxes = new_boxes
    #         new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
    #             valid_map
    #         ]
    #     elif proposal_type == "roih":
    #         valid_map = proposal_bbox_inst.scores > thres
    #
    #         # create instances containing boxes and gt_classes
    #         image_shape = proposal_bbox_inst.image_size
    #         new_proposal_inst = Instances(image_shape)
    #
    #         # create box
    #         new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
    #         new_boxes = Boxes(new_bbox_loc)
    #
    #         # add boxes to instances
    #         new_proposal_inst.gt_boxes = new_boxes
    #         new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
    #         new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
    #
    #     return new_proposal_inst

    # def process_pseudo_label(
    #         self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    # ):
    #
    #     list_instances = []
    #     num_proposal_output = 0.0
    #     for proposal_bbox_inst in proposals_rpn_unsup_k:
    #         # thresholding
    #         if psedo_label_method == "thresholding":
    #             proposal_bbox_inst = self.threshold_bbox(
    #                 proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
    #             )
    #         else:
    #             raise ValueError("Unkown pseudo label boxes methods")
    #         num_proposal_output += len(proposal_bbox_inst)
    #         list_instances.append(proposal_bbox_inst)
    #     num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
    #     return list_instances, num_proposal_output

    # mathi
    def compute_iou_with_each_box(self, true_boxes, pred_boxes, pred_scores, mask_score_with_depth_sim):
        num_pred_box = true_boxes.shape[0]
        # match_idx = []
        # match_score =[]
        pred_boxes = pred_boxes.cpu()
        ggiou = generalized_box_iou(true_boxes, pred_boxes)
        ggiou = self.threshold[1] * ggiou + (1-self.threshold[1]) * mask_score_with_depth_sim
        # ggiou = self.threshold[1] * ggiou + (1 - self.threshold[1]) * pred_scores.repeat((num_pred_box, 1))
        m_idx = linear_sum_assignment(ggiou, maximize=True)
        # norm_idx = np.zeros_like(m_idx[0])
        # norm_idx[m_idx[0]] = m_idx[1]
        idx_giou = ggiou[m_idx[0], m_idx[1]]
        # norm_idx_giou = np.zeros_like(idx_giou)
        # norm_idx_giou[m_idx[0]] = idx_giou
        # return norm_idx, norm_idx_giou
        return m_idx[1], idx_giou
        # for i, true_box in enumerate(true_boxes):
        #     true_box = true_box.repeat(num_pred_box, 1)
        #     iou, giou = compute_ious(true_box, pred_boxes)
        #     # rank_score = self.threshold[1] * giou + (1-self.threshold[1]) * pred_scores
        #     # rank_score = self.iou * iou + self.score * pred_scores
        #     rank_score = giou
        #     idx = int(rank_score.argmax())
        #     max_score = rank_score[idx]
        #     if max_score > self.threshold[0]:
        #         # match_idx.append([i, idx])
        #         match_idx.append(idx)
        #         match_score.append(float(max_score))
        #         pred_boxes[idx] = 0
        #         pred_scores[idx] = 0  # 去除已经被匹配的box
        #     else:
        #         # match_idx.append([i, -1])
        #         match_idx.append(-1)
        #         match_score.append(0.)

        # return match_idx, match_score


    def merge_pseudo_label(self, ori_data, teacher_prediction, depth_sims):
        # 取出ture bbox和 pred bbox
        # device = teacher_prediction[0]['instances'].pred_boxes.device
        for single_data, single_pred, depth_sim in zip(ori_data, teacher_prediction, depth_sims):
        # for single_data, single_pred, mask_feature, depth_sim in zip(ori_data, teacher_prediction,
        #                                                                  mask_feature_attention, depth_sims):
            true_box = single_data["instances"].gt_boxes.tensor
            pred_box = single_pred["instances"].pred_boxes.tensor
            # single_data['mask_feature'] = mask_feature.cpu()
            if not single_pred["instances"].has("pred_global_masks") or len(pred_box)< len(true_box):
                print(len(pred_box)< len(true_box))
                print(len(pred_box))
                print(len(true_box))
                return ori_data
            pred_masks = single_pred["instances"].pred_masks #pred_masks = single_pred["instances"].pred_masks
            # one_mask = torch.ones_like(pred_masks[0])
            # pred_masks = torch.cat((pred_masks, one_mask[None]), dim=0)
            # pred_masks = (pred_masks>0.5).float()
            bitmasks_from_box = []
            # depth_map = single_data["depth_map"][2::4, 2::4]
            for box in true_box:
                bit_mask = torch.zeros_like(pred_masks[0])
                bit_mask[int(box[1]):int(box[3] + 1), int(box[0]):int(box[2] + 1)] = 1.0
                bitmasks_from_box.append(bit_mask[2::4, 2::4])
            bitmasks_from_box = torch.stack(bitmasks_from_box, dim =0).unsqueeze(1)
            pred_mask_with_box = bitmasks_from_box * pred_masks[None, :, 2::4, 2::4]
            # pred_mask_with_box = pred_mask_with_box[:,:,2::4, 2::4]
            depth_sim = depth_sim[2::4, 2::4]
            # depth_fg = [depth_sim>0.5].sum()
            mask_num = pred_mask_with_box.sum(dim=(2,3)).clamp(min=1)
            # mask_with_depth_sim = (pred_mask_with_box * depth_sim).sum(dim=(2, 3))
            mask_fg = (pred_mask_with_box *(depth_sim>0.5)).sum(dim=(2, 3))
            mask_edge = (pred_mask_with_box *(depth_sim<0.1)).sum(dim=(2, 3))
            mask_score_with_depth_sim = (mask_fg / mask_num) - (mask_edge / mask_num) * 0
            pred_scores = single_pred["instances"].scores
            match_idx, match_score = self.compute_iou_with_each_box(true_box, pred_box, pred_scores.cpu(), mask_score_with_depth_sim.cpu())
            # if len(match_idx)!= len(single_data["instances"]):
            #     print(len(match_idx))
            #     print(len(single_data["instances"]))
            #     return ori_data
            match_masks = pred_masks[match_idx]
            single_data["instances"].pseudo_masks = match_masks.cpu()
            single_data["instances"].pseudo_scores = torch.tensor(match_score)
            # pred_feature = single_pred["instances"].top_feat
            # giou = compute_iou(true_box, pred_box)  # bbox和每个pred box均进行计算
        return ori_data

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)


    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data_strong, data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        # label_data_q, label_data_aug = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        # unlabel_data_q = self.remove_label(unlabel_data_q)  # 在此之前的操作都不需要，只需要不同数据增强获得的两组图像
        # unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.MODEL.BOXINST.SEMSUPER:  # 在此处,  data aug for different models(teacher and student)

            # input both strong and weak supervised data into model
            # label_data_q.extend(label_data_k)
            _, loss_dict = self.model(
                data, branch="student")

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            # weight losses
            # loss_dict = {}
            # for key in record_dict.keys():
            #     if key[:4] == "loss":
            #         loss_dict[key] = record_dict[key] * 1
            # losses = sum(loss_dict.values())

            """
                    If you need to accumulate gradients or do something similar, you can
                    wrap the optimizer with your custom `zero_grad()` method.
                    """
            self.optimizer.zero_grad()
            losses.backward()

            self._write_metrics(loss_dict, data_time)

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self.optimizer.step()

        else:
            if self.iter == self.cfg.MODEL.BOXINST.SEMSUPER:  # EMA
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
            elif self.iter % self.cfg.TEST.EVAL_PERIOD == 0:  # 1
                self._update_teacher_model(keep_rate=0.00)
            elif (
                    self.iter - self.cfg.MODEL.BOXINST.SEMSUPER
            ) % self.cfg.MODEL.BOXINST.TEACHER_UPDATE_ITER == 0:  # 1
                self._update_teacher_model(
                    keep_rate=self.cfg.MODEL.BOXINST.TEACHER_EMA_KEEP_RATE)

            # record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (teacher_prediction, depth_sims) = self.model_teacher(data, branch="teacher")
                # (teacher_prediction, mask_feature_attention, depth_sims) = self.model_teacher(data, branch="teacher")

            #  Pseudo-labeling
            # cur_threshold = self.threshold

            data_strong = self.merge_pseudo_label(data_strong, teacher_prediction, depth_sims)
            # data = self.merge_pseudo_label(data, teacher_prediction, mask_feature_attention, depth_sims)

            _, loss_dict = self.model(data_strong, branch="student")

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            # weight losses
            # loss_dict = {}
            # for key in record_dict.keys():
            #     if key[:4] == "loss":
            #         loss_dict[key] = record_dict[key] * 1
            # losses = sum(loss_dict.values())

            """
                    If you need to accumulate gradients or do something similar, you can
                    wrap the optimizer with your custom `zero_grad()` method.
                    """
            self.optimizer.zero_grad()
            losses.backward()

            self._write_metrics(loss_dict, data_time)

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self.optimizer.step()



            # 
            # joint_proposal_dict = {}
            # joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            # (
            #     pesudo_proposals_rpn_unsup_k,
            #     nun_pseudo_bbox_rpn,
            # ) = self.process_pseudo_label(
            #     proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            # )
            # joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # 
            # # Pseudo_labeling for ROI head (bbox location/objectness)
            # pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            #     proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            # )
            # joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            # 
            # #  add pseudo-label to unlabeled data
            # 
            # unlabel_data_q = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            # 
            # all_label_data = label_data_q + label_data_k
            # all_unlabel_data = unlabel_data_q

            # record_all_label_data, _, _, _ = self.model(
            #     all_label_data, branch="supervised"
            # )  # compute loss
            # record_dict.update(record_all_label_data)
            # record_all_unlabel_data, _, _, _ = self.model(
            #     all_unlabel_data, branch="supervised"
            # )  # here
            # new_record_all_unlabel_data = {}
            # for key in record_all_unlabel_data.keys():
            #     new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
            #         key
            #     ]
            # record_dict.update(new_record_all_unlabel_data)
            # 
            # # weight losses
            # loss_dict = {}
            # for key in record_dict.keys():
            #     if key[:4] == "loss":
            #         if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
            #             # pseudo bbox regression <- 0
            #             loss_dict[key] = record_dict[key] * 0
            #         elif key[-6:] == "pseudo":  # unsupervised loss
            #             loss_dict[key] = (
            #                     record_dict[key] *
            #                     self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
            #             )
            #         else:  # supervised loss
            #             loss_dict[key] = record_dict[key] * 1
            # 
            # losses = sum(loss_dict.values())

        # metrics_dict = record_dict
        # metrics_dict["data_time"] = data_time
        # self._write_metrics(metrics_dict)
        # 
        # self.optimizer.zero_grad()
        # losses.backward()
        # self.optimizer.step()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)

        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     """
    #     Returns:
    #         iterable
    #
    #     It calls :func:`detectron2.data.build_detection_train_loader` with a customized
    #     DatasetMapper, which adds categorical labels as a semantic mask.
    #     """
    #     if cfg.MODEL.FCPOSE_ON:
    #         mapper = FCPoseDatasetMapper(cfg, True)
    #     else:
    #         mapper = DatasetMapperWithBasis(cfg, True)
    #     return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = DatasetMapperTwoCrop(cfg, True)
            # This customized mapper produces two augmented images from a single image
            # instance. This mapper makes sure that the two augmented images have the same
            # cropping and thus the same size.
            return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        # pre = (self.cfg.OUTPUT_DIR).split('/')[:-1]
        suf = (self.cfg.OUTPUT_DIR).split('/')[-1]
        log_dir = (self.cfg.OUTPUT_DIR).replace(suf, 'tensorboard/' + suf)
        return default_writers(log_dir, self.max_iter)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:

        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        # model = Trainer.build_model(cfg)
        # AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        # res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    # trainer.checkpointer()
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
