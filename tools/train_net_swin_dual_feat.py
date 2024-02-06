#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
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
import itertools
import logging
import os
import time
from collections import OrderedDict

# import numpy as np
import torch
# import torchshow as ts
# import operator
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
# from detectron2.data import MetadataCatalog, build_detection_train_loader
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from adet.data.build import build_detection_semisup_train_loader_two_crops
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from adet.checkpoint.myhook import MyEvalHook
from detectron2.engine import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    # COCOPanopticEvaluator,
    DatasetEvaluators,
    # LVISEvaluator,
    # PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.env import TORCH_VERSION
# from adet.data.my_dataset_mapper_p2m import DatasetMapperWithBasis
from adet.data.my_dataset_mapper_two_crop import DatasetMapperTwoCrop
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer, Checkpoint_With_Interrupt
# from adet.evaluation import TextEvaluator
from adet.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from detectron2.engine.defaults import default_writers
from adet.modeling.ts_ensemble import EnsembleTSModel
# from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
from adet.utils.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from fvcore.nn.precise_bn import get_bn_modules
# from detectron2.structures.boxes import Boxes
from adet.layers.copy_paste import copy_and_paste
from typing import List, Mapping, Optional
import warnings

warnings.filterwarnings("ignore")
from adet.modeling.backbone import add_swint_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
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

        #         # create an teacher model
        #         if cfg.MODEL.BOXINST.SEMSUPER[0]>cfg.SOLVER.MAX_ITER:

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
        self.skip = True

        self.register_hooks(self.build_hooks())  # how to use the hook? and how to register?

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
        ret.append(MyEvalHook(cfg.TEST.EVAL_PERIOD,
                              test_and_save_results_teacher, begin_eval=cfg.MODEL.BOXINST.SEMSUPER[0]))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # self._trainer.model.module.mask_head._iter = torch.tensor([self.start_iter],
            #                                                    device=self._trainer.model.module.mask_head._iter.device)
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
        self.model.module.mask_head._iter = torch.tensor([self.start_iter],
                                                               device=self.model.module.mask_head._iter.device)
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

    def compute_iou_with_each_box(self, true_boxes, pred_boxes, pred_scores, mask_score_with_depth_sim, semi_rate):
        num_pred_box = true_boxes.shape[0]
        # match_idx = []
        # match_score =[]
        pred_boxes = pred_boxes.cpu()
        ggiou = generalized_box_iou(true_boxes, pred_boxes)

        # m_idx0 = linear_sum_assignment(ggiou, maximize=True)
        # # norm_idx = np.zeros_like(m_idx[0])
        # # norm_idx[m_idx[0]] = m_idx[1]
        # idx_giou0 = ggiou[m_idx0[0], m_idx0[1]]

        ggiou = self.threshold[1] * ggiou + semi_rate * mask_score_with_depth_sim \
                + (1 - (self.threshold[1] + semi_rate)) * pred_scores.repeat((num_pred_box, 1))
        # ggiou = self.threshold[1] * ggiou + (1 - self.threshold[1]) * pred_scores.repeat((num_pred_box, 1))
        m_idx = linear_sum_assignment(ggiou, maximize=True)
        # norm_idx = np.zeros_like(m_idx[0])
        # norm_idx[m_idx[0]] = m_idx[1]
        idx_giou = ggiou[m_idx[0], m_idx[1]]
        depth_score = mask_score_with_depth_sim[m_idx[0], m_idx[1]]
        # norm_idx_giou = np.zeros_like(idx_giou)
        # norm_idx_giou[m_idx[0]] = idx_giou
        # return norm_idx, norm_idx_giou
        return m_idx[1], idx_giou, depth_score

    def merge_pseudo_label(self, ori_data, teacher_prediction, depth_sims, semi_rate, mask_feat):
        # 取出ture bbox和 pred bbox
        # device = teacher_prediction[0]['instances'].pred_boxes.device
        # for single_data, single_pred, depth_sim in zip(ori_data, teacher_prediction, depth_sims):
        for single_data, single_pred, depth_sim, single_mask_feat in zip(ori_data, teacher_prediction, depth_sims,
                                                                         mask_feat):
            true_box = single_data["instances"].gt_boxes.tensor
            pred_box = single_pred["instances"].pred_boxes.tensor
            single_data['mask_feature'] = single_mask_feat.cpu()
            ins_num = len(single_data["instances"])
            img_size0 = single_data['instances'].image_size[0]
            img_size1 = single_data['instances'].image_size[1]
            single_data["instances"].pseudo_masks = torch.ones((ins_num, img_size0, img_size1))
            single_data["instances"].pseudo_scores = torch.zeros((ins_num))
            if not single_pred["instances"].has("pred_masks"):
                print("no pred masks")
                continue
            pred_masks = single_pred["instances"].pred_masks
            # pred_depths = single_pred["instances"].pred_depths
            pred_scores = single_pred["instances"].scores
            if len(pred_box) < len(true_box):
                print(len(pred_box))
                print(len(true_box))
                if len(true_box) > 50:
                    gap = (len(true_box) - len(pred_box)) * 2
                    pred_box = torch.cat((pred_box, pred_box[:gap]), dim=0)
                    pred_masks = torch.cat((pred_masks, pred_masks[:gap]), dim=0)
                    # pred_depths = torch.cat((pred_depths, pred_depths[:gap]), dim=0)
                    pred_scores = torch.cat((pred_scores, pred_scores[:gap]), dim=0)
                    print("out of memory")
                    # return ori_data
                else:
                    pred_box = torch.cat((pred_box, pred_box), dim=0)
                    pred_masks = torch.cat((pred_masks, pred_masks), dim=0)
                    # pred_depths = torch.cat((pred_depths, pred_depths), dim=0)
                    pred_scores = torch.cat((pred_scores, pred_scores), dim=0)
                if len(pred_box) < len(true_box):
                    print("no pred masks a ")
                    continue
            # one_mask = torch.ones_like(pred_masks[0])
            # pred_masks = torch.cat((pred_masks, one_mask[None]), dim=0)
            # pred_masks = (pred_masks>0.5).float()
            bitmasks_from_box = []
            # depth_map = single_data["depth_map"][2::4, 2::4]
            for box in true_box:
                bit_mask = torch.zeros_like(pred_masks[0])
                bit_mask[int(box[1]):int(box[3] + 1), int(box[0]):int(box[2] + 1)] = 1.0
                bitmasks_from_box.append(bit_mask[2::4, 2::4])
            bitmasks_from_box = torch.stack(bitmasks_from_box, dim=0).unsqueeze(1)
            pred_mask_with_box = bitmasks_from_box * pred_masks[None, :, 2::4, 2::4]
            # pred_mask_with_box = pred_mask_with_box[:,:,2::4, 2::4]
            depth_sim = depth_sim[2::4, 2::4]
            # depth_fg = [depth_sim>0.5].sum()
            mask_num = pred_mask_with_box.sum(dim=(2, 3)).clamp(min=1)
            # mask_with_depth_sim = (pred_mask_with_box * depth_sim).sum(dim=(2, 3))
            mask_fg = (pred_mask_with_box * (depth_sim > 0.5)).sum(dim=(2, 3))
            mask_edge = (pred_mask_with_box * (depth_sim < 0.1)).sum(dim=(2, 3))
            mask_score_with_depth_sim = (mask_fg / mask_num) + (mask_edge / mask_num) * 1
            match_idx, match_score, depth_score = self.compute_iou_with_each_box(true_box, pred_box,
                                                                                 pred_scores.cpu(),
                                                                                 mask_score_with_depth_sim.cpu(),
                                                                                 semi_rate)

            match_masks = pred_masks[match_idx]
            # match_masks0 = pred_masks[match_idx0]
            # name = ((single_data['file_name']).split('/')[-1]).split('.')[0]
            # from torchvision.utils import save_image
            # # a = (match_masks0>0.5).float()
            # # a[:,-10:,] = 1.0
            # # save_image((a.reshape(-1, a.shape[-1]))[None,None], f'/data8T/yl/Dataset/coco/010{name}.png')
            # if not match_masks.equal(match_masks0):
            #     # print(match_idx, "\n", match_idx0)
            #     a = (match_masks0 > 0.5).float()
            #     a[:, -10:, ] = 1.0
            #     save_image((a.reshape(-1, a.shape[-1]))[None, None], f'/data8T/yl/Dataset/coco/a{name}.png')
            #     b = (match_masks > 0.5).float()
            #     b[:, -10:, ] = 1.0
            #     save_image((b.reshape(-1, b.shape[-1]))[None, None], f'/data8T/yl/Dataset/coco/b{name}.png')
            #     # ts.save((match_masks0>0.5).float(), f'/data8T/yl/Dataset/coco/010{name}.png')
            #     # ts.save((match_masks>0.5).float(), f'/data8T/yl/Dataset/coco/011{name}.png')
            # match_depths = pred_depths[match_idx]
            single_data["instances"].depth_score = depth_score.cpu()
            single_data["instances"].pseudo_masks = match_masks.cpu()
            # single_data["instances"].mask_feature = mask_feat.cpu()
            # single_data["instances"].pseudo_depths = match_depths.cpu()
            single_data["instances"].pseudo_scores = torch.tensor(match_score)
            single_data["instances"].paste_flag = torch.zeros_like(match_score)
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
        # if self.skip:
        #     iii = self.iter % (self.data_loader.dataset.dataset._dataset._addr.size//self.data_loader.batch_size)
        #     while iii:
        #         _, _ = next(self._trainer._data_loader_iter)
        #         iii -= 1
        #     self.skip = False
        #     print(time.perf_counter() - start)
        #
        data_strong, data = next(self._trainer._data_loader_iter)
        # if self._trainer.iter % 100 == 0:
        #     print(self._trainer.iter)
        #     print(data[0]['file_name'])
        data_time = time.perf_counter() - start

        if self.iter < self.cfg.MODEL.BOXINST.SEMSUPER[0]:  # 在此处,  data aug for different models(teacher and student)

            # input both strong and weak supervised data into model
            # label_data_q.extend(label_data_k)
            _, loss_dict = self.model(
                data, branch="student")

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

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
            if self.iter == self.cfg.MODEL.BOXINST.SEMSUPER[0]:  # EMA
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
            # elif self.iter % self.cfg.TEST.EVAL_PERIOD == 0:  # 1
            #     self._update_teacher_model(keep_rate=0.00)
            elif (
                    self.iter - self.cfg.MODEL.BOXINST.SEMSUPER[0]
            ) % self.cfg.MODEL.BOXINST.TEACHER_UPDATE_ITER == 0:  # 1
                self._update_teacher_model(
                    keep_rate=self.cfg.MODEL.BOXINST.TEACHER_EMA_KEEP_RATE[0])
            semi_rate = (1 - self.iter / self.cfg.SOLVER.MAX_ITER) * self.cfg.MODEL.BOXINST.TEACHER_EMA_KEEP_RATE[1]
            with torch.no_grad():
                # (teacher_prediction, depth_sims) = self.model_teacher(data, branch="teacher")
                (teacher_prediction, mask_feat, depth_sims) = self.model_teacher(data, branch="teacher")

            #  Pseudo-labeling
            # cur_threshold = self.threshold

            data_strong = self.merge_pseudo_label(data_strong, teacher_prediction, depth_sims, semi_rate, mask_feat)
            # print(data[0]['file_name'])
            # data = self.merge_pseudo_label(data, teacher_prediction, mask_feature_attention, depth_sims)
            new_merged_data = copy_and_paste(data_strong, data_strong[::-1])
            if len(new_merged_data) > 0:
                data_strong.extend(new_merged_data)
            _, loss_dict = self.model(data_strong, branch="student")
            ## extend data, choose one to copy and paste

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

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperTwoCrop(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper=mapper)
        # return build_detection_train_loader(cfg, mapper=mapper)

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     if cfg.MODEL.FCPOSE_ON:
    #         mapper = FCPoseDatasetMapper(cfg, True)
    #         return build_detection_train_loader(cfg, mapper=mapper)
    #     else:
    #         mapper = DatasetMapperTwoCrop(cfg, True)
    #         # This customized mapper produces two augmented images from a single image
    #         # instance. This mapper makes sure that the two augmented images have the same
    #         # cropping and thus the same size.
    #         return build_detection_semisup_train_loader_two_crops(cfg, mapper)

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
        # if evaluator_type == "coco_panoptic_seg":
        #     evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        # if evaluator_type == "lvis":
        #     return LVISEvaluator(dataset_name, cfg, True, output_folder)
        # if evaluator_type == "text":
        #     return TextEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        # elif evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        # elif evaluator_type == "lvis":
        #     return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
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

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
                "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
            }
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_swint_config(cfg)
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

        if cfg.TEST.get("MODEL", False):
            if cfg.TEST.MODEL == 'modelStudent':
                res = Trainer.test(cfg, ensem_ts_model.modelStudent)
            else:
                res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        else:
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
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
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
