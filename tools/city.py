# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
from itertools import chain
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import setup_logger

from cityscapesscripts.helpers.labels import labels
from detectron2.data import MetadataCatalog, DatasetCatalog
from tools.train_net import Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = logging.getLogger(__name__)


def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_cityscapes_instances(image_dir, gt_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    files = _get_cityscapes_files(image_dir, gt_dir)

    logger.info("Preprocessing cityscapes annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=4)

    ret = pool.map(
        functools.partial(_cityscapes_files_to_dict),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    save=gt_dir.split('/')[-1]
    json.dump(ret, open(f'/data8T/yl/Dataset/cityscapes/{save}_nomask.json', 'w'))
    return ret


def _cityscapes_files_to_dict(files):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.
    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).
    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    from shapely.geometry import MultiPolygon, Polygon

    with PathManager.open(json_file, "r") as f:
        jsonobj = json.load(f)
    ret = {
        "file_name": image_file.split('val/')[-1].split('train/')[-1].split('test/')[-1],
        "image_id": os.path.basename(image_file),
        "height": jsonobj["imgHeight"],
        "width": jsonobj["imgWidth"],
    }

    # `polygons_union` contains the union of all valid polygons.
    polygons_union = Polygon()

    # CityscapesScripts draw the polygons in sequential order
    # and each polygon *overwrites* existing ones. See
    # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
    # We use reverse order, and each polygon *avoids* early ones.
    # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
    for obj in jsonobj["objects"][::-1]:
        if "deleted" in obj:  # cityscapes data format specific
            continue
        label_name = obj["label"]

        try:
            label = name2label[label_name]
        except KeyError:
            if label_name.endswith("group"):  # crowd area
                label = name2label[label_name[: -len("group")]]
            else:
                raise
        if label.id < 0:  # cityscapes data format
            continue

        # Cityscapes's raw annotations uses integer coordinates
        # Therefore +0.5 here
        poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
        # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
        # polygons for evaluation. This function operates in integer space
        # and draws each pixel whose center falls into the polygon.
        # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
        # We therefore dilate the input polygon by 0.5 as our input.
        poly = Polygon(poly_coord).buffer(0.5, resolution=4)

        if not label.hasInstances or label.ignoreInEval:
            # even if we won't store the polygon it still contributes to overlaps resolution
            polygons_union = polygons_union.union(poly)
            continue

        # Take non-overlapping part of the polygon
        poly_wo_overlaps = poly.difference(polygons_union)
        if poly_wo_overlaps.is_empty:
            continue
        polygons_union = polygons_union.union(poly)

        anno = {}
        anno["iscrowd"] = label_name.endswith("group")
        anno["category_id"] = label.id

        if isinstance(poly_wo_overlaps, Polygon):
            poly_list = [poly_wo_overlaps]
        elif isinstance(poly_wo_overlaps, MultiPolygon):
            poly_list = poly_wo_overlaps.geoms
        else:
            raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

        poly_coord = []
        for poly_el in poly_list:
            # COCO API can work only with exterior boundaries now, hence we store only them.
            # TODO: store both exterior and interior boundaries once other parts of the
            # codebase support holes in polygons.
            poly_coord.append(list(chain(*poly_el.exterior.coords)))
        anno["segmentation"] = poly_coord
        (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

        anno["bbox"] = (xmin, ymin, xmax, ymax)
        anno["bbox_mode"] = BoxMode.XYXY_ABS

        annos.append(anno)

    ret["annotations"] = annos
    return ret


def get_cityscapes_dict(image_dir, gt_dir):
    dicts = load_cityscapes_instances(
        image_dir, gt_dir, from_json=True, to_polygons=True
    )
    logger.info("Done loading {} samples.".format(len(dicts)))


def train():
    cfg.OUTPUT_DIR = "./output/cityscapes/multi-2-2/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.WARMUP_ITERS = 500
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


# def test():
#     cfg.MODEL.WEIGHTS = './output/cityscapes/multi-2-2/model_final.pth'
#     model = Trainer.build_model(cfg)
#     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#         cfg.MODEL.WEIGHTS, resume=False
#     )
#     # cfg.TEST.AUG.CORRUPT = True
#     # Trainer.test_with_TTA(cfg, model)
#     Trainer.test(cfg, model)


if __name__ == "__main__":
    cityscapes_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]

    image_dir = "/data8T/yl/Dataset/cityscapes/leftImg8bit/"
    gt_dir = "/data8T/yl/Dataset/cityscapes/gtFine/"

    for d in ["train",'test','val']:
        DatasetCatalog.register("cityscapes_" + d,
                                lambda x=image_dir + d, y=gt_dir + d: load_cityscapes_instances(x, y))
        MetadataCatalog.get("cityscapes_" + d).set(thing_classes=cityscapes_classes,
                                                   evaluator_type="coco")

    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("../configs/Cityscapes/faster_rcnn_R-50-FPN.yaml"))

    cfg.DATASETS.TRAIN = ("cityscapes_val")
    cfg.DATASETS.TEST = ("cityscapes_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cityscapes_classes)
    cfg.SOLVER.IMS_PER_BATCH = 4
    train()




