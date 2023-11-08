# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from cgi import test
import logging
import os
from collections import OrderedDict
from click import argument
from tqdm import tqdm
import torch
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# from viz_utils import *
import pdb
import sys
import torchvision
import glob
import copy

import itertools

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode , Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, PascalVOCDetectionEvaluator

from detectron2.utils.file_io import PathManager

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET

import json

from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.data.ho import load_ho_voc_instances
from hodetector.modeling import roi_heads

from html_editor import write_html, write_separate_html, write_obj_html
#register your data

_datasets_root = "/y/evacheng/val_hands/"
  # register_ho_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand", "targetobject", "secondobject"])
register_coco_instances(name = '100DOH_hand_VAL', metadata = {}, json_file = "/y/evacheng/hands_new/annotations/val.json", image_root = "/y/evacheng/hands_new/val/")
MetadataCatalog.get(f'100DOH_hand_VAL').set(evaluator_type='coco')




#load the config file, configure the threshold value, load weights 
cfg = get_cfg()
cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
cfg.MODEL.WEIGHTS ="/y/evacheng/weights/ptRend_all_lr_0.01_relation_0.1_other_0.1_new_grasp_0054999.pth"

# Create predictor
predictor = DefaultPredictor(cfg)

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("100DOH_hand_VAL", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "100DOH_hand_VAL")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)
