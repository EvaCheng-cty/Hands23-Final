
from cgi import test
import os
import torch
import cv2
import random
import numpy as np
import pdb
import copy
import argparse
import glob

import json
import shutil

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


from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET


from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.modeling import roi_heads

from html_editor import write_html, write_separate_html, write_obj_html


def main():
    json_file_dir = ""


if __name__ == '__main__':
    main()
