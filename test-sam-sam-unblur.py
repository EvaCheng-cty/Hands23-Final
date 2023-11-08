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
import numpy as np
from collections import defaultdict
import contextlib
import io
from tabulate import tabulate

import itertools
import logging
logger = logging.getLogger(__name__)

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

from detectron2.utils.logger import create_small_table

from collections import defaultdict
# from cos_eval import EPICKEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

from detectron2 import _C



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET

import json
import time

from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.data.ho import load_ho_voc_instances
from hodetector.modeling import roi_heads

from html_editor import write_html, write_separate_html, write_obj_html



_datasets_root = "/launch/evacheng/datasets"

mask_source = "SAM"
# mask_source = "Ayda"

#image_source = "val"
#image_source = "val_unblur"
image_source = "test_unblur"

register_coco_instances(name = '100DOH_hand_VAL', metadata = {}, json_file = os.path.join(_datasets_root, "annotations", mask_source, "test.json"), image_root = os.path.join(_datasets_root, "test"))
MetadataCatalog.get(f'100DOH_hand_VAL').set(evaluator_type='coco')


class CosCOCOEVAL(COCOeval_opt):
    
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = CosParams(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            

class CosParams:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = CosParams(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            

def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    cocoeval_fn=CosCOCOEVAL,
    img_ids=None,
    max_dets_per_image=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = CosCOCOEVAL(coco_gt, coco_dt, iou_type)
    #     
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        # In the case that user supplies a custom input for max_dets_per_image,
        # apply COCOevalMaxDets to evaluate AP with the custom input.
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


class CosEvaluator(COCOEvaluator):

    # def __init__(
    #     self,
    #     dataset_name,
    #     tasks=None,
    #     distributed=True,
    #     output_dir=None,
    #     *,
    #     max_dets_per_image=None,
    #     use_fast_impl=True,
    #     kpt_oks_sigmas=(),
    #     allow_cached_coco=True,
    # ):
    #     """
    #     Args:
    #         dataset_name (str): name of the dataset to be evaluated.
    #             It must have either the following corresponding metadata:

    #                 "json_file": the path to the COCO format annotation

    #             Or it must be in detectron2's standard dataset format
    #             so it can be converted to COCO format automatically.
    #         tasks (tuple[str]): tasks that can be evaluated under the given
    #             configuration. A task is one of "bbox", "segm", "keypoints".
    #             By default, will infer this automatically from predictions.
    #         distributed (True): if True, will collect results from all ranks and run evaluation
    #             in the main process.
    #             Otherwise, will only evaluate the results in the current process.
    #         output_dir (str): optional, an output directory to dump all
    #             results predicted on the dataset. The dump contains two files:

    #             1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
    #                contains all the results in the format they are produced by the model.
    #             2. "coco_instances_results.json" a json file in COCO's result format.
    #         max_dets_per_image (int): limit on the maximum number of detections per image.
    #             By default in COCO, this limit is to 100, but this can be customized
    #             to be greater, as is needed in evaluation metrics AP fixed and AP pool
    #             (see https://arxiv.org/pdf/2102.01066.pdf)
    #             This doesn't affect keypoint evaluation.
    #         use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
    #             Although the results should be very close to the official implementation in COCO
    #             API, it is still recommended to compute results with the official API for use in
    #             papers. The faster implementation also uses more RAM.
    #         kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
    #             See http://cocodataset.org/#keypoints-eval
    #             When empty, it will use the defaults in COCO.
    #             Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
    #         allow_cached_coco (bool): Whether to use cached coco json from previous validation
    #             runs. You should set this to False if you need to use different validation data.
    #             Defaults to True.
    #     """
    #     self._logger = logging.getLogger(__name__)
    #     self._distributed = distributed
    #     self._output_dir = output_dir

    #     if use_fast_impl and (COCOeval_opt is COCOeval):
    #         self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
    #         use_fast_impl = False
    #     self._use_fast_impl = use_fast_impl

    #     # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
    #     # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
    #     # 3rd element (100) is used as the limit on the number of detections per image when
    #     # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
    #     # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
    #     if max_dets_per_image is None:
    #         max_dets_per_image = [1, 10, 100]
    #     else:
    #         max_dets_per_image = [1, 10, max_dets_per_image]
    #     self._max_dets_per_image = max_dets_per_image

    #     if tasks is not None and isinstance(tasks, CfgNode):
    #         kpt_oks_sigmas = (
    #             tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
    #         )
    #         self._logger.warn(
    #             "COCO Evaluator instantiated using config, this is deprecated behavior."
    #             " Please pass in explicit arguments instead."
    #         )
    #         self._tasks = None  # Infering it from predictions should be better
    #     else:
    #         self._tasks = tasks

    #     self._cpu_device = torch.device("cpu")

    #     self._metadata = MetadataCatalog.get(dataset_name)
    #     if not hasattr(self._metadata, "json_file"):
    #         if output_dir is None:
    #             raise ValueError(
    #                 "output_dir must be provided to COCOEvaluator "
    #                 "for datasets not in COCO format."
    #             )
    #         self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

    #         cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
    #         self._metadata.json_file = cache_path
    #         convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

    #     json_file = PathManager.get_local_path(self._metadata.json_file)
    #     with contextlib.redirect_stdout(io.StringIO()):
    #         self._coco_api = COCO(json_file)

    #     # Test set json files do not contain annotations (evaluation must be
    #     # performed using the COCO evaluation server).
    #     self._do_evaluation = "annotations" in self._coco_api.dataset
    #     if self._do_evaluation:
    #         self._kpt_oks_sigmas = kpt_oks_sigmas

            
    # def evaluate(self, img_ids=None):
    #     """
    #     Args:
    #         img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
    #     """
    #     if self._distributed:
    #         comm.synchronize()
    #         predictions = comm.gather(self._predictions, dst=0)
    #         predictions = list(itertools.chain(*predictions))

    #         if not comm.is_main_process():
    #             return {}
    #     else:
    #         predictions = self._predictions

    #     if len(predictions) == 0:
    #         self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
    #         return {}

    #     if self._output_dir:
    #         PathManager.mkdirs(self._output_dir)
    #         file_path = os.path.join(self._output_dir, "instances_predictions.pth")
    #         with PathManager.open(file_path, "wb") as f:
    #             torch.save(predictions, f)
        
    #     #
    #     # pdb.set_trace()

    #     self._results = OrderedDict()
    #     if "proposals" in predictions[0]:
    #         self._eval_box_proposals(predictions)
    #     if "instances" in predictions[0]:
    #         self._eval_predictions(predictions, img_ids=img_ids)
    #     # Copy so the caller can do whatever with results
    #     return copy.deepcopy(self._results)
    
    
    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )


        #pdb.set_trace()
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
          
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=CosCOCOEVAL,#if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            # pdb.set_trace()

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

      
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]


        # pdb.set_trace()

        results_for_output = {}

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        
        results_for_output["AP50"] = results_per_category
        # pdb.set_trace()

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP-50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        
        results_for_output["mAP"] = results_per_category
        
        # pdb.set_trace()

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        f = open(os.path.join( "/home/evacheng/400k_sam_on_sam_test/", "metrics.json"), 'w')
        json.dump(results_for_output, f, indent=4)
        return results

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):


        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

      
        
        
        if evaluator_type == "coco":
            evaluator_list = [
                # choose the task you want to evaluate below, and indicate the corresponding dataset used in the config file
                
                #EPICKEvaluator('epick_visor_2022_val_hos', output_dir=output_folder, eval_task='obj_box'),
                # EPICKEvaluator('epick_visor_2022_val_handside', output_dir=output_folder, eval_task='handside'),
                # EPICKEvaluator('epick_visor_2022_val_contact', output_dir=output_folder, eval_task='contact'),
                # EPICKEvaluator('epick_visor_2022_val_combineHO', output_dir=output_folder, eval_task='combineHO'),
                #COCOEvaluator('100DOH_hand_VAL', output_dir=output_folder)
                CosEvaluator('100DOH_hand_VAL')
                ]
            
            return DatasetEvaluators(evaluator_list)


   


def main(args):

    #load the config file, configure the threshold value, load weights 
    cfg = get_cfg()
    
    cfg.HAND = 0.8
    cfg.FIRSTOBJ = 0.3
    cfg.SECONDOBJ = 0.3

    cfg.HAND_RELA = 0.3
    cfg.OBJ_RELA = 0.3

    cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    cfg.OUTPUT_DIR = "/home/evacheng/400k_sam_on_sam_test_unblur/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = "/y/evacheng/final_weights/sam_blur_1_model_0399999.pth"
    cfg.freeze()

    default_setup(cfg, args)

    model = Trainer.build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=False
            )
    
    res = Trainer.test(cfg, model)
   
    verify_results(cfg, res)
    return res

  
if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    # class_names = MetadataCatalog.get('100DOH_hand_train').thing_classes

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
        )

