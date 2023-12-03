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

import logging, sys
import os
from collections import OrderedDict
import torch

from datetime import timedelta

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetCatalog
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.projects.point_rend import add_pointrend_config

from hodetector.data import register_ho_pascal_voc, hoMapper
#from hodetector.modeling.roi_heads.roi_heads import get_ratio
from hodetector.modeling import roi_heads

import matplotlib.pyplot as plt
import numpy as np
import pdb

# _datasets_root = "/home/evacheng/allMerged7Blur/"



_datasets_root = "/home/evacheng/datasets/SAM/"

#split changed based on dataset //Feb 5 2023
for d in ["TRAIN"]:
    register_ho_pascal_voc(name=f'100DOH_hand_{d}',dirname=_datasets_root, year = 2007, split=d, json_file=os.path.join(_datasets_root, "annotations", d.lower()+".json"), class_names=["hand", "firstobject", "secondobject"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='coco')


# #split changed based on dataset //Feb 5 2023
# for d in ["TRAIN", "VAL"]:
#     register_ho_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand", "targetobject", "secondobject"])
#     MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

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
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
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
        logger = logging.getLogger("detectron2.trainer")
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

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=hoMapper(cfg, False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        print(cfg)
        return build_detection_train_loader(
            cfg, mapper=hoMapper(cfg, True)
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.INPUT.RANDOM_FLIP = "none"

    add_pointrend_config(cfg)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    
    cfg = setup(args)
    print(cfg)
    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    trainer.resume_or_load(resume=True)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )


    
    trainer.train()

    return 0


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
        timeout = timedelta(minutes=60)
        )

