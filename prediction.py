# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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



# _datasets_root = "/y/dywsjtu/data/VOCdevkit2007_handobj_100K/VOC2007"
_datasets_root = "/w/fouhey/hands2/allMerged4/"
for d in ["train", "val"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand", "targetobject"])
    metadata = MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')

save_dir = '/home/evacheng/public_html/index.html/6/'

second_objs_dir = "./second_obj.txt"



left_hand_color = (181, 90, 0)
right_hand_color = (32, 50, 220)
first_obj_color = (10, 194, 255)
second_object_bbox_color = (115, 159, 0)

def tell_grasp(grasp_type):
    if grasp_type == 0:
        return "NP-Palm"
    elif grasp_type == 1:
        return  "NP-Fin"
    elif grasp_type == 2:
        return "Pow-Pris"
    elif grasp_type == 3:
        return "Pre-Pris"
    elif grasp_type == 4:
        return "Pow-Circ"
    elif grasp_type == 5:
        return "Pre-Circ"
    elif grasp_type == 6:
        return  "Later"
    # elif grasp_type == 7:
    #     return "Exten"
    elif grasp_type == 7:
        return "Other"
    else:
        pdb.set_trace()

def tell_grasp_clean(grasp_type):
    if grasp_type == 0:
        return "NP-P"
    elif grasp_type == 1:
        return  "NP-F"
    elif grasp_type == 2:
        return "Pow-P"
    elif grasp_type == 3:
        return "Pre-P"
    elif grasp_type == 4:
        return "Pow-C"
    elif grasp_type == 5:
        return "Pre-C"
    elif grasp_type == 6:
        return  "Lat"
    # elif grasp_type == 7:
    #     return "Exten"
    elif grasp_type == 7:
        return "Other"
    else:
        pdb.set_trace()


def parse_grasp(grasp_scores):
    grasp_dict = {}

    for type, score in zip(["NP-Palm","NP-Fin", "Pow-Pris", "Pre-Pris", "Pow-Circ", "Pre-Circ", "Later","Other"], grasp_scores):
        grasp_dict[type] = str(round(score.item(),4))
    
    return grasp_dict

def tell_contact(contact):
    if contact == 0:
        return "no_contact"
    elif contact == 1:
        return "other_person_contact"
    elif contact == 2:
        return "self_contact"
    elif contact == 3:
        return "object_contact"
    elif contact == 4:
        return "objects_contact"
    else:
        pdb.set_trace()


def tell_contact_clean(contact):
    if contact == 0:
        return "no"
    elif contact == 1:
        return "other_p"
    elif contact == 2:
        return "self"
    elif contact == 3:
        return "obj"
    elif contact == 4:
        return "objs"
    else:
        pdb.set_trace()
                
def tell_touch(touch):
    if touch == 0:
        return "tool_,_touched"
    elif touch == 1:
        return "tool_,_held"
    elif touch == 2:
        return "tool_,_used"
    elif touch == 3:
        return "container_,_touched"
    elif touch == 4:
        return "container_,_held"
    elif touch == 5:
        return "neither_,_touched"
    elif touch == 6:
        return "neither_,_held"
    else:
                print("error!")
                pdb.set_trace()


def tell_touch_clean(touch):
    if touch == 0:
        return "T-T"
    elif touch == 1:
        return "T-H"
    elif touch == 2:
        return "T-U"
    elif touch == 3:
        return "C-T"
    elif touch == 4:
        return "C-H"
    elif touch == 5:
        return "N-T"
    elif touch == 6:
        return "N-H"
    else:
                print("error!")
                pdb.set_trace()




def parse_touch(touch_scores):
    touch_dict = {}

    for touch,score in zip(["tool_,_touched", "tool_,_held", "tool_,_used", "container_,_touched", "container_,_held","neither_,_touched", "neither_,_held"], touch_scores):
        touch_dict[touch] = str(round(score.item(),4))

    return touch_dict

class Hands:
    def __init__(self, hand_id, hand_bbox, hand_mask, contactState, hand_side, grasp, pred_score, grasp_scores = None):
        self.id = hand_id
        self.hand_bbox = hand_bbox
        self.contactState = tell_contact(contactState) if not isinstance(contactState, str) else contactState
        self.contactState_clean = tell_grasp_clean(contactState) 
        self.hand_side = ("right_hand" if hand_side==1 else "left_hand") if not isinstance(hand_side, str) else hand_side
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_score = None
        self.second_obj_bbox = None
        self.grasp = tell_grasp(grasp) if not isinstance(grasp, str) else grasp
        self.grasp_clean = tell_grasp_clean(grasp)
        self.grasp_scores = grasp_scores
        self.hand_mask = hand_mask
        self.pred_score = round(pred_score,2)
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_clean = None
        self.obj_masks = None
        self.second_obj_bbox = None
        self.second_obj_masks = None
        self.has_first = False
        self.has_second = False
        self.obj_pred_score = None
        self.sec_obj_pred_score = None

    def set_first_obj(self, obj_bbox , obj_touch , obj_masks, pred_score, touch_scores = None):
        self.obj_bbox = obj_bbox
        self.obj_touch = tell_touch(obj_touch)
        self.obj_touch_clean = tell_touch(obj_touch)
        self.obj_masks = obj_masks
        self.obj_pred_score = round(pred_score,2)
        self.has_first  = True
        self.obj_touch_score = touch_scores
         
    def set_second_obj(self, obj_bbox, obj_masks, pred_score):
        self.second_obj_bbox = obj_bbox
        self.second_obj_masks = obj_masks
        self.sec_obj_pred_score = round(pred_score,2)
        self.has_second = True
    
    def message(self):
        return str(self.hand_side) + ' | ' + str(self.contactState) + ' | ' + str(self.hand_bbox).replace('[','').replace(']','') + ' | ' +  str(self.obj_bbox).replace('[','').replace(']','') + ' | ' +str(self.obj_touch) + ' | ' + str(self.second_obj_bbox).replace('[','').replace(']','') + " | " + str(self.grasp) +'\n'

    def save_masks(self, save_dir, im, img_id):
        ims = copy.deepcopy(im)
        ims[:,:,:] = 0
        ims[self.hand_mask, :] = 255

        cv2.imwrite(save_dir+"2_"+str(self.id)+"_"+img_id, ims)

        if self.has_first:
            ims[:,:,:] = 0
            ims[self.obj_masks, :] = 255
            cv2.imwrite(save_dir+"3_"+str(self.id)+"_"+img_id, ims)

            if self.has_second:
                ims[:,:,:] = 0
                ims[self.second_obj_masks, :] = 255
                cv2.imwrite(save_dir+"5_"+str(self.id)+"_"+img_id, ims)
    
    def vis(self, im):
        curr_box = self.hand_bbox
        hand_color =  left_hand_color if self.hand_side == "left_hand" else right_hand_color
        im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), hand_color, 2)   
        cv2.putText(im, str(self.id), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, str(self.grasp) + " " + str(round(self.pred_score,2)), (int(curr_box[0])+20, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        cv2.putText(im, self.contactState, (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        blendMask(im, self.hand_mask, hand_color, 0.7)
        
        if self.has_first:
            object_box = self.obj_bbox
            im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), first_obj_color , 2) 
            cv2.putText(im, str(self.id), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)   
            im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), first_obj_color, 2)
                
            # cv2.putText(im, str(round(scores[i].item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            blendMask(im, self.obj_masks, first_obj_color, 0.7)

            if self.has_second:
                curr_box = self.obj_bbox
                object_box = self.second_obj_bbox
                im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), second_object_bbox_color , 2)   
                cv2.putText(im, str(self.id), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)  
                im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), second_object_bbox_color, 2)
                blendMask(im, self.second_obj_masks,second_object_bbox_color, 0.7)

        return im
    

    def vis_clean(self, im):
        curr_box = self.hand_bbox
        hand_color =  left_hand_color if self.hand_side == "left_hand" else right_hand_color
        
        im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), hand_color, 2)   
        cv2.putText(im, str(self.id), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, str(self.grasp_clean) + " " + str(round(self.pred_score,2)), (int(curr_box[0])+20, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, self.contactState_clean, (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        blendMask(im, self.hand_mask, hand_color, 0.7)
        
        if self.has_first:
            object_box = self.obj_bbox
            im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), first_obj_color , 2) 
            cv2.putText(im, str(self.id), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)
            cv2.putText(im, str(self.obj_touch_clean), (int(object_box[0])+20, int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)    
            im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), first_obj_color, 2)
                
            # cv2.putText(im, str(round(scores[i].item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            blendMask(im, self.obj_masks, first_obj_color, 0.7)

            if self.has_second:
                curr_box = self.obj_bbox
                object_box = self.second_obj_bbox
                im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), second_object_bbox_color , 2)   
                cv2.putText(im, str(self.id), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)  
                im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), second_object_bbox_color, 2)
                blendMask(im, self.second_obj_masks,second_object_bbox_color, 0.7)

        return im
    
    
    def get_dict(self, img):
        img["hands"].append(
            {"id":self.id,
             "hand_side": self.hand_side,
             "pred_score": str(self.pred_score),
             "grasp": self.grasp,
             "contact": self.contactState,
            }
        )

        if self.has_first:
            img["fir_objs"].append(
                {"id": self.id,
                 "pred_score": str(self.obj_pred_score),
                 "touch": self.obj_touch,
                }
            )

            if self.has_second:
                img["sec_objs"].append(
                    {"id": self.id,
                     "pred_score": str(self.sec_obj_pred_score),
                    }
                )
        
        return img

    def get_json(self):
        info = {}
        info['hand_id'] = self.id 
        info['hand_bbox'] = [str(x) for x in self.hand_bbox]
        info['contact_state'] = self.contactState
        info['hand_side'] = self.hand_side
        info['obj_bbox'] = [str(x) for x in self.obj_bbox] if self.obj_bbox is not None else None
        info['obj_touch'] = str(self.obj_touch)
        info['obj_touch_scores'] = parse_touch(self.obj_touch_score) if self.has_first else None
        info['second_obj_bbox'] = [str(x) for x in self.second_obj_bbox]  if self.second_obj_bbox is not None else None
        info['grasp'] = self.grasp
        info['grasp_scores'] = parse_grasp(self.grasp_scores) 
        info['hand_pred_score'] = str(self.pred_score)
        info['obj_pred_score'] = str(self.obj_pred_score)
        info['sec_obj_pred_score'] = str(self.sec_obj_pred_score)

        return info

    @classmethod
    def from_json(cls, data):
        hand_id = data["hand_id"]
        hand_bbox = [float(x) for x in data["hand_bbox"]]
        contactState = data["contact_state"]
        hand_side = data["hand_side"]
        hand_pred_score =  float(data["hand_pred_score"])
        grasp = data["grasp"]
        grasp_score = {}
        for key in data["grasp_scores"].keys():
            grasp_score[key] = float(data["grasp_scores"][key])


        hand = cls(hand_id = hand_id, hand_bbox = hand_bbox, hand_mask = None, contactState = contactState, hand_side = hand_side, grasp = grasp, pred_score = hand_pred_score, grasp_scores = grasp_score)

        if isinstance(data["obj_bbox"], list):
            obj_bbox = [float(x) for x in data["obj_bbox"]]
            obj_touch = data["obj_touch"]
            obj_pred_scores = float(data["obj_pred_score"])
            touch_score = {}
            for key in data["obj_touch_scores"].keys():
                touch_score[key] = float(data["obj_touch_scores"][key])

            hand.set_first_obj(obj_bbox=obj_bbox, obj_touch=obj_touch, obj_masks=None, pred_score=obj_pred_scores, touch_scores=touch_score)

            if data["second_obj_bbox"] != "null":
                second_obj_bbox = [float(x) for x in data["second_obj_bbox"]]
                second_object_pred_score = float(data["sec_obj_pred_score"])
                hand.set_second_obj(obj_bbox = second_obj_bbox, obj_masks=None, pred_score=second_object_pred_score)
        return hand



def blendMask(I,mask,color, alpha):
    if mask == None:
        return
    
    for c in range(3):
        try:
            Ic = I[:,:,c]
            Ic[mask] = ((Ic[mask].astype(np.float32)*alpha) + (float(color[c])*(1-alpha))).astype(np.uint8)
            I[:,:,c] = Ic
        except:
            pdb.set_trace()



def deal_output(im, predictor):
    outputs = predictor(im)

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
    pred_classes =  outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()


    interaction = torch.tensor(pred_dz[:, 4])
    hand_side = torch.tensor(pred_dz[:, 5])
    grasp =  torch.tensor(pred_dz[:, 6])
    touch_type = torch.tensor(pred_dz[:, 7])
    contact_state = torch.tensor(pred_dz[:,8])

    scores = torch.tensor(pred_dz[:,9])
    grasp_scores = torch.tensor(pred_dz[:,10:18])
    touch_scores = torch.tensor(pred_dz[:,18:25])

    hand_list = []
    anno_list = []

    count = 0

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:
            curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], contactState=int(contact_state[i].item()),hand_side=hand_side[i].item(), grasp = grasp[i].item(), pred_score= pred_scores[i], grasp_scores= grasp_scores[i])

            if interaction[i] >=0:
                obj_id = int(interaction[i])

                curr_hand.set_first_obj(obj_bbox=pred_boxes[obj_id], obj_touch= touch_type[obj_id].item(), obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], touch_scores=touch_scores[obj_id])

                if interaction[obj_id] >=0:
                    second_obj_id = int(interaction[obj_id])

                    curr_hand.set_second_obj(obj_bbox=pred_boxes[second_obj_id], obj_masks=pred_masks[second_obj_id], pred_score= pred_scores[second_obj_id])

            hand_list.append(curr_hand)
            anno_list.append(curr_hand.message())

    return hand_list, anno_list



def set_cfg(args, model_weights = None):

    cfg = get_cfg()
    cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")

    if args.model_weights is not None:
        cfg.MODEL.WEIGHTS = args.model_weights
    else:
        cfg.MODEL.WEIGHTS = "/y/evacheng/weights/ptRend_all_lr_0.01_relation_0.1_other_0.1_new_grasp_0054999.pth" if model_weights is None else model_weights
    
    thresh = args.thresh if args.thresh is not None else 0.3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh

    hand_thresh = args.hand_thresh if args.hand_thresh is not None else thresh
    cfg.HAND = hand_thresh

    first_obj_thresh = args.first_obj_thresh if args.first_obj_thresh is not None else thresh
    cfg.FIRSTOBJ = first_obj_thresh

    second_obj_thresh = args.second_obj_thresh if args.second_obj_thresh is not None else thresh
    cfg.SECONDOBJ = second_obj_thresh


    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 3000

    cfg.freeze()

    return cfg

def init_img(id):
    img = {}
    img["img_id"] = id
    img["hands"] = []
    img["fir_objs"] = []
    img["sec_objs"] = []

    return img


def model_prediction(cfg, save_dir, test_on_blur_img = True, split_file_dir = "/w/fouhey/hands2/allMerged7Splits/VAL.txt", mess = None):

    blur_img_dir = "/w/fouhey/hands2/allMerged7Blur/"
    not_blur_img_dir = "/w/fouhey/hands2/allMerged7/"

    img_dir = blur_img_dir if test_on_blur_img else not_blur_img_dir
    
    predictor = DefaultPredictor(cfg)
    os.makedirs(save_dir, exist_ok=True)
    
    
    split_file = open(split_file_dir, "r")

    images = split_file.readlines()
    images = sorted(images)

    random.seed(42)
    random.shuffle(images)

    images = images[:2000]
           


    vis = {}
    vis["save_dir"] = save_dir
    vis["images"] = []


      
    
    for test_img in images:

        im = cv2.imread(img_dir+test_img.replace('\n', ''))

        # img = init_img(test_img.replace(".jpg","").replace('\n',''))

        img = {}
        img["file_name"] = test_img.replace(".jpg","").replace('\n','')
        img["predictions"] = []

        #convert model outputs into predictions
        #where hand_lists is a list of Hand objects
        #anno_lists is a list of predictions - 1 line for each hand
        hand_lists, anno_list = deal_output(im = im, predictor= predictor)

        #save the prediction in corresponding txt file
        f = open(save_dir+test_img.replace("\n","")+ mess +".txt", "w+")
        f.writelines(anno_list)

        #save the masks and process the final visualization image
        for hands in hand_lists:
            hands.save_masks(save_dir, im, test_img)
            im = hands.vis(im)
            # img = hands.get_dict(img)
            img['predictions'].append(hands.get_json())
        
        # save the image
        cv2.imwrite(save_dir+test_img.replace('.jpg','').replace('\n','')+mess+'.jpg', im)
        vis["images"].append(img)

    
    f = open(save_dir+"results"+mess+".json", 'w')
    json.dump(vis, f, indent=4)


def parse_model_list(t_):
    t = ''
    if t_== 'Blur on Blur':
               t = '-b-b'
    elif t_== 'Blur on not Blur':
               t = '-b-not-b'
    elif t_ == 'not Blur on Blur':
               t = '-not-b-b'
    elif t_ == 'not Blur on not Blur':
               t = '-not-b-not-b'
    return t


def parse_sam(t_):
        t = ''
        t =  "-sam" if "Sam" in t_ else "-all"
        return t


def write_compare_model(file_dir, img_list, link_list = [], curr_folder = '', model_list = ["Blur on Blur", "Blur on not Blur", "not Blur on Blur", "not Blur on not Blur"], model_list_parser = parse_model_list,  message = ""):
   
   file_dir_ =  os.path.join(file_dir , curr_folder, 'index.html')
   f = open(file_dir_, "w+")
   f.truncate(0)
  
  
   html = ET.Element('html')
   body = ET.SubElement(html, 'body')

   h2_2 = ET.SubElement(body, 'h2')
   h2_2.text = "-" + message

   if len(link_list) >0:
       for link in link_list:
            next_im = ET.SubElement(body, 'form')
            next_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/" + file_dir.replace("/home/evacheng/public_html/","")+ link + "/index.html") 
            input = ET.SubElement(next_im, "input")
            input.set("type", "submit")
            input.set("value", str(link))

   table = ET.SubElement(body, "table")
   table.set("width", "100%")
   table.set("border", "1px solid black")
   table.set("border-collapse", "collapse")

   width = str(int(100/len(model_list))) +"%"

   
   tr = ET.SubElement(table, "tr")
   for t in model_list:
       td = ET.SubElement(tr, "th")
       td.text = str(t)

       td.set("width", width)
       td.set("border", "1px solid black")
       td.set("border-collapse", "collapse")
       
    
   for i in range(len(img_list)):
       tr = ET.SubElement(table, "tr")
 
       for t_ in model_list:
           
           t = model_list_parser(t_)
           
           if os.path.exists(file_dir+str(img_list[i]).replace(".jpg","") + t +".jpg"):
                tb = ET.SubElement(tr, "td")

                img = ET.SubElement(tb, "image")
                img.set("src", "./" + str(img_list[i]).replace(".jpg","") + t +".jpg")
                
                img.set("width", "100%")
                img.set("border", "1px solid black")
                img.set("border-collapse", "collapse")

                if curr_folder != '':
                    shutil.copy(file_dir+ str(img_list[i]).replace(".jpg","") + t +".jpg",  os.path.join(file_dir , curr_folder,  os.path.join(file_dir , curr_folder, str(img_list[i]).replace(".jpg","") + t +".jpg")))
           else:
               print(str(img_list[i]).replace(".jpg","") + t +".jpg")
       
     
   tree = ET.ElementTree(html)
   ET.indent(tree, space='\t', level=0)
   tree.write(open(file_dir_, 'wb'))

def write_separate_compare_model(file_dir, img_list ,model_list = ["Blur on Blur", "Blur on not Blur", "not Blur on Blur", "not Blur on not Blur"], message = ""):
   

   for i in range(len(img_list)):
        img = img_list[i]
        save_dir = os.path.join(file_dir, img.replace('.jpg',''))
        os.makedirs(save_dir, exist_ok=True)

   
   
        file_dir_ = save_dir + '/index.html'
        f = open(file_dir_, "w+")
        f.truncate(0)


  
  
        html = ET.Element('html')
        body = ET.SubElement(html, 'body')

        h2_2 = ET.SubElement(body, 'h2')
        h2_2.text = "-" + message

        prev_im_id = (img_list[-1] if i ==0 else img_list[i-1]).replace('.jpg','')

        

        if prev_im_id is not None:
            prev_im = ET.SubElement(body, 'form')
            prev_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/" + file_dir.replace("/home/evacheng/public_html/","") +prev_im_id +'/index.html') 
            input = ET.SubElement(prev_im, "input")
            input.set("type", "submit")
            input.set("value", "Go to previous image")

        next_im_id = (img_list[i+1] if i+1 <len(img_list) else img_list[0]).replace('.jpg','')
      
        if next_im_id is not None:
            next_im = ET.SubElement(body, 'form')
            next_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/" + file_dir.replace("/home/evacheng/public_html/","") +next_im_id+'/index.html') 
            input = ET.SubElement(next_im, "input")
            input.set("type", "submit")
            input.set("value", "Go to next image")
        
        

        table = ET.SubElement(body, "table")
        table.set("width", "100%")
        table.set("border", "1px solid black")
        table.set("border-collapse", "collapse")

   
        tr = ET.SubElement(table, "tr")
        for t_ in model_list:
            td = ET.SubElement(tr, "th")
            td.text = str(t_)

            td.set("width", "25%")
            td.set("border", "1px solid black")
            td.set("border-collapse", "collapse")


        
        
        tr = ET.SubElement(table, "tr")
        for t_ in model_list:

            t = ''
            
            if t_== 'Blur on Blur':
                t = '-b-b'
            elif t_== 'Blur on not Blur':
                t = '-b-not-b'
            elif t_ == 'not Blur on Blur':
                t = '-not-b-b'
            elif t_ == 'not Blur on not Blur':
                t = '-not-b-not-b'
            
            if os.path.exists(file_dir+str(img_list[i]).replace(".jpg","") + t +".jpg"):
                    tb = ET.SubElement(tr, "td")

                    org_img_dir = file_dir + str(img_list[i]).replace(".jpg","") + t +".jpg"
                    curr_im_dir = save_dir + '/'+ str(img_list[i]).replace(".jpg","") + t +".jpg"
                   
                    # pdb.set_trace()
                    shutil.copy(src=org_img_dir, dst=curr_im_dir)

                    img = ET.SubElement(tb, "image")
                    img.set("src", '.' +curr_im_dir.replace(save_dir, ''))
                    
                    img.set("width", "100%")
                    img.set("border", "1px solid black")
                    img.set("border-collapse", "collapse")
            else:
                print(str(img).replace(".jpg","") + t +".jpg")
       
     
        tree = ET.ElementTree(html)
        ET.indent(tree, space='\t', level=0)
        tree.write(open(file_dir_, 'wb'))

def write_batch_compare_model(file_dir, img_list ,model_list = ["Blur on Blur", "Blur on not Blur", "not Blur on Blur", "not Blur on not Blur"], model_list_parser = parse_model_list, message = "", split_size = 100):
    num_pages = int(len(img_list)/split_size)

    dir_list = []

    for i in range(num_pages):
        save_dir = os.path.join(file_dir, str(i*split_size).zfill(10))
        os.makedirs(save_dir, exist_ok=True)

        dir_list.append(str(i*split_size).zfill(10))
    

    for i in range(num_pages):
        write_compare_model(file_dir=file_dir, img_list = img_list[i*split_size:(i+1)*split_size], link_list=dir_list, curr_folder=dir_list[i], model_list= model_list, model_list_parser=model_list_parser)





def sam_versus_org(args, file_dir):

    for model in ["/y/evacheng/weights/final_on_blur_model_0174999.pth", "/y/evacheng/weights/apr_on_sam_model_0174999.pth"]:
        # for test_im in [True, False]:

            cfg = set_cfg(args, model_weights=model)
            on_blur = '-sam' if "sam" not in model else '-all'
            # test_set = '-b' if test_im else '-not-b'
            model_prediction(cfg = cfg, save_dir=file_dir, test_on_blur_img= True, split_file_dir="/w/fouhey/hands2/allMerged7Splits/VAL.txt", mess = on_blur)

    img_list = []

    for file in glob.glob(file_dir +"*.jpg"):
        img = file.replace(file_dir,"")

        img_name = img.split('-')[0]+'.jpg'
        if img_name not in img_list:
            img_list.append(img_name)
    

    
    write_batch_compare_model(file_dir = file_dir, img_list= img_list, model_list=["Trained on Sam", "Trained on Blur"], model_list_parser= parse_sam)
    # write_separate_compare_model(file_dir='/home/evacheng/public_html/index.html/blur_not_blur_100k/', img_list=img_list)





def main():

    
    parser = argparse.ArgumentParser()

    #bounding box score threshs for a proposed bounding box to be considered
    parser.add_argument("--thresh") 
    parser.add_argument("--hand_thresh")
    parser.add_argument("--first_obj_thresh")
    parser.add_argument("--second_obj_thresh")

    parser.add_argument("--model_weights")

    args = parser.parse_args()

    file_dir = "/home/evacheng/public_html/index.html/sam_all_175k/"

    sam_versus_org(args=args, file_dir = file_dir)

    #for model in ["/y/evacheng/final_weights/blur_model_0229999.pth","/y/evacheng/final_weights/not_blur_model_0229999.pth"]:
    #for model in ["/y/evacheng/final_weights/not_blur_model_0229999.pth"]:
    #     for test_im in [True, False]:

    #         cfg = set_cfg(args, model_weights=model)
    #         on_blur = '-b' if "not" not in model else '-not-b'
    #         test_set = '-b' if test_im else '-not-b'
    #         model_prediction(cfg = cfg, save_dir=file_dir, test_on_blur_img= test_im, split_file_dir="/w/fouhey/hands2/allMerged7Splits/VAL.txt", mess = on_blur+test_set)

    # img_list = []

    # for file in glob.glob(file_dir +"*.jpg"):
    #     img = file.replace(file_dir,"")

    #     img_name = img.split('-')[0]+'.jpg'
    #     if img_name not in img_list:
    #         img_list.append(img_name)
    #write_batch_compare_model(file_dir='/home/evacheng/public_html/index.html/blur_not_blur_230k/', img_list= img_list)
    # write_separate_compare_model(file_dir='/home/evacheng/public_html/index.html/blur_not_blur_100k/', img_list=img_list)



  




        




   

if __name__ == '__main__':
    main()
