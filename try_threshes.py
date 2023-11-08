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

import json

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


# hand_bbox_color = (255,0,0)
# object_bbox_color = (0,0, 255)
# second_object_bbox_color = (255, 200, 0)

# hand_object_relation_color = (0, 255, 0)
# tool_second_object_relation_color = (255, 80, 255)

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

def tell_contact(contact):
    if contact == 0:
        return "no_contact"
    elif contact == 1:
        return "other_person_contact"
    elif contact == 2:
        return "self_contact"
    elif contact == 3:
        return "object_contact"
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
        return "TT"
    elif touch == 1:
        return "TH"
    elif touch == 2:
        return "TU"
    elif touch == 3:
        return "CT"
    elif touch == 4:
        return "CH"
    elif touch == 5:
        return "NT"
    elif touch == 6:
        return "NH"
    else:
                print("error!")
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
        


def parse_grasp(grasp_scores):
    grasp_dict = {}

    for type, score in zip(["NP-Palm","NP-Fin", "Pow-Pris", "Pre-Pris", "Pow-Circ", "Pre-Circ", "Later","Other"], grasp_scores):
        grasp_dict[type] = str(round(score.item(),4))
    
    return grasp_dict



def parse_touch(touch_scores):
    touch_dict = {}

    for touch,score in zip(["tool_,_touched", "tool_,_held", "tool_,_used", "container_,_touched", "container_,_held","neither_,_touched", "neither_,_held"], touch_scores):
        touch_dict[touch] = str(round(score.item(),4))

    return touch_dict


class Hands:
    def __init__(self, hand_id, hand_bbox, hand_mask, contactState, hand_side, grasp, pred_score, grasp_scores = None, hand_inter_score = None, obj_inter_score = None):
        self.id = hand_id
        self.hand_bbox = hand_bbox
        self.contactState = tell_contact(contactState)
        self.contactState_clean = tell_grasp_clean(contactState)
        self.hand_side = "R" if hand_side==1 else "L"
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_score = None
        self.second_obj_bbox = None
        self.grasp = tell_grasp(grasp)
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
        self.hand_inter_score = None
        self.obj_inter_score = None
    
    def set_first_obj(self, obj_bbox , obj_touch , obj_masks, pred_score, touch_scores = None, hand_inter_score = None):
        self.obj_bbox = obj_bbox
        self.obj_touch = tell_touch(obj_touch)
        self.obj_touch_clean = tell_touch_clean(obj_touch)
        self.obj_masks = obj_masks
        self.obj_pred_score = round(pred_score,2)
        self.has_first  = True
        self.obj_touch_score = touch_scores
        self.hand_inter_score = hand_inter_score
         
    def set_second_obj(self, obj_bbox, obj_masks, pred_score, obj_inter_score = None):
        self.second_obj_bbox = obj_bbox
        self.second_obj_masks = obj_masks
        self.sec_obj_pred_score = round(pred_score,2)
        self.has_second = True
        self.obj_inter_score = obj_inter_score
    
    def message(self):
        return str(self.hand_side) + ' | ' + str(self.contactState) + ' | ' + str(self.hand_bbox).replace('[','').replace(']','') + ' | ' +  str(self.obj_bbox).replace('[','').replace(']','') + ' | ' +str(self.obj_touch) + ' | ' + str(self.second_obj_bbox).replace('[','').replace(']','') + " | " + str(self.grasp) +'\n'

    def save_masks(self, save_dir, im, img_id, mess = None):
        ims = copy.deepcopy(im)
        ims[:,:,:] = 0
        ims[self.hand_mask, :] = 255

        # pdb.set_trace()
        img_id = img_id.strip('\n')
        
        if mess != None:
            save_dir = os.path.join(save_dir, "masks"+mess)
            # file_name = os.path.join(save_dir, "2_"+str(self.id)+"_"+img_id+".json" )
            os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(save_dir+'/2_'+str(self.id)+'_'+img_id, ims)

        if self.has_first:
            ims[:,:,:] = 0
            ims[self.obj_masks, :] = 255
            cv2.imwrite(save_dir+'/3_'+str(self.id)+'_'+img_id, ims)

            if self.has_second:
                ims[:,:,:] = 0
                ims[self.second_obj_masks, :] = 255
                cv2.imwrite(save_dir+'/5_'+str(self.id)+'_'+img_id, ims)
    
    def vis(self, im):
        curr_box = self.hand_bbox
        hand_color =  left_hand_color if self.hand_side == "left_hand" else right_hand_color
        im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), hand_color, 2)   
        #cv2.putText(im, str(self.id), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, str(self.grasp) + " " + str(round(self.pred_score,2)), (int(curr_box[0])+20, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        cv2.putText(im, self.contactState, (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        blendMask(im, self.hand_mask, hand_color, 0.7)
        
        if self.has_first:
            object_box = self.obj_bbox
            im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), first_obj_color , 2) 
            cv2.putText(im, str(self.obj_touch_clean), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)   
            im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), first_obj_color, 2)
                
            cv2.putText(im, str(round(self.hand_inter_score.item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)
            blendMask(im, self.obj_masks, first_obj_color, 0.7)

            if self.has_second:
                curr_box = self.obj_bbox
                object_box = self.second_obj_bbox
                im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), second_object_bbox_color , 2)   
                #cv2.putText(im, str(self.id), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)  
                im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), second_object_bbox_color, 2)
                cv2.putText(im, str(round(self.obj_inter_score.item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)

                blendMask(im, self.second_obj_masks,second_object_bbox_color, 0.7)

        return im
    

    def vis_clean(self, im):
        curr_box = self.hand_bbox
        hand_color =  left_hand_color if self.hand_side == "left_hand" else right_hand_color
        
        im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), hand_color, 2)   
        #cv2.putText(im, str(self.id), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, str(self.grasp_clean) + " " + str(round(self.pred_score,2)), (int(curr_box[0])+20, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2) 
        cv2.putText(im, self.contactState_clean, (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        blendMask(im, self.hand_mask, hand_color, 0.7)
        
        if self.has_first:
            object_box = self.obj_bbox
            im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), first_obj_color , 2) 
            cv2.putText(im, str(round(self.obj_pred_score,2)), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)
            cv2.putText(im, str(self.obj_touch_clean), (int(object_box[0])+20, int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)    
            im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), first_obj_color, 2)
                
            cv2.putText(im, str(round(self.hand_inter_score,2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_obj_color, 2)
            blendMask(im, self.obj_masks, first_obj_color, 0.7)

            if self.has_second:
                curr_box = self.obj_bbox
                object_box = self.second_obj_bbox
                im = cv2.rectangle(im, (int(object_box[0]), int(object_box[1])), (int(object_box[2]), int(object_box[3])), second_object_bbox_color , 2)   
                cv2.putText(im, str(round(self.sec_obj_pred_score,2)), (int(object_box[0]), int(object_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)  
                im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), second_object_bbox_color, 2)
                cv2.putText(im, str(round(self.obj_inter_score,2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2)
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




def blendMask(I,mask,color, alpha):
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
    inter_scores = torch.tensor(pred_dz[:, 25:])

    hand_list = []
    anno_list = []

    count = 0

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:
            curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], contactState=int(contact_state[i].item()),hand_side=hand_side[i].item(), grasp = grasp[i].item(), pred_score= pred_scores[i])

            if interaction[i] >=0:
                obj_id = int(interaction[i])
                curr_hand.set_first_obj(obj_bbox=pred_boxes[obj_id], obj_touch= touch_type[obj_id].item(), obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], touch_scores= touch_scores[obj_id], hand_inter_score= scores[i])

                if interaction[obj_id] >=0:
                    second_obj_id = int(interaction[obj_id])
                    

                    curr_hand.set_second_obj(obj_bbox=pred_boxes[second_obj_id], obj_masks=pred_masks[second_obj_id], pred_score= pred_scores[second_obj_id], obj_inter_score=scores[obj_id])

            hand_list.append(curr_hand)
            anno_list.append(curr_hand.message())

    return hand_list, anno_list



def set_cfg(thresh, hand_thresh, fir_obj_thresh, sec_obj_thresh, hand_rela, obj_rela):

    cfg = get_cfg()
    cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")

  
    cfg.MODEL.WEIGHTS = "/y/evacheng/final_weights/final_on_blur_model_0399999.pth"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thresh)

    cfg.HAND = float(hand_thresh)

    
    cfg.FIRSTOBJ = float(fir_obj_thresh)

    cfg.SECONDOBJ = float(sec_obj_thresh)

    cfg.HAND_RELA = hand_rela
    cfg.OBJ_RELA = obj_rela

    cfg.freeze()

    return cfg

def init_img(id):
    img = {}
    img["img_id"] = id
    img["hands"] = []
    img["fir_objs"] = []
    img["sec_objs"] = []

    return img

def predict(save_dir, thresh, hand_thresh, fir_obj_thresh, sec_obj_thresh, hand_rela, obj_rela, images):
    
    #set configuration
    cfg = set_cfg(thresh, hand_thresh, fir_obj_thresh, sec_obj_thresh, hand_rela, obj_rela)
    predictor = DefaultPredictor(cfg)
    
   
    # save_dir = '/home/evacheng/public_html/index.html/results/400k/obj_0.3_sec_0.3_all_rela_0.3/'
    os.makedirs(save_dir, exist_ok=True)
    
    #
    # split_file = open("/w/fouhey/hands2/allMerged7Splits/VAL.txt", "r")
 
    # images = split_file.readlines()
    # images = sorted(images)

    # random.seed(42)
    # random.shuffle(images)
           
    #can limit the number of images processed
    # images = images[:300]

   
    
    for test_img in images:

        im = cv2.imread("/w/fouhey/hands2/allMerged7Blur/"+test_img.replace('\n', ''))

        #convert model outputs into predictions
        #where hand_lists is a list of Hand objects
        #anno_lists is a list of predictions - 1 line for each hand
        hand_lists, anno_list = deal_output(im = im, predictor= predictor)

        #save the prediction in corresponding txt file
        # f = open(save_dir+test_img.replace("\n","") +".txt", "w+")
        # f.writelines(anno_list)

        #save the masks and process the final visualization image
        for hands in hand_lists:
            im = hands.vis(im)
            
        
        # pdb.set_trace()
        
        # save the image
       
        cv2.imwrite(os.path.join(save_dir, test_img.strip('\n')), im)
           
       
    write_obj_html(message="", image_dir=save_dir, thresh_list=[0.01, 0.1, 0.3, 0.5, 0.7], root_dir='/home/evacheng/public_html/index.html/results/400k/')

def main():
   
    split_file = open("/w/fouhey/hands2/allMerged7Splits/VAL.txt", "r")
 
    images = split_file.readlines()
    images = sorted(images)

    random.seed(42)
    random.shuffle(images)
    image_lists = []
    # image_lists = ["ND_4XXQwpGCu1s_frame008074","EK_0128_P08_16_frame_0000009061",'ND_bQ3MXpMsHqY_frame015250', 'ND_Zco7MCgmEnI_frame024601', "ND_Zco7MCgmEnI_frame024601", "ND_SW0RgMbjfnI_frame000601", "ND_rerYMXCrHSs_frame005084", "ND_PASZk6s-7cs_frame012901",
    #                 "ND_Ppu4AIAS30Y_frame014251", "AR_9HvbAaq670Y_12_270_45", "ND_RmG8z9Tl8V8_frame004501", "CC_000000441824", "EK_0002_P06_101_frame_0000005967", 'ND_pQ-vvPhpAmE_frame012901']

    # image_lists = [x+".jpg\n" for x in image_lists]

    for img in images[:1000]:
        image_lists.append(img)

    
    #predict('/home/evacheng/public_html/index.html/results/400k_test/', 0.3, 0.8, 0.3, 0.3, 0.5, 0.3, image_lists)

    root_dir = '/home/evacheng/public_html/index.html/results/400k_final/'

    os.makedirs(root_dir, exist_ok=True)

    # for hand_rela in [0.01, 0.1, 0.3, 0.5, 0.7]:
    #     for obj_rela in [0.01, 0.1, 0.3, 0.5, 0.7]:
    #         mess = "obj_0.3_hand_rela_"+str(hand_rela)+"_obj_rela_"+str(obj_rela)
    #         save_dir = os.path.join(root_dir, mess)
    #         predict(save_dir, 0.3, 0.8, 0.3, 0.3, hand_rela, obj_rela, image_lists)

    hand_rela = 0.3
    obj_rela = 0.3

    mess = "obj_0.3_hand_rela_"+str(hand_rela)+"_obj_rela_"+str(obj_rela)
    save_dir = os.path.join(root_dir, mess)
    predict(save_dir, 0.3, 0.8, 0.3, 0.3, hand_rela, obj_rela, image_lists)
    
    
    

if __name__ == '__main__':
    main()
