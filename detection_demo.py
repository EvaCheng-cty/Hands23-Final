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


hand_bbox_color = (255,0,0)
object_bbox_color = (0,0, 255)
second_object_bbox_color = (255, 200, 0)

hand_object_relation_color = (0, 255, 0)
tool_second_object_relation_color = (255, 80, 255)

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

def blendMask(I,mask,color, alpha):
    for c in range(3):
        Ic = I[:,:,c]
        Ic[mask] = ((Ic[mask].astype(np.float32)*alpha) + (float(color[c])*(1-alpha))).astype(np.uint8)
        I[:,:,c] = Ic



def find_idx(bbox_removed, bbox_list, pred_classes):
    # pdb.set_trace()
    ious = torchvision.ops.box_iou(bbox_removed, bbox_list)
    idx = torch.where(ious>=0.3)[1]

    if idx.shape[-1] == 1:
        return  idx[0]
    else:
        for i in idx:
            if pred_classes[i] == 1:
                return i


def deal_img(im, predictor, clean, file, id, separate, mess, save_dir = ""):

    second_obj = None

    outputs = predictor(im)

    if outputs == "None":
        return
        
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

    inter_prob = torch.tensor(pred_dz[:, 10:])

    t = inter_prob.clone()

    
   
    # print('&&&&&&&&&&&&&&&&&&&&')
    # print(pred_classes)
    # print(scores)
    # print(interaction)
    # print(pred_scores)

    if len(np.where(pred_classes == 2)[0]) !=0:
       q = open(second_objs_dir, "w+")
       lines = q.readlines()
       lines.append(id)
       q.writelines(lines)
    

    if clean:
        pred_hands = pred_boxes.copy()
        pred_hands[:,:] = 0
        pred_objects = pred_hands.copy()
        pred_second = pred_hands.copy()
       
        for i in range(len(scores)):
                if pred_classes[i] == 0:
                    pred_hands[i] = pred_boxes[i,:]
                elif pred_classes[i] == 1:
                    pred_objects[i] = pred_boxes[i,:]
                else:
                    pred_second[i] = pred_boxes[i,:]
       
    
   
        idx_hands = torch.ops.torchvision.nms(torch.tensor(pred_hands), torch.tensor(pred_scores), 0.3)
        idx_objects = torch.ops.torchvision.nms(torch.tensor(pred_objects), torch.tensor(pred_scores), 0.3)
        


        idx = []
        for i in idx_hands:
            if pred_classes[i] == 0:
                idx.append(i)
        for i in idx_objects:
            if pred_classes[i] == 1:
                idx.append(i)
        for i in range(len(pred_classes)):
            if pred_classes[i] == 2:
                idx.append(i)
        
        idx, temp = (torch.tensor(idx)).sort()


        if idx.shape[0] != scores.shape[0]:

            interaction = interaction[idx]
            
            try:
                for i in range(interaction.shape[0]):
                    if interaction[i]>=0:
                        new_idx = torch.where(idx == interaction[i])[0]
                        interaction[i] = new_idx[0] if new_idx.shape[0]>0 else find_idx(torch.tensor(pred_boxes[int(interaction[i])]).reshape(1,-1),torch.tensor(pred_boxes[idx]), pred_classes[idx])
            except Exception as e:
                pdb.set_trace()
            pred_boxes = pred_boxes[idx]
            pred_dz = pred_dz[idx]
            pred_classes =  pred_classes[idx]
            scores = scores[idx]
            pred_scores = pred_scores[idx]
            inter_prob = inter_prob[idx]
            hand_side = hand_side[idx]
            contact_state = contact_state[idx]
            touch_type = touch_type[idx]
            grasp= grasp[idx]
            
            temp = [None]*len(idx)
            for i in range(len(idx)):
                try:
                    temp[i] = inter_prob[i][idx]
                except:
                    pdb.set_trace()
                
            inter_prob = temp
        

        for i in range(len(idx)):
            for j in range(len(idx)):
                try:
                    assert t[idx[i]][idx[j]] == inter_prob[i][j]
                except:
                    pdb.set_trace() 
            
          
       
    try:
        
        L = len(pred_classes)
    except:
        L = 1
        pred_classes = [pred_classes]
    
    touch = ["N/A"]*L

    for i in range(L):
        if pred_classes[i] == 1:
            if touch_type[i] == 0:
                touch[i] = "tool_,_touched"
            elif touch_type[i] == 1:
                touch[i] = "tool_,_held"
            elif touch_type[i] == 2:
                touch[i] = "tool_,_used"
            elif touch_type[i] == 3:
                touch[i] = "container_,_touched"
            elif touch_type[i] == 4:
                touch[i] =  "container_,_held"
            elif touch_type[i] == 5:
                touch[i] = "neither_,_touched"
            elif touch_type[i] == 6:
                touch[i] = "neither_,_held"
            else:
                print("error!")
                pdb.set_trace()
        


    W = 5
    infos = np.ndarray((L,W))
    infos[:,0] = pred_classes
    infos[:,1] = interaction
    infos[:,2] = scores
    infos[:,3] = pred_scores
    infos[:,4] = [int(x) for x in touch_type]

    if 0 not in pred_classes:
        return
    
  

    num_boxes = len(infos)
    infos = [str(x).replace('\n','').replace('[','').replace(']', '\n') for x in infos]
    probs = [str(x).replace('\n','').replace('[','').replace(']', '\n') for x in inter_prob]

    infos = infos+probs

    infos.insert(0, str(num_boxes) + '\n')

    img_save_name = mess+'@'+id if mess !='' else id
   
    infos.insert(0,img_save_name + '\n')
  

    assert(len(infos) == 2*num_boxes+2)

    file.writelines(infos)


    color = None

    draw = False
   
    for i in range(len(pred_classes)):
        
        if pred_classes[i] == 0:
            curr_box = pred_boxes[i]
            color = left_hand_color if hand_side[i] == 0 else right_hand_color
            draw = True

            contact = ''

            if contact_state[i] == 0:
                contact = 'no'
            elif contact_state[i] == 1:
                contact = 'other'
            elif contact_state[i] == 2:
                contact = 'self'
            elif contact_state[i] == 3:
                contact = 'obj'
            else:
                if contact_state[i] != -1:
                    # pdb.set_trace()
                    pass
            
            im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), color , 2)   
            cv2.putText(im, str(i), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
            cv2.putText(im, "L" if hand_side[i]==0 else "R", (int(curr_box[0])+10, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
            cv2.putText(im, str(tell_grasp(grasp[i])) + " " + str(round(pred_scores[i],2)), (int(curr_box[0])+20, int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
            cv2.putText(im, contact, (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 
        
            if interaction[i] != -1:
                assert pred_classes[int(interaction[i])] == 1 
                object_box = pred_boxes[int(interaction[i])]
                im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), color, 2)

                
                cv2.putText(im, str(round(scores[i].item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
               
        elif pred_classes[i] == 1:
            curr_box = pred_boxes[i]
            color = first_obj_color

            

            # only draw object bounding boxes when 1)it is in interaction with hands 2) it is a tool/container 3)it touches a second object
            #if torch.where(interaction == i)[0].shape[0] !=0 or touch_type[i] <5 or interaction[i] != -1:

            if torch.where(interaction == i)[0].shape[0] !=0 or interaction[i] != -1:



                draw = True
            
                im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), color , 2)   
                cv2.putText(im, str(i), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
                cv2.putText(im, touch[i] + " " + str(round(pred_scores[i],2)), (int(curr_box[0]), int(curr_box[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 164, 0), 2) 

                if interaction[i] != -1:
                    try:
                        assert pred_classes[int(interaction[i])] == 2
                    except:
                        pdb.set_trace()
                    object_box = pred_boxes[int(interaction[i])]
                    im = cv2.line(im, (int((curr_box[0]+curr_box[2])/2), int((curr_box[1]+curr_box[3])/2)),(int((object_box[0]+object_box[2])/2), int((object_box[1]+object_box[3])/2)), color, 2)

                    cv2.putText(im, str(round(scores[i].item(),2)), (int((curr_box[0]+curr_box[2]+object_box[0]+object_box[2])/4), int((curr_box[1]+curr_box[3]+object_box[1]+object_box[3])/4)),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        elif pred_classes[i] == 2:

            if torch.where(interaction == i)[0].shape[0] !=0:
                curr_box = pred_boxes[i]
                color = second_object_bbox_color

                draw = True

                im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), second_object_bbox_color , 2)   
                cv2.putText(im, str(i) + " " + str(round(pred_scores[i],2)), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_object_bbox_color, 2) 


        if draw:
            blendMask(im, pred_masks[i], color, 0.65)
        
        draw = False

    # pdb.set_trace()

    if separate:
        dir = save_dir+ 'index.html/'+ id.replace('.jpg', '/')
        os.makedirs(dir, exist_ok=True)
        dir = dir + img_save_name.replace('\n','')
        cv2.imwrite(dir, im)
        # pdb.set_trace()
    else:
        cv2.imwrite(save_dir+id.replace(".jpg","")+mess+".jpg", im)
        # pdb.set_trace()
    
    return second_obj
    

def deal_obj_detect(im, predictor, file_dir, id):
    outputs = predictor(im)
   

    pred_classes =  outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()


    for i in range(len(pred_classes)):
        curr_box = pred_boxes[i]
        if pred_classes[i] == 0:
            if pred_scores[i] < 0.85:
                continue
            color = hand_bbox_color
        elif pred_classes[i] == 1:
            if pred_scores[i] < 0.7:
                continue
            color = object_bbox_color
        else:
            print(pred_boxes[i])
            print(np.where(pred_masks[i] == False))
            color = second_object_bbox_color
        im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), color , 2) 
        cv2.putText(im, str(i)+" "+str(pred_scores[i]), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

        blendMask(im, pred_masks[i], color, 0.5)

        cv2.imwrite(file_dir+id, im)

def deal_mask(im, predictor, file_dir, id):
    outputs = predictor(im)
   

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
    pred_classes =  outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()

   

    interaction = torch.tensor(pred_dz[:, 4])

    deal_obj_detect(im, predictor, file_dir, id)
    
    count = 0

    is_saved = [-1]*len(pred_classes)

    #pdb.set_trace()
    for i in range(len(pred_classes)):
        curr_box = pred_boxes[i]
        if pred_classes[i] == 0:
            if pred_scores[i] < 0.85:
                continue
            # color = hand_bbox_color

            im[:,:,:] = 0
            im[pred_masks[i], :] = 255

            cv2.imwrite(file_dir+"2_"+str(count)+"_"+id, im)

            if interaction[i] >=0 and is_saved[interaction[i]] < 0:
                im[:,:,:] = 0
                im[pred_masks[interaction[i]], :] = 255

                cv2.imwrite(file_dir+"3_"+str(count)+"_"+id, im)

                is_saved[interaction[i]] = count
            
            count = count+1

        elif pred_classes[i] == 1:
            if pred_scores[i] < 0.7:
                continue
            
            if interaction[i] >=0:
                im[:,:,:] = 0
                im[pred_masks[interaction[i]], :] = 255



                row_num = count if is_saved[i]<0 else is_saved[i]

                cv2.imwrite(file_dir+"5_"+str(row_num)+"_"+id, im)

                if is_saved[i] == False:
                    im[:,:,:] = 0
                    im[pred_masks[i], :] = 255
                    cv2.imwrite(file_dir+"3_"+str(row_num)+"_"+id, im)

                    is_saved[i] = count

            
                count = count+1






        # else:
        #     print(pred_boxes[i])
        #     print(np.where(pred_masks[i] == False))
        #     color = second_object_bbox_color
        # im = cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), color , 2) 
        # cv2.putText(im, str(i)+" "+str(pred_scores[i]), (int(curr_box[0]), int(curr_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

        # blendMask(im, pred_masks[i], color, 0.5)

        # cv2.imwrite(file_dir+id, im)




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


    return
def visualize_official(im, predictor, out_path):
    # im = cv2.imread(img_path)
    try:

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        cv2.imwrite(out_path, result[:, :, ::-1])
    except Exception as e:
        # pdb.set_trace()
        print(e)



def write_compare_model(file_dir, model_list, img_list, message = ""):
   file_dir_ = file_dir + '/index.html'
   f = open(file_dir_, "w+")
   f.truncate(0)
  
  
   html = ET.Element('html')
   body = ET.SubElement(html, 'body')

   h2_2 = ET.SubElement(body, 'h2')
   h2_2.text = "-" + message

   table = ET.SubElement(body, "table")
   table.set("width", "100%")
   table.set("border", "1px solid black")
   table.set("border-collapse", "collapse")

   
   tr = ET.SubElement(table, "tr")
   for t in model_list:
       td = ET.SubElement(tr, "th")
       td.text = str(t)

       td.set("width", "30%")
       td.set("border", "1px solid black")
       td.set("border-collapse", "collapse")
       
    
   for i in range(len(img_list)):
       tr = ET.SubElement(table, "tr")
 
       for t in model_list:
           
           if os.path.exists(file_dir+str(img_list[i]).replace("\n", "").replace(".jpg","") + t +".jpg"):
                tb = ET.SubElement(tr, "td")

                img = ET.SubElement(tb, "image")
                img.set("src", "./" + str(img_list[i]).replace("\n", "").replace(".jpg","") + t +".jpg")
                
                img.set("width", "30%")
                img.set("border", "1px solid black")
                img.set("border-collapse", "collapse")
           else:
               print(str(img_list[i]).replace("\n", "").replace(".jpg","") + t +".jpg")
       
     
   tree = ET.ElementTree(html)
   ET.indent(tree, space='\t', level=0)
   tree.write(open(file_dir_, 'wb'))




def main():

    arguments = sys.argv
    thresh_val = 0.0001

    if 'thresh' in arguments:
        idx = arguments.index('thresh')
        thresh_val = float(arguments[idx+1])

    
    # load cfg and model
    cfg = get_cfg()
    cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    # cfg.MODEL.WEIGHTS = '/y/evacheng/new_mode/model_3199999.pth'
    # cfg.MODEL.WEIGHTS = "/y/evacheng/model_2023_test/model_0069999.pth"
    # cfg.MODEL.WEIGHTS = "/y/evacheng/new_mode/model_5099999.pth"
    #cfg.MODEL.WEIGHTS = "/y/evacheng/all_head_model_weights/model_0281999.pth" 
    cfg.MODEL.WEIGHTS = "/y/evacheng/weights/ptRend_all_lr_0.01_relation_0.1_other_0.1_new_grasp_0054999.pth"
    #cfg.MODEL.WEIGHTS = "/y/evacheng/together_mar_01/model_0429999.pth"
    #cfg.MODEL.WEIGHTS = "/y/evacheng/_500_trained_on_EK/model_0139999.pth" #overfit on 500 EK images
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_val  # 0.5 , set the testing threshold for this modeli
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 3000

    cfg.MYNAME = 0.3

    pdb.set_trace()


    
    second_obj_list = []

    clean = False

    if "clean" in arguments:
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        clean = True

    separate = "separate" in arguments

    mess = ''

    if "-m" in arguments:
        idx = arguments.index('-m')
        mess = arguments[idx+1]

    

    # data path
    demo_dir = '/y/evacheng/100_images/JPEGImages/'
    save_dir = '/home/evacheng/public_html/'
    image_dir = '/y/evacheng/100DOH/JPEGImages/'
    f = open('/y/evacheng/100DOH/ImageSets/Main/test.txt')

    debug_info_dir = '/home/evacheng/hand_detector.d2-master/Model_2023/debug_info.txt'
    file = open(debug_info_dir, 'w+')

    os.makedirs(save_dir, exist_ok=True)
    
    predictor = DefaultPredictor(cfg)

    if "COCO" in arguments:

        for test_img in os.listdir(demo_dir):

            id = test_img
            test_img = os.path.join(demo_dir, id)
            im = cv2.imread(test_img)

            deal_img(im, predictor, clean, file, id, separate, mess)
            # pdb.set_trace()
           

    elif "test" in arguments:
        #  img_file = open("/y/evacheng/COCO_split/test.txt", "r")
         img_file = open("/w/fouhey/hands2/allMerged6Splits/VAL.txt", "r")
         #img_file = open("/y/evacheng/allMerged6Splits/TRAIN.txt", "r")

         #img_file = open('/y/evacheng/5000_COCO_split.txt', "r")

         #img_file = open("/y/evacheng/data_prep_handv2/more.txt", "r")

         #img_file = open("/y/evacheng/500EKSplits/TEST.txt")
         images = img_file.readlines()

         images = sorted(images)

         random.seed(42)

         random.shuffle(images)
         
        #  images = images[:500]
         images = images[:1]

         g = open("./images_list.txt", "w+")
         g.writelines(images)

        #  count = 0

        #  pdb.set_trace()
        
         for test_img in images:
            im = cv2.imread("/w/fouhey/hands2/allMerged6Blur/"+test_img.replace('\n', ''))

            
            id = test_img.replace('\n', '')

            second = None

            thresh = "0.01"
           

            try:
                if "obj" in arguments:
                    #file_dir = "/home/evacheng/public_html/index.html/COCO5KVal183K_0.5_0.3_0.8_relation/"

                    file_dir = "/home/evacheng/public_html/index.html/ValPtRend55k_compare/"
                    # visualize_official(im, predictor, file_dir+id)
                    #deal_obj_detect(im, predictor, file_dir, id)

                    deal_img(im, predictor, clean, file, id, separate, "-"+thresh, file_dir)
                    #write_obj_html(image_dir=file_dir)
                elif "mask" in arguments:
                    file_dir = "/home/evacheng/public_html/index.html/masks_249k_0.8_0.5_0.3/"
                    deal_mask(im, predictor, file_dir, id)
                    write_obj_html(message = "PtRend trained from scratch with base lr 0.01 and all losses scaled by 0.1",image_dir=file_dir)
                
                # elif "COCO" in arguments:

                #     write_obj_html


                # else: 
                #     second = deal_img(im, predictor, clean, file, id, separate, mess)

                # if second != None:
                #     second_obj_list.append(second)
            except Exception as e:
                # pdb.set_trace()
                print(e)
            
            # if count >500:
            #     print(second_obj_list)
            #     if "obj" in arguments:
            #         return
            #     break

            # count = count +1


    elif "train" in arguments:
         img_file = open("/y/evacheng/COCO_split/train.txt", "r")

         
         images = img_file.readlines()
         random.shuffle(images)

         count = 0
        
         for test_img in images:
            im = cv2.imread("/w/fouhey/hands2/allMerged4/"+test_img.replace('\n', '') + ".jpg")
            id = test_img.replace('\n', '')+".jpg"

            try:
                deal_img(im, predictor, clean, file, id, separate, mess)
            except:
                print(test_img)
            
            if count >200:
                break

            count = count +1
    
    elif "blur" in arguments:
         demo_dir = "/w/fouhey/hands2/allMerged4Blur"
         count = 0

         for test_img in os.listdir(demo_dir):

            id = test_img
            test_img = os.path.join(demo_dir, id)
            im = cv2.imread(test_img)

            deal_img(im, predictor, clean, file, id, separate, mess)
            if count >= 1000:
                break
            if count == 0:
                print(id)
            count = count+1
    elif "split" in arguments:

        for file in glob.glob("/y/evacheng/data_prep_handv2/*.txt"):

            if "5000" in file:
                continue

            img_file = open(file, "r")

            
            images = img_file.readlines()

            # pdb.set_trace()

            for test_img in images:
                im = cv2.imread("/w/fouhey/hands2/allMerged6Blur/"+test_img.replace('\n', ''))
                id = test_img.replace('\n', '')

                try:
                 
                        file_dir = "/home/evacheng/public_html/index.html/VAL_" + file.replace("/y/evacheng/data_prep_handv2/", "").replace(".txt", "")
                        deal_img(im, predictor, clean, file, id, separate, mess, file_dir)
                        write_obj_html(image_dir=file_dir)
                    
                except Exception as e:
                    # pdb.set_trace()
                    print(e)
            
        
    else:

        for test_img in f.readlines():
    
            id = test_img.replace('\n','')+'.jpg'
            test_img = os.path.join(image_dir, id)
        
            im = cv2.imread(test_img)

            deal_img(im, predictor, clean, file, id, separate, mess)

    # file.close()
   

    
    # message = '' if len(arguments) == 1 else arguments[-1]
    # if "together" in arguments:
    #     write_html(message)

    # write_separate_html(debug_info_dir, message+" training dataset")

     
    

if __name__ == '__main__':
    main()
