from code import interact
from email.quoprimime import body_check
import glob
import numbers
from tkinter import image_names
from turtle import numinput
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET
import pdb
import os
import sys
import copy
import torch
import json

# image_dir = "/home/evacheng/public_html/index.html.folder"
# image_dir = '/y/evacheng/output_curr_model/'
image_dir = '/home/evacheng/public_html/'
file_dir = "/home/evacheng/public_html/index.html/"
debug_file = '/home/evacheng/hand_detector.d2-master/Model_2023/debug_info.txt'

def tell_touch(touch_type):

    if touch_type == '0.':
         return "tool_,_touched"
    elif touch_type == '1.':
         return "tool_,_held"
    elif touch_type == '2.':
         return "tool_,_used"
    elif touch_type == '3.':
         return "container_,_touched"
    elif touch_type == '4.':
         return  "container_,_held"
    elif touch_type == '5.':
         return "neither_,_touched"
    elif touch_type == '6.':
         return "neither_,_held"
    else:
         return "N/A"

def write_separate_html(debug_file = debug_file, message = ''):
  
   de_f = open(debug_file, 'r')
   infos = de_f.readlines()

   prev_im_id = ''
   next_im_id = ''
   
   
   count = 0
   pdb.set_trace()
   while count < len(infos):
      
      image_name = infos[count].replace('\n', '').replace(')','')
      img_name = image_name.split('@')[-1]
      # mess = "" if len( image_name.split('-')[-1]) == 1 else  image_name.split('-')[-2]
      dir = file_dir+ img_name.replace('.jpg', '/') + img_name.replace('.jpg', '.html')
      # os.makedirs(dir, exist_ok=True)

      body = ET.Element('html')
      xml_dir = dir.replace('html', 'xml').replace('xml/', 'html/')

   
    
      try:
         f = open(dir, 'x')
         xml_file = open(xml_dir, 'x')
         html = ET.Element('html')
         body = ET.SubElement(html, 'body')
         tree =  ET.ElementTree(html)
 
         if prev_im_id != '':
            prev_im = ET.SubElement(body, 'form')
            prev_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/index.html/"+prev_im_id +'/'+prev_im_id+'.html')
            input = ET.SubElement(prev_im, "input")
            input.set("type", "submit")
            input.set("value", "Go to previous image")
         
       

         try:
            
            count_next = count + 2*int(infos[count+1]) +2
            next_image_name = infos[count_next].replace('\n', '')
            next_img_id = next_image_name.split('@')[-1]
            next_img_id = next_img_id.replace(".jpg", "")


            next_im = ET.SubElement(body, 'form')
            next_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/index.html/"+next_img_id +'/'+next_img_id+'.html')
            input = ET.SubElement(next_im, "input")
            input.set("type", "submit")
            input.set("value", "Go to next image")
         except:
            
            pass

      except:
         
         
         f = open(dir, 'w')
         
         pdb.set_trace()

        
         html = ET.parse(xml_dir)

         c = 0

         for elems in html.iter():
            if elems.tag == "body":
               body = elems
               c = c+1
            
            if elems.tag == "form":
               if c == 1:
                  prev_im = elems
                  c = c + 1
               else:
                  next_im = elems
                  break
      
      
      text = ET.SubElement(body, 'p')
      text.text = img_name + "   " + message

      table = ET.SubElement(body, 'table')
      table.set("style", "width:100%")
         
      tr1 = ET.SubElement(table, 'tr')
      td = ET.SubElement(tr1, 'th')
      td.text = "bbox"
      tr2 = ET.SubElement(table, 'tr')
      td = ET.SubElement(tr2, 'td')
      td.text = "pred classes"
      tr3 = ET.SubElement(table, 'tr')
      td = ET.SubElement(tr3, 'td')
      td.text = "interactions"
      tr4 = ET.SubElement(table, 'tr')
      td = ET.SubElement(tr4, 'td')
      td.text = "scores"
      tr5 = ET.SubElement(table, 'tr')
      td = ET.SubElement(tr5, 'td')
      td.text = "bbox_score"
      # tr6 = ET.SubElement(table, 'tr')
      # td = ET.SubElement(tr6, 'td')
      # td.text = "touch type"

      

      num_boxes = int(infos[count+1])

      hand_idx = []
      object_idx = []
      


      for i in range(num_boxes):
         try:
            vals = infos[count+2+i].split()
         except:
            pdb.set_trace()

         pred_class = vals[0]
         interactions = vals[1]
         scores = vals[2]
         bbox_scores = vals[3]
         touch_type = vals[4]

         th = ET.SubElement(tr1, 'th')
         th.text = "bbox " + str(i)
         td = ET.SubElement(tr2, 'td')
         td.text = pred_class
         td = ET.SubElement(tr3, 'td')
         td.text = str(interactions) + " " + str(tell_touch(touch_type))
         td = ET.SubElement(tr4, 'td')
         td.text = scores
         td = ET.SubElement(tr5, 'td')
         td.text = bbox_scores
         # td = ET.SubElement(tr6, 'td')
         # td.text = tell_touch(touch_type)


         if pred_class == "0.":
            hand_idx.append(i)
         else:
            object_idx.append(i)
      
      
      img = ET.SubElement(body, 'img')
      img.set("src", image_name)
      img.set("alt", "separate model output")
      

      text = ET.SubElement(body, 'p')
      text.text = "Interaction score between hands(rows) and objects(columns)"
      inter_table = ET.SubElement(body, 'table')
      inter_table.set("style", "width:100%")
      
      
      H = len(hand_idx)
      W = len(object_idx)

    
      temp1 = infos[count+2+num_boxes:count+2+num_boxes+num_boxes]
      temp2 = [x.split() for x in temp1]
      temp3 = [temp2[x] for x in hand_idx]
      vals = []
      for i in range(H):
         row = [temp3[i][j] for j in object_idx]
         vals.append(row)



      tr1 = ET.SubElement(inter_table, 'tr')
      td = ET.SubElement(tr1, 'th')
      td.text = "bbox"
      for i in range(W):
         td = ET.SubElement(tr1, 'td')
         td.text =  "bbox "+str(object_idx[i])
     

      for i in range(H):
         tr = ET.SubElement(inter_table, 'tr')
         td = ET.SubElement(tr, 'th')
         td.text = "bbox "+str(hand_idx[i])

         for j in range(W):
            td = ET.SubElement(tr, 'td')
            td.text = vals[i][j]
            

        
      
      prev_im_id = img_name.replace(".jpg", "")

     
      
      
      tree = ET.ElementTree(html)
      try:
         ET.indent(tree, space='\t', level=0)
      except:
         tree = html
         ET.indent(tree, space='\t', level=0)
         

    
     
      tree.write(open(dir, 'wb'))
      tree.write(open(xml_dir, 'wb'))
      
      count = count + 2*num_boxes +2
    





def write_html(message = ''):
   f = open(file_dir, "w+")
   f.truncate(0)
    # f.close()
   de_f = open(debug_file, 'r')
    

   html = ET.Element('html')
   body = ET.SubElement(html, 'body')

   h2_2 = ET.SubElement(body, 'h2')
   h2_2.text = "Output from model with iou, more mlp hidden layers, more positional information, more topk"+message

   infos = de_f.readlines()



   for file in  glob.glob(image_dir + "/*.jpg"):
   
      image = file.replace(image_dir, '')

      
      
      text = ET.SubElement(body, 'p')
      text.text = image

      try:
         idx = infos.index(image+ '\n')
         length = int(infos[idx+1])

      #    pred_class = str(infos[idx+1])[1:-2].split()
      # #    pred_class = [int(x) for x in pred_class]
      #    interactions = str(infos[idx+2])[1:-3].split()
      # #    interactions = [int(x) for x in interactions]
      #    scores = str(infos[idx+3])[1:-3].split()
      # #    scores = [float(x) for x in scores]c

         table = ET.SubElement(body, 'table')
         table.set("style", "width:100%")
         
         tr1 = ET.SubElement(table, 'tr')
         td = ET.SubElement(tr1, 'th')
         td.text = "bbox"
         tr2 = ET.SubElement(table, 'tr')
         td = ET.SubElement(tr2, 'td')
         td.text = "pred classes"
         tr3 = ET.SubElement(table, 'tr')
         td = ET.SubElement(tr3, 'td')
         td.text = "interactions"
         tr4 = ET.SubElement(table, 'tr')
         td = ET.SubElement(tr4, 'td')
         td.text = "scores"
         tr5 = ET.SubElement(table, 'tr')
         td = ET.SubElement(tr5, 'td')
         td.text = "bbox_score"

         for i in range(length):
            vals = infos[idx+2+i].split()

            pred_class = vals[0]
            interactions = vals[1]
            scores = vals[2]
            bbox_scores = vals[3]
            th = ET.SubElement(tr1, 'th')
            th.text = "bbox " + str(i)
            td = ET.SubElement(tr2, 'td')
            td.text = pred_class
            td = ET.SubElement(tr3, 'td')
            td.text = interactions
            td = ET.SubElement(tr4, 'td')
            td.text = scores
            td = ET.SubElement(tr5, 'td')
            td.text = bbox_scores

         

         # tr5 =  ET.SubElement(table, 'tr')
         # td5 = ET.SubElement(tr5, 'td')
         img = ET.SubElement(body, 'img')
         # img.set("src", file.replace("/y/evacheng/output_curr_model", "."))
         img.set("src", "../"+file.replace(image_dir, ""))
         img.set("alt", "separate model output")
      except Exception as e:
         print(image)
       
       
     
   tree = ET.ElementTree(html)
   ET.indent(tree, space='\t', level=0)
   tree.write(open(file_dir, 'wb'))
    
    # f = open(file_dir, "w+")
    # f.write('\n')
    # f.close()


def write_obj_html(message = '', image_dir = "", thresh_list = [], root_dir = ''):

   # file_dir = "/home/evacheng/public_html/index.html/7/index.html"
   file_dir = image_dir + '/index.html'
   f = open(file_dir, "w+")
   f.truncate(0)
  
   # image_dir = "/home/evacheng/public_html/index.html/7/"

   html = ET.Element('html')
   body = ET.SubElement(html, 'body')

   h2_2 = ET.SubElement(body, 'h2')
   h2_2.text = "-" + message

   if len(thresh_list) != 0:
      for hand_rela in thresh_list:
         for obj_rela in thresh_list:
            next_im = ET.SubElement(body, 'form')
            next_im.set("action", "https://epicfail.eecs.umich.edu/~evacheng/" + root_dir.replace("/home/evacheng/public_html/","")+ "obj_0.3_hand_rela_"+str(hand_rela)+"_obj_rela_"+str(obj_rela) + "/index.html") 
            input = ET.SubElement(next_im, "input")
            input.set("type", "submit")
            input.set("value", "obj_0.3_hand_rela_"+str(hand_rela)+"_obj_rela_"+str(obj_rela))



   # infos = de_f.readlines()



   for file in  glob.glob(image_dir + "/*.jpg"):
   
      image = file.replace(image_dir, '')

      
      
      text = ET.SubElement(body, 'p')
      text.text = image

      try:
         # idx = infos.index(image+ '\n')
         # length = int(infos[idx+1])

       
        
         img = ET.SubElement(body, 'img')
         img.set("src", "./"+file.replace(image_dir, ""))
         img.set("alt", "separate model output")
      except Exception as e:
         print(image)
       
       
     
   tree = ET.ElementTree(html)
   ET.indent(tree, space='\t', level=0)
   tree.write(open(file_dir, 'wb'))

def write_single_image(img, im_dir, prev_im_name, next_im_name):

      open(os.path.join(im_dir, img["image_id"]+".html"), "w+")

      html = ET.Element('html')
      body = ET.SubElement(html, 'body')


      if prev_im_name is not None:
         prev_im = ET.SubElement(body, 'form')
         prev_im.set("action", os.path.join(im_dir.replace(img["image_id"], prev_im_name), prev_im_name+".html") )
         input = ET.SubElement(prev_im, "input")
         input.set("type", "submit")
         input.set("value", "Go to previous image")
      
      if next_im_name is not None:
         next_im = ET.SubElement(body, 'form')
         next_im.set("action", os.path.join(im_dir.replace(img["image_id"], next_im_name), next_im_name+".html") )
         input = ET.SubElement(next_im, "input")
         input.set("type", "submit")
         input.set("value", "Go to next image")
      
      
      

      


 
   

def generate_html_from_json(json_dir):
   f = open(json_dir, "r")
   data = json.load(f)

   save_dir = data["save_dir"]
   images = data["images"]

   prev_im_name = images[-1]["img_id"]

   for i in range(len(images)):
      img = images[i]
      img_name = img["img_id"]
      im_dir = os.path.join(save_dir, "visualization", img_name)
      os.makedirs(im_dir, exist_ok= True)

      next_im_name = images[i+1]["img_id"] if i < len(images)-1 else images[0]["img_id"]

      write_single_image(img, im_dir, prev_im_name, next_im_name)




def main():
   # write_html()
  # write_separate_html()
  write_obj_html()
  


if __name__ == '__main__':
    main()