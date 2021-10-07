from tkinter import *
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import math
import sys
import tensorflow as tf

#Window 
root=Tk()
root.geometry("1280x1024")

#Detection
MODEL_NAME = 'inference_graph'
IMAGE_NAME = '1.jpg'

tf.gfile = tf.io.gfile
sess = tf.compat.v1.Session()

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `cavity`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef() 
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#Icon
root.title('Dental Cavity Detection' )
program_directory=sys.path[0]
root.iconphoto(True, PhotoImage(file=os.path.join(program_directory, "logo.png")))

#Background
b1=Frame(root,width=1280,height=250,bg="linen")
b1.grid(column=0,row=0)
b2=Frame(root,width=1280,height=315,bg="bisque")
b2.grid(column=0,row=1)
b3=Frame(root,width=1280,height=310,bg="gray88")
b3.grid(column=0,row=2)
b4=Frame(root,width=1280,height=100,bg="salmon")
b4.grid(column=0,row=3)

#Text
t1=Label(root,text="Dental Cavity Detection",bg="linen",fg="coral",font=("Comic Sans MS",65))
t1.place(x=260,y=55)
tend1=Label(root,text="by MASH",bg="salmon",font=("Arial",10))
tend1.place(x=1210,y=910)
tend1=Label(root,text="© 2017-2021 Dental Cavity Detection.All Rights Reserved",bg="salmon",font=("Arial",10))
tend1.place(x=5,y=890)

#Logo
logo=Image.open('logo.png')
logo=ImageTk.PhotoImage(logo)
logo_label=Label(image=logo,bg="linen")
logo_label.image=logo
logo_label.grid(column=0,row=0,sticky='nw')

def next():
    global count
    count +=1
    if count>len(List):
        count=0 #Restart count from zero if all images in list loaded
    print(default+'/'+List[count])
    image =Image.open(default+'/'+List[count])
    r = image.resize((510,260))
    photo=ImageTk.PhotoImage(r)
    l1.configure(image=photo)
    l1.image=photo
    
count=0
def back():
    global count
    count -=1 # backwards the list items
    if count <0: #if list is less then 0 means becomes empty then leght of list minus from 1
        count= len(List)-1
    image =Image.open(default+'/'+List[count])
    r = image.resize((510,260))
    photo=ImageTk.PhotoImage(r)
    l1.configure(image=photo)
    l1.image=photo 

def gamma():
    global count
    image=cv2.imread(default+'/'+List[count])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma = 1.5
    gamma_corrected = np.array(255*(gray / 255) ** gamma, dtype = 'uint8')    
    im = Image.fromarray(gamma_corrected)
    r = im.resize((510,260))
    photo=ImageTk.PhotoImage(r)
    l2.configure(image=photo)
    l2.image=photo
    
def clahe():
    global count
    image=cv2.imread(default+'/'+List[count])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma = 1.5
    gamma_corrected = np.array(255*(gray / 255) ** gamma, dtype = 'uint8')
    cl = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    cl_img = cl.apply(gamma_corrected)
    cv2.imwrite('/home/pi/Desktop/Dental Cavity Detection/object_detection/write/'+List[count],cl_img)
    im = Image.fromarray(cl_img)
    r = im.resize((510,260))
    photo=ImageTk.PhotoImage(r)
    l3.configure(image=photo)
    l3.image=photo
    
def detection():
    global count
    image1=cv2.imread('/home/pi/Desktop/Dental Cavity Detection/object_detection/write/'+List[count])
    image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image1,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.30)
    
    im = Image.fromarray(image1)
    r = im.resize((510,260))
    photo1=ImageTk.PhotoImage(r)
    l4.configure(image=photo1)
    l4.image=photo1

nextbutton=Button(root,text="Next",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=next)
nextbutton.place(x=5,y=260)
backbutton=Button(root,text="Back",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=back)
backbutton.place(x=5,y=360)
t2=Label(root,text="Gray Scale Image/Current Image",bg="bisque",fg="coral",font=("Comic Sans MS",15))
t2.place(x=330,y=530)
gammabutton=Button(root,text="Gamma\nCorrection",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=gamma)
gammabutton.place(x=5,y=460)
t3=Label(root,text="Gamma Corrected Image",bg="bisque",fg="coral",font=("Comic Sans MS",15))
t3.place(x=890,y=530)
clahebutton=Button(root,text="CLAHE",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=clahe)
clahebutton.place(x=5,y=575)
t4=Label(root,text="CLAHE",bg="gray88",fg="coral",font=("Comic Sans MS",15))
t4.place(x=450,y=840)
detectionbutton=Button(root,text="Cavity\nDetection",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=detection)
detectionbutton.place(x=5,y=675)        
t5=Label(root,text="Cavity Detection",bg="gray88",fg="coral",font=("Comic Sans MS",15))
t5.place(x=940,y=840)
Exitbutton=Button(root,text="Exit",width=15,height=2,font=("Franklin Gothic Demi",15),bg="light slate gray",command=root.destroy)
Exitbutton.place(x=5,y=775)


default=r'/home/pi/Desktop/Dental Cavity Detection/object_detection/xrays0'
exe=('jpg','png','jpeg')
List=[file for file in os.listdir(default) if file.endswith(exe)]
image=Image.open(default+'/'+List[0])
resize_image = image.resize((510,260))
photo=ImageTk.PhotoImage(resize_image)
    
l1=Label(image=photo)
l1.place(x=235,y=260) 
l2 = Label(image=photo)
l2.place(x=760,y=260)
l3 = Label(image=photo)
l3.place(x=235,y=575)
l4 = Label(image=photo)
l4.place(x=760,y=575)
    
root.mainloop()